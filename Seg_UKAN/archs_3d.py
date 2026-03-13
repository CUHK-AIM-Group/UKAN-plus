"""3D U-KAN++ architecture for volumetric medical image segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from timm.models.layers import DropPath, trunc_normal_
from kan import KANLinear


class DWConv3D(nn.Module):
    """3D Depthwise Convolution with BN + ReLU."""
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm3d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, D, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PatchEmbed3D(nn.Module):
    """3D Patch Embedding using 3D Conv."""
    def __init__(self, vol_size=96, patch_size=3, stride=2, in_chans=64, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, D, H, W


class ConvLayer3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class D_ConvLayer3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class MSKANLayer3D(nn.Module):
    """Multi-scale KAN Layer (3D) with parallel branches at different grid resolutions."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 grid_sizes=[3, 5, 10], spline_order=3, fusion_type='softmax',
                 drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.num_branches = len(grid_sizes)
        self.fusion_type = fusion_type

        kan_kwargs = dict(
            spline_order=spline_order, scale_noise=0.1, scale_base=1.0,
            scale_spline=1.0, base_activation=torch.nn.SiLU,
            grid_eps=0.02, grid_range=[-1, 1],
        )

        self.branches = nn.ModuleList()
        for g in grid_sizes:
            if not no_kan:
                branch = nn.ModuleDict({
                    'fc1': KANLinear(in_features, hidden_features, grid_size=g, **kan_kwargs),
                    'fc2': KANLinear(hidden_features, out_features, grid_size=g, **kan_kwargs),
                    'fc3': KANLinear(hidden_features, out_features, grid_size=g, **kan_kwargs),
                })
            else:
                branch = nn.ModuleDict({
                    'fc1': nn.Linear(in_features, hidden_features),
                    'fc2': nn.Linear(hidden_features, out_features),
                    'fc3': nn.Linear(hidden_features, out_features),
                })
            branch['dw1'] = DWConv3D(hidden_features)
            branch['dw2'] = DWConv3D(hidden_features)
            branch['dw3'] = DWConv3D(hidden_features)
            self.branches.append(branch)

        if self.num_branches > 1:
            if fusion_type == 'softmax':
                self.fusion_proj = nn.Linear(out_features * self.num_branches, self.num_branches)
            elif fusion_type == 'concat':
                self.fusion_conv = nn.Linear(out_features * self.num_branches, out_features)

        self.drop = nn.Dropout(drop)

    def _forward_branch(self, branch, x, D, H, W):
        B, N, C = x.shape
        out = branch['fc1'](x.reshape(B * N, C)).reshape(B, N, -1).contiguous()
        out = branch['dw1'](out, D, H, W)
        out = branch['fc2'](out.reshape(B * N, -1)).reshape(B, N, -1).contiguous()
        out = branch['dw2'](out, D, H, W)
        out = branch['fc3'](out.reshape(B * N, -1)).reshape(B, N, -1).contiguous()
        out = branch['dw3'](out, D, H, W)
        return out

    def forward(self, x, D, H, W):
        if self.num_branches == 1:
            return self._forward_branch(self.branches[0], x, D, H, W)

        branch_outputs = [self._forward_branch(b, x, D, H, W) for b in self.branches]

        if self.fusion_type == 'softmax':
            concat = torch.cat(branch_outputs, dim=-1)
            alpha = F.softmax(self.fusion_proj(concat), dim=-1)
            fused = sum(alpha[..., i:i+1] * branch_outputs[i] for i in range(self.num_branches))
        elif self.fusion_type == 'concat':
            fused = self.fusion_conv(torch.cat(branch_outputs, dim=-1))
        elif self.fusion_type == 'sum':
            fused = sum(branch_outputs) / self.num_branches
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")
        return fused


class MSKANBlock3D(nn.Module):
    def __init__(self, dim, grid_sizes=[3, 5, 10], spline_order=3,
                 fusion_type='softmax', drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = norm_layer(dim)
        self.layer = MSKANLayer3D(
            in_features=dim, hidden_features=dim,
            grid_sizes=grid_sizes, spline_order=spline_order,
            fusion_type=fusion_type, drop=drop, no_kan=no_kan,
        )

    def forward(self, x, D, H, W):
        x = x + self.drop_path(self.layer(self.norm(x), D, H, W))
        return x


class UKANPP3D(nn.Module):
    """3D U-KAN++ with Multi-scale KAN Blocks for volumetric segmentation."""
    def __init__(self, num_classes, input_channels=1, embed_dims=[64, 128, 256],
                 grid_sizes=[3, 5, 10], fusion_type='softmax', no_kan=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], **kwargs):
        super().__init__()

        kan_input_dim = embed_dims[0]

        # Encoder conv stages
        self.encoder1 = ConvLayer3D(input_channels, kan_input_dim // 4)
        self.encoder2 = ConvLayer3D(kan_input_dim // 4, kan_input_dim // 2)
        self.encoder3 = ConvLayer3D(kan_input_dim // 2, kan_input_dim)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])
        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        ms_kwargs = dict(grid_sizes=grid_sizes, fusion_type=fusion_type, no_kan=no_kan)

        # Encoder KAN stages
        self.patch_embed3 = PatchEmbed3D(patch_size=3, stride=2,
                                         in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = PatchEmbed3D(patch_size=3, stride=2,
                                         in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.block1 = nn.ModuleList([MSKANBlock3D(
            dim=embed_dims[1], drop=drop_rate, drop_path=dpr[0],
            norm_layer=norm_layer, **ms_kwargs)])
        self.block2 = nn.ModuleList([MSKANBlock3D(
            dim=embed_dims[2], drop=drop_rate, drop_path=dpr[1],
            norm_layer=norm_layer, **ms_kwargs)])

        # Decoder KAN stages
        self.dblock1 = nn.ModuleList([MSKANBlock3D(
            dim=embed_dims[1], drop=drop_rate, drop_path=dpr[0],
            norm_layer=norm_layer, **ms_kwargs)])
        self.dblock2 = nn.ModuleList([MSKANBlock3D(
            dim=embed_dims[0], drop=drop_rate, drop_path=dpr[1],
            norm_layer=norm_layer, **ms_kwargs)])

        # Decoder conv stages
        self.decoder1 = D_ConvLayer3D(embed_dims[2], embed_dims[1])
        self.decoder2 = D_ConvLayer3D(embed_dims[1], embed_dims[0])
        self.decoder3 = D_ConvLayer3D(embed_dims[0], embed_dims[0] // 2)
        self.decoder4 = D_ConvLayer3D(embed_dims[0] // 2, embed_dims[0] // 4)
        self.decoder5 = D_ConvLayer3D(embed_dims[0] // 4, embed_dims[0] // 4)

        self.final = nn.Conv3d(embed_dims[0] // 4, num_classes, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]

        # Encoder conv stages
        out = F.relu(F.max_pool3d(self.encoder1(x), 2, 2))
        t1 = out
        out = F.relu(F.max_pool3d(self.encoder2(out), 2, 2))
        t2 = out
        out = F.relu(F.max_pool3d(self.encoder3(out), 2, 2))
        t3 = out

        # Tokenized MS-KAN stage 1
        out, D, H, W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, D, H, W)
        out = self.norm3(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        t4 = out

        # Bottleneck
        out, D, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, D, H, W)
        out = self.norm4(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        # Decoder
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=2, mode='trilinear', align_corners=False))
        out = torch.add(out, t4)
        _, _, D, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            out = blk(out, D, H, W)

        out = self.dnorm3(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=2, mode='trilinear', align_corners=False))
        out = torch.add(out, t3)
        _, _, D, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, D, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=2, mode='trilinear', align_corners=False))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=2, mode='trilinear', align_corners=False))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=2, mode='trilinear', align_corners=False))

        return self.final(out)
