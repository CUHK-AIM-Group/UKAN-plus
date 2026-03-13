"""U-KAN++ Hybrid Diffusion Model with Multi-scale KAN blocks."""

import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from Diffusion.Model_UKAN_Hybrid import (
    KANLinear, Swish, swish, TimeEmbedding, DownSample, UpSample,
    OverlapPatchEmbed, DW_bn_relu, SingleConv, DoubleConv,
    D_SingleConv, D_DoubleConv, AttnBlock, ResBlock,
)


class ms_kan(nn.Module):
    """Multi-scale KAN layer for diffusion models."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 grid_sizes=[3, 5, 10], fusion_type='softmax'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.num_branches = len(grid_sizes)
        self.fusion_type = fusion_type

        kan_kwargs = dict(
            spline_order=3, scale_noise=0.1, scale_base=1.0,
            scale_spline=1.0, base_activation=Swish,
            grid_eps=0.02, grid_range=[-1, 1],
        )

        self.branches = nn.ModuleList()
        for g in grid_sizes:
            self.branches.append(
                KANLinear(in_features, hidden_features, grid_size=g, **kan_kwargs)
            )

        if self.num_branches > 1:
            if fusion_type == 'softmax':
                self.fusion_proj = nn.Linear(hidden_features * self.num_branches,
                                             self.num_branches)
            elif fusion_type == 'concat':
                self.fusion_conv = nn.Linear(hidden_features * self.num_branches,
                                             hidden_features)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        if self.num_branches == 1:
            out = self.branches[0](x.reshape(B * N, C))
            return out.reshape(B, N, C).contiguous()

        branch_outputs = []
        for fc in self.branches:
            out = fc(x.reshape(B * N, C)).reshape(B, N, -1).contiguous()
            branch_outputs.append(out)

        if self.fusion_type == 'softmax':
            concat = torch.cat(branch_outputs, dim=-1)
            alpha = F.softmax(self.fusion_proj(concat), dim=-1)
            fused = sum(alpha[..., i:i+1] * branch_outputs[i]
                        for i in range(self.num_branches))
        elif self.fusion_type == 'concat':
            fused = self.fusion_conv(torch.cat(branch_outputs, dim=-1))
        elif self.fusion_type == 'sum':
            fused = sum(branch_outputs) / self.num_branches
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")
        return fused


class msShiftedBlock(nn.Module):
    """shiftedBlock with Multi-scale KAN."""
    def __init__(self, dim, mlp_ratio=4., drop_path=0., norm_layer=nn.LayerNorm,
                 grid_sizes=[3, 5, 10], fusion_type='softmax'):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(256, dim),
        )

        self.kan = ms_kan(in_features=dim, hidden_features=mlp_hidden_dim,
                          grid_sizes=grid_sizes, fusion_type=fusion_type)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, temb):
        temb = self.temb_proj(temb)
        x = self.drop_path(self.kan(self.norm2(x), H, W))
        x = x + temb.unsqueeze(1)
        return x


class UKan_PP_Hybrid(nn.Module):
    """U-KAN++ Hybrid Diffusion Model with MS-KAN blocks."""
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout,
                 grid_sizes=[3, 5, 10], fusion_type='softmax'):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        attn = []
        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            h_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, h_ch=h_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = h_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            h_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, h_ch=h_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = h_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )

        # MS-KAN middle blocks
        embed_dims = [256, 320, 512]
        norm_layer = nn.LayerNorm
        dpr = [0.0, 0.0, 0.0]
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=64 // 4, patch_size=3, stride=2,
            in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=64 // 8, patch_size=3, stride=2,
            in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])
        self.dnorm3 = norm_layer(embed_dims[1])

        ms_kwargs = dict(grid_sizes=grid_sizes, fusion_type=fusion_type)

        self.kan_block1 = nn.ModuleList([msShiftedBlock(
            dim=embed_dims[1], mlp_ratio=1, drop_path=dpr[0],
            norm_layer=norm_layer, **ms_kwargs)])
        self.kan_block2 = nn.ModuleList([msShiftedBlock(
            dim=embed_dims[2], mlp_ratio=1, drop_path=dpr[1],
            norm_layer=norm_layer, **ms_kwargs)])
        self.kan_dblock1 = nn.ModuleList([msShiftedBlock(
            dim=embed_dims[1], mlp_ratio=1, drop_path=dpr[0],
            norm_layer=norm_layer, **ms_kwargs)])

        self.decoder1 = D_SingleConv(embed_dims[2], embed_dims[1])
        self.decoder2 = D_SingleConv(embed_dims[1], embed_dims[0])

        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        temb = self.time_embedding(t)
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)

        t3 = h
        B = x.shape[0]

        h, H, W = self.patch_embed3(h)
        for blk in self.kan_block1:
            h = blk(h, H, W, temb)
        h = self.norm3(h)
        h = h.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = h

        h, H, W = self.patch_embed4(h)
        for blk in self.kan_block2:
            h = blk(h, H, W, temb)
        h = self.norm4(h)
        h = h.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        h = swish(F.interpolate(self.decoder1(h, temb), scale_factor=(2, 2), mode='bilinear'))
        h = torch.add(h, t4)

        _, _, H, W = h.shape
        h = h.flatten(2).transpose(1, 2)
        for blk in self.kan_dblock1:
            h = blk(h, H, W, temb)

        h = self.dnorm3(h)
        h = h.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        h = swish(F.interpolate(self.decoder2(h, temb), scale_factor=(2, 2), mode='bilinear'))

        h = torch.add(h, t3)

        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h
