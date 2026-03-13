"""
Spline Activation Maps (SAM) for U-KAN / U-KAN++.

Computes interpretability heatmaps by extracting gradients from KANLinear
spline activations. Supports multi-layer aggregation and comparison with
GradCAM and random baselines.
"""
import torch
import torch.nn.functional as F
import numpy as np
from kan import KANLinear


def compute_sam(model, input_tensor, target_class=0, layer_agg='mean'):
    """
    Compute Spline Activation Map.
    Eq. 5: SAM(h,w) = (1/C) * sum_c |d phi_c(x) / dx|

    Args:
        model: trained U-KAN/U-KAN++ model
        input_tensor: (1, C, H, W) input image tensor
        target_class: class index for gradient computation
        layer_agg: 'mean' or 'max' aggregation across layers

    Returns:
        sam_map: (H, W) numpy array, normalized to [0, 1]
    """
    model.eval()
    H_in, W_in = input_tensor.shape[2], input_tensor.shape[3]
    input_tensor = input_tensor.clone().detach().requires_grad_(True)

    layer_grads = []

    def make_hook(layer_idx):
        def hook_fn(module, inp, out):
            x = inp[0].detach().requires_grad_(True)
            bases = module.b_splines(x)
            spline_out = F.linear(
                bases.view(x.size(0), -1),
                module.scaled_spline_weight.view(module.out_features, -1),
            )
            grad = torch.autograd.grad(
                spline_out.sum(), x, create_graph=False
            )[0]
            grad_mag = grad.abs().mean(dim=-1)
            layer_grads.append(grad_mag.detach())
        return hook_fn

    hooks = []
    idx = 0
    for name, module in model.named_modules():
        if isinstance(module, KANLinear):
            hooks.append(module.register_forward_hook(make_hook(idx)))
            idx += 1

    with torch.enable_grad():
        model(input_tensor)

    for h in hooks:
        h.remove()

    if not layer_grads:
        return np.zeros((H_in, W_in))

    sam_maps = []
    for grad_mag in layer_grads:
        N = grad_mag.shape[0]
        h = w = int(np.sqrt(N))
        if h * w != N:
            continue
        spatial = grad_mag.view(1, 1, h, w)
        upsampled = F.interpolate(spatial, size=(H_in, W_in),
                                  mode='bilinear', align_corners=False)
        sam_maps.append(upsampled.squeeze().cpu().numpy())

    if not sam_maps:
        return np.zeros((H_in, W_in))

    sam_maps = np.stack(sam_maps, axis=0)
    result = sam_maps.mean(axis=0) if layer_agg == 'mean' else sam_maps.max(axis=0)

    if result.max() > result.min():
        result = (result - result.min()) / (result.max() - result.min())
    return result


def compute_gradcam(model, input_tensor, target_class=0):
    """
    Compute GradCAM from the last Conv2d layer (baseline comparison).
    """
    model.eval()
    H_in, W_in = input_tensor.shape[2], input_tensor.shape[3]

    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    if target_layer is None:
        return np.zeros((H_in, W_in))

    activations, gradients = [], []

    def fwd_hook(module, inp, out):
        activations.append(out)

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    output = model(input_tensor)

    target = output[:, target_class].sum()
    model.zero_grad()
    target.backward()

    fh.remove()
    bh.remove()

    act = activations[0].detach()
    grad = gradients[0].detach()
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(H_in, W_in), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()

    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam


def compute_random_baseline(H, W):
    """Random baseline saliency map."""
    return np.random.rand(H, W).astype(np.float32)


# ---- Faithfulness (Table IX) ----

def deletion_curve(model, input_tensor, saliency_map, steps=100):
    """Progressively mask pixels by decreasing saliency. Returns confidence list."""
    model.eval()
    B, C, H, W = input_tensor.shape
    num_pixels = H * W

    with torch.no_grad():
        baseline_score = torch.sigmoid(model(input_tensor)).mean().item()

    order = np.argsort(-saliency_map.flatten())
    scores = [baseline_score]
    masked = input_tensor.clone()
    pps = max(1, num_pixels // steps)

    for step in range(1, steps + 1):
        s, e = (step - 1) * pps, min(step * pps, num_pixels)
        idxs = order[s:e]
        rows, cols = idxs // W, idxs % W
        masked[:, :, rows, cols] = 0.0
        with torch.no_grad():
            scores.append(torch.sigmoid(model(masked)).mean().item())
    return scores


def insertion_curve(model, input_tensor, saliency_map, steps=100):
    """Progressively reveal pixels by decreasing saliency. Returns confidence list."""
    model.eval()
    B, C, H, W = input_tensor.shape
    num_pixels = H * W

    revealed = torch.zeros_like(input_tensor)
    order = np.argsort(-saliency_map.flatten())

    with torch.no_grad():
        scores = [torch.sigmoid(model(revealed)).mean().item()]

    pps = max(1, num_pixels // steps)

    for step in range(1, steps + 1):
        s, e = (step - 1) * pps, min(step * pps, num_pixels)
        idxs = order[s:e]
        rows, cols = idxs // W, idxs % W
        revealed[:, :, rows, cols] = input_tensor[:, :, rows, cols]
        with torch.no_grad():
            scores.append(torch.sigmoid(model(revealed)).mean().item())
    return scores


def compute_auc(scores):
    """Area under the curve (trapezoidal), normalized to [0,1]."""
    return float(np.trapz(scores, dx=1.0 / (len(scores) - 1)))


def compute_aopc(deletion_scores):
    """Average Over Perturbation Curve."""
    s = np.array(deletion_scores)
    return float((s[0] - s[1:]).mean())


# ---- Localization (Table X) ----

def iou_at_threshold(saliency_map, gt_mask, threshold=0.5):
    """IoU between thresholded saliency and GT mask."""
    pred = (saliency_map >= threshold).astype(np.float32)
    gt = (gt_mask > 0).astype(np.float32)
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)


def pointing_game(saliency_map, gt_mask):
    """1 if max saliency point is inside GT mask, else 0."""
    idx = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
    return float(gt_mask[idx[0], idx[1]] > 0)
