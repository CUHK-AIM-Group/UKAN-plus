"""
STAG explanation maps, CAM baselines, and interpretability metrics.

- STAG map: the decoder gate G read from the forward pass, upsampled to the
  input size and averaged over the decoder stages that use STAG.
- Baselines: Grad-CAM, Grad-CAM++ and Score-CAM on the output of the last
  decoder block (decoder5). The target score is the sum of foreground logits
  over the pixels predicted as foreground for the unperturbed input. Weight
  formulas follow the pytorch-grad-cam implementation.
- Faithfulness: deletion/insertion curves of Dice (%) with shared all-zero
  and complete-input endpoints.
- Localization: IoU@0.5 and Pointing Game.
"""
import numpy as np
import torch
import torch.nn.functional as F


def _normalize(cam):
    """Min-max normalize a (B, 1, h, w) tensor per sample."""
    flat = cam.flatten(1)
    lo = flat.min(dim=1)[0].view(-1, 1, 1, 1)
    cam = cam - lo
    hi = cam.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1)
    return cam / (hi + 1e-7)


def _to_input_size(cam, H, W):
    """ReLU-ed CAM (1, 1, h, w) -> normalized (H, W) numpy map in [0, 1]."""
    cam = _normalize(cam)
    cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
    return _normalize(cam)[0, 0].cpu().numpy()


@torch.no_grad()
def compute_stag(model, input_tensor):
    """STAG map (H, W) in [0, 1] for a single image (1, C, H, W)."""
    model.eval()
    H, W = input_tensor.shape[2:]
    model(input_tensor)
    gates = [g for g in model.last_stag_maps.values() if g is not None]
    if not gates:
        raise ValueError('the model has no active STAG gates')
    maps = [F.interpolate(g.float(), size=(H, W), mode='bilinear', align_corners=False)
            for g in gates]
    return _normalize(torch.stack(maps).mean(dim=0))[0, 0].cpu().numpy()


def _foreground(model, input_tensor):
    """Pixels predicted as foreground for the unperturbed input (all pixels if none)."""
    with torch.no_grad():
        fg = (torch.sigmoid(model(input_tensor)[:, :1]) > 0.5).float()
    return fg if fg.sum() > 0 else torch.ones_like(fg)


def _target_score(logits, fg):
    return (logits[:, :1] * fg).flatten(1).sum(dim=1)


def _activations(model, layer, input_tensor, fg, need_grad):
    store = {}
    handle = layer.register_forward_hook(lambda m, i, o: store.__setitem__('act', o))
    try:
        with torch.set_grad_enabled(need_grad):
            logits = model(input_tensor)
            act = store['act']
            grads = (torch.autograd.grad(_target_score(logits, fg).sum(), act)[0]
                     if need_grad else None)
    finally:
        handle.remove()
    return act.detach(), grads


def compute_cam(model, input_tensor, method='gradcam', layer=None, batch_size=16):
    """Grad-CAM, Grad-CAM++ or Score-CAM map (H, W) in [0, 1] for a single image."""
    model.eval()
    H, W = input_tensor.shape[2:]
    layer = model.decoder5 if layer is None else layer
    fg = _foreground(model, input_tensor)

    if method == 'gradcam':
        act, grads = _activations(model, layer, input_tensor, fg, need_grad=True)
        weights = grads.mean(dim=(2, 3), keepdim=True)
    elif method == 'gradcampp':
        act, grads = _activations(model, layer, input_tensor, fg, need_grad=True)
        g2 = grads.pow(2)
        g3 = g2 * grads
        sum_act = act.sum(dim=(2, 3), keepdim=True)
        aij = g2 / (2 * g2 + sum_act * g3 + 1e-6)
        aij = torch.where(grads != 0, aij, torch.zeros_like(aij))
        weights = (F.relu(grads) * aij).sum(dim=(2, 3), keepdim=True)
    elif method == 'scorecam':
        act, _ = _activations(model, layer, input_tensor, fg, need_grad=False)
        with torch.no_grad():
            masks = F.interpolate(act, size=(H, W), mode='bilinear', align_corners=True)
            lo = masks.flatten(2).min(dim=2)[0][..., None, None]
            hi = masks.flatten(2).max(dim=2)[0][..., None, None]
            masks = ((masks - lo) / (hi - lo + 1e-8))[0][:, None]  # (K, 1, H, W)
            scores = []
            for k in range(0, masks.shape[0], batch_size):
                batch = input_tensor * masks[k:k + batch_size]
                scores.append(_target_score(model(batch), fg))
            weights = torch.softmax(torch.cat(scores), dim=0).view(1, -1, 1, 1)
    else:
        raise ValueError(f'Unknown CAM method: {method}')

    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    return _to_input_size(cam, H, W)


# ---- Faithfulness ----

@torch.no_grad()
def _dice_percent(model, x, gt):
    """Dice (%) of the thresholded prediction, computed as in metrics.iou_score."""
    pred = torch.sigmoid(model(x)[0, 0]) > 0.5
    inter = (pred & gt).sum().item()
    union = (pred | gt).sum().item()
    iou = (inter + 1e-5) / (union + 1e-5)
    return 100.0 * 2 * iou / (iou + 1)


@torch.no_grad()
def perturbation_curves(model, input_tensor, gt_mask, saliency_map, steps=100):
    """Deletion and insertion curves of Dice (%) for one image.

    Deletion zeroes pixels in decreasing saliency order starting from the
    complete input; insertion restores them in the same order starting from
    the all-zero input. Both curves have steps + 1 points and share the
    complete-input and all-zero endpoints.
    """
    model.eval()
    B, C, H, W = input_tensor.shape
    device = input_tensor.device
    gt = torch.as_tensor(np.asarray(gt_mask) > 0, device=device)
    order = torch.as_tensor(np.argsort(-saliency_map.reshape(-1), kind='stable'), device=device)
    bounds = np.linspace(0, H * W, steps + 1).round().astype(int)

    complete = _dice_percent(model, input_tensor, gt)
    zero = _dice_percent(model, torch.zeros_like(input_tensor), gt)

    source = input_tensor.reshape(B, C, H * W)
    deleted = input_tensor.clone()
    inserted = torch.zeros_like(input_tensor)
    deletion, insertion = [complete], [zero]
    for k in range(1, steps):
        idx = order[bounds[k - 1]:bounds[k]]
        deleted.view(B, C, H * W)[:, :, idx] = 0.0
        inserted.view(B, C, H * W)[:, :, idx] = source[:, :, idx]
        deletion.append(_dice_percent(model, deleted, gt))
        insertion.append(_dice_percent(model, inserted, gt))
    deletion.append(zero)
    insertion.append(complete)
    return deletion, insertion


def compute_auc(scores):
    """Area under a curve sampled uniformly on [0, 1] (trapezoidal), in the units of scores."""
    trapezoid = getattr(np, 'trapezoid', None) or np.trapz
    return float(trapezoid(scores, dx=1.0 / (len(scores) - 1)))


def compute_aopc(deletion_scores):
    """Average Over Perturbation Curve."""
    s = np.array(deletion_scores)
    return float((s[0] - s[1:]).mean())


# ---- Localization ----

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
