"""Sliding-window inference for 3D volumes."""
import itertools

import torch
import torch.nn.functional as F


def _window_starts(size, window, step):
    if size <= window:
        return [0]
    starts = list(range(0, size - window + 1, step))
    if starts[-1] != size - window:
        starts.append(size - window)
    return starts


@torch.no_grad()
def sliding_window_inference(volume, model, window, overlap=0.5, sw_batch_size=1):
    """Average the logits of overlapping windows over a (1, C, D, H, W) volume.

    Volumes smaller than the window are zero-padded at the far end and the
    padding is cropped from the output.
    """
    spatial = tuple(volume.shape[2:])
    pad = [max(w - s, 0) for s, w in zip(spatial, window)]
    if any(pad):
        volume = F.pad(volume, [p for q in reversed(pad) for p in (0, q)])
    size = tuple(volume.shape[2:])
    steps = [max(int(round(w * (1 - overlap))), 1) for w in window]
    origins = list(itertools.product(
        *[_window_starts(s, w, st) for s, w, st in zip(size, window, steps)]))

    out = count = None
    for i in range(0, len(origins), sw_batch_size):
        batch = origins[i:i + sw_batch_size]
        patches = torch.cat([volume[..., d:d + window[0], h:h + window[1], w:w + window[2]]
                             for d, h, w in batch])
        logits = model(patches)
        if out is None:
            out = torch.zeros((1, logits.shape[1]) + size, device=volume.device)
            count = torch.zeros((1, 1) + size, device=volume.device)
        for (d, h, w), lg in zip(batch, logits):
            out[..., d:d + window[0], h:h + window[1], w:w + window[2]] += lg.float()
            count[..., d:d + window[0], h:h + window[1], w:w + window[2]] += 1
    out /= count
    return out[..., :spatial[0], :spatial[1], :spatial[2]]
