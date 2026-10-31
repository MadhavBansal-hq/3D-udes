"""Pure-PyTorch stand-in for the `fused_ssim` CUDA extension (which needs nvcc to build on Windows).
Same signature and maths (11x11 Gaussian window, sigma 1.5); a little slower, identical results."""
import torch
import torch.nn.functional as F

_C1, _C2 = 0.01 ** 2, 0.03 ** 2
_windows = {}


def _window(channels, device, dtype):
    key = (channels, device, dtype)
    if key not in _windows:
        g = torch.exp(-((torch.arange(11, dtype=torch.float32) - 5) ** 2) / (2 * 1.5 ** 2))
        g = (g / g.sum()).to(device=device, dtype=dtype)
        _windows[key] = (g[:, None] * g[None, :]).expand(channels, 1, 11, 11).contiguous()
    return _windows[key]


def fused_ssim(img1, img2, padding="same", train=True):
    c = img1.shape[1]
    w = _window(c, img1.device, img1.dtype)
    pad = 5 if padding == "same" else 0
    conv = lambda x: F.conv2d(x, w, padding=pad, groups=c)
    mu1, mu2 = conv(img1), conv(img2)
    s11 = conv(img1 * img1) - mu1 ** 2
    s22 = conv(img2 * img2) - mu2 ** 2
    s12 = conv(img1 * img2) - mu1 * mu2
    ssim = ((2 * mu1 * mu2 + _C1) * (2 * s12 + _C2)) / ((mu1 ** 2 + mu2 ** 2 + _C1) * (s11 + s22 + _C2))
    return ssim.mean()
