import numpy as np
import torch


@torch.no_grad()
def freq_mask(x, y, rate=0.1, dim=1):
    x_len = x.shape[dim]
    y_len = y.shape[dim]
    xy = torch.cat([x, y], dim=1)
    xy_f = torch.fft.rfft(xy, dim=dim)
    m = torch.rand_like(xy_f, dtype=xy.dtype) < rate

    freal = xy_f.real.masked_fill(m, 0)
    fimag = xy_f.imag.masked_fill(m, 0)
    xy_f = torch.complex(freal, fimag)
    xy = torch.fft.irfft(xy_f, dim=dim)

    if x_len + y_len != xy.shape[dim]:
        xy = torch.cat([x[:, 0:1, ...], xy], 1)

    return torch.split(xy, [x_len, y_len], dim=dim)


@torch.no_grad()
def freq_mix(x, y, rate=0.1, dim=1):
    x_len = x.shape[dim]
    y_len = y.shape[dim]
    xy = torch.cat([x, y], dim=dim)
    xy_f = torch.fft.rfft(xy, dim=dim)

    m = torch.rand_like(xy_f, dtype=xy.dtype) < rate
    amp = abs(xy_f)
    _, index = amp.sort(dim=dim, descending=True)
    dominant_mask = index > 2
    m = torch.bitwise_and(m, dominant_mask)
    freal = xy_f.real.masked_fill(m, 0)
    fimag = xy_f.imag.masked_fill(m, 0)

    b_idx = np.arange(x.shape[0])
    np.random.shuffle(b_idx)
    x2, y2 = x[b_idx], y[b_idx]
    xy2 = torch.cat([x2, y2], dim=dim)
    xy2_f = torch.fft.rfft(xy2, dim=dim)

    m = torch.bitwise_not(m)
    freal2 = xy2_f.real.masked_fill(m, 0)
    fimag2 = xy2_f.imag.masked_fill(m, 0)

    freal += freal2
    fimag += fimag2

    xy_f = torch.complex(freal, fimag)
    xy = torch.fft.irfft(xy_f, dim=dim)

    if x_len + y_len != xy.shape[dim]:
        xy = torch.cat([x[:, 0:1, ...], xy], 1)

    return torch.split(xy, [x_len, y_len], dim=dim)
