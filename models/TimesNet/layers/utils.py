import torch


def FFT_for_Period(x, k=2):
    """
    Uses FFT to find dominant periods in a time series.

    x: [Batch, Time, Channels]
    returns:
        periods: list of top-k periods
        amplitudes: FFT magnitude used as aggregation weights
    """

    # FFT along time dimension
    fft_result = torch.fft.rfft(x, dim=1)

    # Average over batch and channels to get global frequency importance
    amplitude = fft_result.abs().mean(dim=0).mean(dim=-1)
    amplitude[0] = 0  # ignore zero frequency

    _, top_freq = torch.topk(amplitude, k)
    periods = x.shape[1] // top_freq.cpu().numpy()

    return periods, fft_result.abs().mean(dim=-1)[:, top_freq]