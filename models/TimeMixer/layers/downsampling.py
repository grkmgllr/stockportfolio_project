import torch
import torch.nn as nn


class MultiScaleDownsampler(nn.Module):
    """
    Creates multi-scale versions of x_enc (and x_mark if provided).
    Returns:
      x_list:  [x_T, x_T/w, x_T/w^2, ...] each [B,T_i,C]
      mark_list: same lengths if given
    """
    def __init__(self, enc_in: int, window: int, layers: int, method: str = "avg"):
        super().__init__()
        self.window = window
        self.layers = layers
        self.method = method

        if method == "max":
            self.pool = nn.MaxPool1d(window)
        elif method == "avg":
            self.pool = nn.AvgPool1d(window)
        elif method == "conv":
            self.pool = nn.Conv1d(enc_in, enc_in, kernel_size=3, stride=window, padding=1, bias=False)
        else:
            self.pool = None

    def forward(self, x_enc, x_mark=None):
        if self.pool is None:
            return [x_enc], [x_mark] if x_mark is not None else None

        # B,T,C -> B,C,T for pooling
        x = x_enc.permute(0, 2, 1)
        x_list = [x_enc]
        mark_list = [x_mark] if x_mark is not None else None

        for _ in range(self.layers):
            x = self.pool(x)  # [B,C,T']
            x_list.append(x.permute(0, 2, 1))

        if x_mark is not None:
            cur = x_mark
            for _ in range(self.layers):
                cur = cur[:, :: self.window, :]
                mark_list.append(cur)

        return x_list, mark_list