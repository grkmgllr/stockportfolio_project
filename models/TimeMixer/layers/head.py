import torch
import torch.nn as nn

class ForecastHead(nn.Module):
    """
    Converts multi-scale encoder outputs -> pred_len outputs and aggregates.
    """
    def __init__(self, seq_len: int, pred_len: int, d_model: int, c_out: int, down_window: int, down_layers: int):
        super().__init__()
        self.pred_len = pred_len
        self.c_out = c_out

        self.predict_layers = nn.ModuleList([
            nn.Linear(seq_len // (down_window ** i), pred_len)
            for i in range(down_layers + 1)
        ])

        self.proj = nn.Linear(d_model, c_out)

    def forward(self, enc_out_list):
        # enc_out_list[i]: [B,T_i,d_model]
        preds = []
        for i, enc_out in enumerate(enc_out_list):
            # make time dimension last for Linear(T_i -> pred_len)
            x = enc_out.permute(0, 2, 1)          # [B,d_model,T_i]
            x = self.predict_layers[i](x)         # [B,d_model,pred_len]
            x = x.permute(0, 2, 1)                # [B,pred_len,d_model]
            x = self.proj(x)                      # [B,pred_len,c_out]
            preds.append(x)

        # aggregate multi-scale predictions
        out = torch.stack(preds, dim=-1).mean(-1)  # mean over scales
        return out