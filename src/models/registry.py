"""Model registry — single place to look up model classes and their configs.

Adding a new forecaster: extend the two dispatch functions here and it
becomes visible to both training and evaluation without touching them.
"""

from models import (
    TimeMixer, TimeMixerConfig,
    TimesNetModel, TimesNetConfig,
    StockMixer, StockMixerConfig,
)


def get_model_config(model_name: str, seq_len: int, pred_len: int,
                     enc_in: int = 5, c_out: int = 2,
                     denorm_indices: tuple | None = None,
                     return_targets: bool = False,
                     num_stocks: int | None = None,
                     market_dim: int = 2):
    """
    Get model config for stock price prediction.

    Input: OHLCV or OHLCV+Vwap+Transactions (enc_in features)
    Output: High, Close + optional MA targets (c_out features)

    denorm_indices maps each output channel to the input channel whose
    per-sample mean/std should be used for NS-Norm / RevIN denormalization.
    When return_targets=True, denormalization is skipped (model outputs returns).

    num_stocks is only used by cross-stock models (StockMixer) and must
    equal the number of tickers packed into each sample by CrossStockDataset.
    """
    if model_name == "TimesNet":
        return TimesNetConfig(
            task_name="long_term_forecast",
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            c_out=c_out,
            d_model=32,
            d_ff=64,
            e_layers=2,
            top_k=3,
            num_kernels=6,
            embed="fixed",
            freq="d",
            dropout=0.1,
            num_class=c_out,
            denorm_indices=denorm_indices,
            return_targets=return_targets,
        )
    if model_name == "TimeMixer":
        # downsampling layers halve seq_len; ensure smallest scale > kernel
        n_layers = 1 if seq_len % 4 != 0 else 2
        min_scale = seq_len // (2 ** n_layers)
        ma_kernel = min(25, min_scale - 2)
        if ma_kernel % 2 == 0:
            ma_kernel -= 1
        ma_kernel = max(3, ma_kernel)
        return TimeMixerConfig(
            historical_lookback_length=seq_len,
            forecast_horizon_length=pred_len,
            number_of_input_features=enc_in,
            number_of_output_features=c_out,
            model_embedding_dimension=64,
            feedforward_hidden_dimension=128,
            number_of_pdm_blocks=2,
            dropout_probability=0.1,
            downsampling_window_size=2,
            number_of_downsampling_layers=n_layers,
            moving_average_kernel_size=ma_kernel,
            denorm_indices=denorm_indices,
            return_targets=return_targets,
        )
    if model_name == "StockMixer":
        if num_stocks is None:
            raise ValueError(
                "StockMixer requires num_stocks (pass the CrossStockDataset's "
                "num_stocks). It is a cross-stock model — see CrossStockDataset."
            )
        if not return_targets:
            raise ValueError(
                "StockMixer is a return-basis model; use return_targets=True."
            )
        return StockMixerConfig(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            c_out=c_out,
            num_stocks=num_stocks,
            market_dim=market_dim,
            denorm_indices=denorm_indices,
            return_targets=return_targets,
        )
    raise ValueError(f"Unknown model: {model_name}")


def get_model(model_name: str, config):
    """Instantiate a model from its config."""
    if model_name == "TimesNet":
        return TimesNetModel(config)
    if model_name == "TimeMixer":
        return TimeMixer(config)
    if model_name == "StockMixer":
        return StockMixer(config)
    raise ValueError(f"Unknown model: {model_name}")
