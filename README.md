# Volatility Risk Premium Backtest

A small Python research project for testing a simple volatility risk premium
idea on S&P 500 data.

The question is:

```text
If VIX proxy > realized volatility + R, what R threshold works best?
```

The project uses real Yahoo Finance data: `SPY` as the S&P 500 tradable proxy
and `^VIX / 100` as a rough 30-day implied volatility proxy. This is not a
perfect option-chain backtest, but it is a practical way to study the IV/RV
relationship without paid data.

## Method

1. Load SPY and VIX data from `yfinance`.
2. Compute rolling realized volatility from SPY log returns.
3. Compute `VRP = vix_proxy - RV`.
4. Search a grid of thresholds `R`.
5. On the train set, choose the `R` that gives the best total PnL.
6. Evaluate that chosen `R` on the test set.

When the signal triggers, the strategy sells a simplified 30-day ATM straddle
and marks it one day later.

## Run

```bash
pip install -r requirements.txt
python main.py
```

To change the experiment, edit the settings at the top of `main.py`:

```python
YFINANCE_PERIOD = "5y"
RV_WINDOW = 21
TRANSACTION_COST = 1.0
TRAIN_FRAC = 0.7
START_R = 0.00
STOP_R = 0.15
STEP_R = 0.01
```

The normal project run uses real SPY and VIX data. There is still a
`USE_SYNTHETIC_DATA` setting in `main.py` for offline debugging, but it is off by
default. If Yahoo Finance data cannot be downloaded, the program raises an error
instead of silently switching to fake data.

## Notes

- Sharpe is calculated from daily portfolio returns, not raw PnL.
- Position sizing is explicit through `contracts`.
- Threshold selection is in-sample on the train set; reported test metrics use
  the selected `R` on unseen data.
- `vix_proxy` is not the true BSM implied volatility of a specific SPY option.
- The backtest reports entry net delta, but it does not run a full delta hedge.

## Limitations

- VIX is a variance-swap-style index and includes skew and variance risk premium.
- The straddle price is simplified with Black-Scholes.
- No margin model, bid-ask spread, assignment, or intraday hedging is included.
- Synthetic data is only available through the explicit `--synthetic` flag.
- The result is a research exercise, not a trading recommendation.
