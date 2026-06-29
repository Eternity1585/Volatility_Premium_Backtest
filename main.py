from pathlib import Path

from src.backtest import optimize_threshold, run_short_straddle_backtest
from src.volatility import add_realized_vol, load_market_data, make_synthetic_data


METRIC_COLS = ["total_return", "total_pnl", "max_drawdown", "sharpe", "avg_abs_delta", "trades"]
SIGNAL_COLS = ["spy", "rv", "vix_proxy", "vrp", "signal", "net_delta", "daily_return", "cum_pnl"]

# Edit values for quick experiments.
USE_SYNTHETIC_DATA = False
YFINANCE_PERIOD = "5y"
RV_WINDOW = 21
TRANSACTION_COST = 1.0
INITIAL_CAPITAL = 10000
CONTRACTS = 1
TRAIN_FRAC = 0.7

OPTIMIZE_R = True
FIXED_R = 0.06
START_R = 0.00
STOP_R = 0.15
STEP_R = 0.01

SAVE_RESULTS = False


def main():
    raw = make_synthetic_data() if USE_SYNTHETIC_DATA else load_market_data(period=YFINANCE_PERIOD)
    data = add_realized_vol(raw, window=RV_WINDOW)

    if OPTIMIZE_R:
        best_R, train_summary, result, test_stats = optimize_threshold(
            data,
            start=START_R,
            stop=STOP_R,
            step=STEP_R,
            transaction_cost=TRANSACTION_COST,
            initial_capital=INITIAL_CAPITAL,
            contracts=CONTRACTS,
            train_frac=TRAIN_FRAC,
        )
        train_stats = train_summary.iloc[0][METRIC_COLS]
    else:
        train_summary = None
        train_stats = None
        best_R = FIXED_R
        result, test_stats = run_short_straddle_backtest(
            data,
            threshold=best_R,
            transaction_cost=TRANSACTION_COST,
            initial_capital=INITIAL_CAPITAL,
            contracts=CONTRACTS,
        )

    test_stats = {k: test_stats[k] for k in METRIC_COLS}
    metric_table = [("Test", test_stats)]
    if train_stats is not None:
        metric_table.insert(0, ("Train", train_stats))

    print("Volatility Risk Premium Backtest")
    print("--------------------------------")
    print(f"Best R threshold:  {best_R:.2%}")
    for label, stats in metric_table:
        print(
            f"{label:>5}: return={stats['total_return']:.2%}, "
            f"pnl=${stats['total_pnl']:.2f}, "
            f"dd=${stats['max_drawdown']:.2f}, "
            f"sharpe={stats['sharpe']:.2f}, "
            f"avg_delta={stats['avg_abs_delta']:.3f}, "
            f"trades={int(stats['trades'])}"
        )
    print()

    if train_summary is not None:
        print("Top train thresholds by total PnL:")
        print(train_summary[["R"] + METRIC_COLS].head(8).to_string(index=False))
        print()

    print("Recent test signals:")
    print(result[SIGNAL_COLS].tail(8))

    if SAVE_RESULTS:
        out = Path("results")
        out.mkdir(exist_ok=True)
        result.to_csv(out / "best_threshold_backtest.csv")
        if train_summary is not None:
            train_summary.to_csv(out / "threshold_search.csv", index=False)
        print("\nSaved results to results/")


if __name__ == "__main__":
    main()
