import pandas as pd
import numpy as np

from src.black_scholes import d1, normal_cdf, straddle_price
from src.volatility import max_drawdown, sharpe_ratio


def add_signal(data, threshold):
    out = data.copy()
    out["signal"] = np.where(out["vix_proxy"] > out["rv"] + threshold, "SELL", "HOLD")
    return out


def threshold_grid(start=0.0, stop=0.15, step=0.01):
    values = np.arange(start, stop + step / 2, step)
    return [round(float(x), 4) for x in values]


def run_short_straddle_backtest(
    data,
    threshold,
    r=0.04,
    days_to_expiry=30,
    transaction_cost=1.0,
    initial_capital=10000,
    contracts=1,
):
    """
    One-day rolling short straddle backtest.

    If IV > RV + R, sell a 30-day ATM straddle and mark it one day later. This
    is a simplified way to study volatility risk premium. Real option selling
    needs margin, bid-ask modeling, assignment handling, and usually delta
    hedging to reduce stock direction risk.

    contracts scales the single-straddle PnL. At contracts=1, total_return
    understates true capital utilisation because no margin or premium-as-capital
    model is applied.
    """
    data = add_signal(data, threshold)
    rows = []
    cum_pnl = 0.0
    T0 = days_to_expiry / 252
    T1 = max((days_to_expiry - 1) / 252, 1 / 252)

    for i in range(len(data) - 1):
        today = data.iloc[i]
        tomorrow = data.iloc[i + 1]
        date = data.index[i]

        pnl = 0.0
        entry = 0.0
        exit_price = 0.0
        net_delta = 0.0
        signal = today["signal"]

        if signal == "SELL":
            K = round(today["spy"] / 5) * 5
            sigma = today["vix_proxy"]
            entry = straddle_price(today["spy"], K, T0, r, sigma)
            exit_price = straddle_price(tomorrow["spy"], K, T1, r, tomorrow["vix_proxy"])
            pnl = (entry - exit_price - transaction_cost) * contracts

            call_delta = normal_cdf(d1(today["spy"], K, T0, r, sigma))
            put_delta = call_delta - 1
            net_delta = -call_delta - put_delta

        cum_pnl += pnl
        rows.append(
            {
                "date": date,
                "spy": today["spy"],
                "rv": today["rv"],
                "vix_proxy": today["vix_proxy"],
                "vrp": today["vrp"],
                "signal": signal,
                "entry_price": entry,
                "exit_price": exit_price,
                "net_delta": net_delta,
                "daily_pnl": pnl,
                "cum_pnl": cum_pnl,
            }
        )

    result = pd.DataFrame(rows).set_index("date")
    return result, summarize(result, initial_capital)


def summarize(result, initial_capital):
    trades = int((result["signal"] == "SELL").sum()) if not result.empty else 0
    total_pnl = float(result["cum_pnl"].iloc[-1]) if not result.empty else 0.0

    if not result.empty:
        result["daily_return"] = result["daily_pnl"] / initial_capital
        sell_delta = result.loc[result["signal"] == "SELL", "net_delta"].abs()
        avg_abs_delta = float(sell_delta.mean()) if not sell_delta.empty else 0.0
    else:
        avg_abs_delta = 0.0

    return {
        "total_pnl": total_pnl,
        "total_return": (initial_capital + total_pnl) / initial_capital - 1,
        "max_drawdown": max_drawdown(result["cum_pnl"]) if not result.empty else 0.0,
        "sharpe": sharpe_ratio(result["daily_return"]) if not result.empty else 0.0,
        "trades": trades,
        "avg_abs_delta": avg_abs_delta,
    }


def optimize_threshold(
    data,
    start=0.0,
    stop=0.15,
    step=0.01,
    transaction_cost=1.0,
    initial_capital=10000,
    contracts=1,
    train_frac=0.7,
):
    cutoff = data.index[int(len(data) * train_frac)]
    train_data = data[data.index < cutoff]
    test_data = data[data.index >= cutoff]
    rows = []

    for R in threshold_grid(start, stop, step):
        _, stats = run_short_straddle_backtest(
            train_data,
            threshold=R,
            transaction_cost=transaction_cost,
            initial_capital=initial_capital,
            contracts=contracts,
        )
        stats["R"] = R
        rows.append(stats)

    train_summary = pd.DataFrame(rows).sort_values("total_pnl", ascending=False)
    best_R = float(train_summary.iloc[0]["R"])
    test_result, test_stats = run_short_straddle_backtest(
        test_data,
        threshold=best_R,
        transaction_cost=transaction_cost,
        initial_capital=initial_capital,
        contracts=contracts,
    )
    return best_R, train_summary, test_result, test_stats
