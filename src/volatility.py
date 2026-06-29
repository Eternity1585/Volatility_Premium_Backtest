import numpy as np
import pandas as pd


TRADING_DAYS = 252


def make_synthetic_data(days=900, seed=7):
    """
    Synthetic SPY/VIX-like data for explicit offline debugging only.

    IV is usually above RV, but not always. That difference is the volatility
    risk premium this project is studying. The column is called vix_proxy so it
    matches the live-data path.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-02", periods=days)

    regime = np.where(np.arange(days) < days * 0.6, 0.16, 0.24)
    shocks = rng.normal(0, regime / np.sqrt(TRADING_DAYS))
    jumps = rng.choice([0, -0.025, 0.025], size=days, p=[0.96, 0.02, 0.02])
    returns = 0.00025 + shocks + jumps

    close = 320 * np.exp(np.cumsum(returns))
    vix_noise = rng.normal(0, 0.035, days)
    vix_proxy = np.clip(regime + 0.035 + vix_noise, 0.08, 0.65)

    return pd.DataFrame({"spy": close, "vix_proxy": vix_proxy}, index=dates)


def load_market_data(period="5y"):
    """
    Load SPY and VIX from yfinance.

    VIX is used as a rough S&P 500 30-day implied volatility proxy. It is not
    the same as a full SPY option surface, but it is good enough for a small
    research project without paid option data.

    VIX is a fair variance swap rate, not a BSM implied vol. Using it as sigma
    in straddle_price() introduces a small systematic bias: VIX incorporates the
    variance risk premium and OTM skew that a single ATM option vol does not.
    Results should be interpreted as approximate.
    """
    import yfinance as yf

    spy = yf.download("SPY", period=period, auto_adjust=True, progress=False)
    vix = yf.download("^VIX", period=period, auto_adjust=True, progress=False)

    if spy.empty or vix.empty:
        raise ValueError("yfinance returned empty SPY or VIX data")

    spy_close = _close_series(spy, "SPY")
    vix_close = _close_series(vix, "^VIX")

    data = pd.DataFrame(
        {
            "spy": spy_close,
            # ^VIX / 100 used as SPY 30d ATM IV proxy; not identical to
            # BSM-implied vol of a specific strike.
            "vix_proxy": vix_close / 100,
        }
    )
    return data.dropna()


def _close_series(data, ticker):
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        if ticker in close.columns:
            close = close[ticker]
        else:
            close = close.iloc[:, 0]
    return close


def add_realized_vol(data, window=21):
    out = data.copy()
    out["return"] = np.log(out["spy"] / out["spy"].shift(1))
    out["rv"] = out["return"].rolling(window).std() * np.sqrt(TRADING_DAYS)
    out["vrp"] = out["vix_proxy"] - out["rv"]
    return out.dropna()


def max_drawdown(series):
    running_high = series.cummax()
    return float((series - running_high).min())


def sharpe_ratio(daily_returns):
    if daily_returns.std() == 0:
        return 0.0
    return float(daily_returns.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS))
