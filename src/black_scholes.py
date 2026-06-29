import math


def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def d1(S, K, T, r, sigma):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("S, K, T, and sigma must be positive")
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def option_price(S, K, T, r, sigma, option_type="call"):
    """Simple Black-Scholes price for a European call or put."""
    x1 = d1(S, K, T, r, sigma)
    x2 = d2(S, K, T, r, sigma)

    if option_type == "call":
        return S * normal_cdf(x1) - K * math.exp(-r * T) * normal_cdf(x2)
    if option_type == "put":
        return K * math.exp(-r * T) * normal_cdf(-x2) - S * normal_cdf(-x1)

    raise ValueError("option_type must be 'call' or 'put'")


def straddle_price(S, K, T, r, sigma):
    """ATM straddle = call + put. Useful for simple volatility exposure."""
    call = option_price(S, K, T, r, sigma, "call")
    put = option_price(S, K, T, r, sigma, "put")
    return call + put
