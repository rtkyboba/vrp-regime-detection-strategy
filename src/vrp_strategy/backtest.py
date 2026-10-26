"""LEGACY proxy P&L and vol-targeting, matching notebooks/05_backtest.ipynb exactly.

Kept so the original 3.08 Sharpe can be reproduced (scripts/audit_sharpe.py) and so notebooks
05-09 keep running. The proxy omits mark-to-market and is NOT a tradable P&L; new work should
use ``vrp_strategy.pnl`` (``pnl.backtest``, ``pnl.mtm_returns``, ``pnl.hold_to_maturity``).


We don't have actual options price data, so P&L is proxied as:
    daily_return = position(t-1) x (iv_daily(t-1) - rv_daily(t))
i.e. collect implied variance, pay realised variance. Position is lagged
one day (size yesterday, collect today) to avoid look-ahead bias.
"""
import numpy as np


def raw_strategy_returns(position, iv_daily, rv_daily):
    """Unit-exposure strategy return series from a position signal."""
    return position.shift(1) * (iv_daily.shift(1) - rv_daily)


def vol_target_scale(strat_ret, target_vol=0.10, vol_window=126, scalar_cap=100.0):
    """Scale returns to a target annualised volatility using trailing vol.

    Returns (scaled_returns, scalar). The scalar for day t is built only
    from information available through t-1 (rolling std is itself shifted
    by one day before use), so there is no look-ahead.
    """
    rolling_vol = strat_ret.rolling(vol_window).std().shift(1) * np.sqrt(252)
    scalar = (target_vol / rolling_vol).clip(upper=scalar_cap)
    return strat_ret * scalar, scalar
