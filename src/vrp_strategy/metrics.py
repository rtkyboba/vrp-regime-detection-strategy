"""Performance-metric utilities, matching notebooks/05_backtest.ipynb."""
import numpy as np


def performance_metrics(returns, periods_per_year=252):
    """Compute standard risk/return metrics for a daily return series.

    Returns a dict (rather than only printing) so results from several
    strategy variants can be stacked into a comparison DataFrame.
    """
    returns = returns.dropna()
    ann_return = returns.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    cum = (1 + returns).cumprod()
    drawdown = cum / cum.cummax() - 1
    max_dd = drawdown.min()

    downside = returns[returns < 0].std() * np.sqrt(periods_per_year)
    sortino = ann_return / downside if downside > 0 else np.nan
    calmar = ann_return / abs(max_dd) if max_dd != 0 else np.nan
    win_rate = (returns > 0).mean()

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "n_obs": len(returns),
    }


def safe_corr(a, b):
    """Pearson correlation without np.corrcoef / Series.corr().

    This local environment's numpy/BLAS build segfaults on np.corrcoef,
    np.cov and Series.corr() (anything that hits a BLAS dot-product
    reduction) while plain elementwise ops (sum, mean, rolling std) are
    fine. This computes the same statistic by hand to sidestep it.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    am = a - a.mean()
    bm = b - b.mean()
    denom = np.sqrt((am**2).sum()) * np.sqrt((bm**2).sum())
    return float((am * bm).sum() / denom) if denom > 0 else np.nan


def print_metrics(metrics, label):
    print(f"\n{label}")
    print(f"  Annualised return:  {metrics['ann_return']*100:.2f}%")
    print(f"  Annualised vol:     {metrics['ann_vol']*100:.2f}%")
    print(f"  Sharpe ratio:       {metrics['sharpe']:.3f}")
    print(f"  Sortino ratio:      {metrics['sortino']:.3f}")
    print(f"  Calmar ratio:       {metrics['calmar']:.3f}")
    print(f"  Max drawdown:       {metrics['max_drawdown']*100:.2f}%")
    print(f"  Win rate:           {metrics['win_rate']*100:.1f}%")
