"""VRP strategy: short-volatility research with causal regime models and honest P&L.

Module map
----------
Current
    data           loaders, the train/test split, tickers (raw files come from scripts/fetch_data.py)
    signals        VRP, slope and realised-vol table built from raw SPX/VIX/VIX3M
    regimes        causal HMM forward filter (``forward_filter``, ``filtered_proba``)
    regime_models  34-method causal regime comparison behind one fit/filter interface
    pnl            honest P&L models (held-to-maturity swap, MTM, SVXY) and the generic backtester
    rnn_signal     purged walk-forward GRU/LSTM and linear/tree baselines
    stats_tests    PSR, DSR, Lo adjustment, stationary bootstrap, PBO, reality check, timing nulls
    alt_strategies vol-managed SPY, term-structure switching, tail hedges (traded ETFs)
    combiner       blend regime and forecast signals; evaluated against same-size constant exposure
Legacy (original build, notebooks 05-09; kept to reproduce the 3.08 headline)
    backtest       proxy P&L and trailing-vol targeting
    metrics        original metric set
"""
