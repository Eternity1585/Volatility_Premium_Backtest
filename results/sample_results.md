# Sample Results

Example command:

```bash
python main.py --synthetic --save
```

Example output from the synthetic dataset:

```text
Best R threshold:  6.00%
Total return:      2.31%
Total PnL:         $231.28
Trades:            223
```

The best threshold in this run is `R = 6%`, meaning the strategy only sells the
30-day ATM straddle when:

```text
IV > RV + 0.06
```

This result should not be read as proof that 6% is the correct threshold in real
markets. It only shows how the research process works: test a grid of risk
premium thresholds, compare PnL and drawdown, then ask whether the result is
stable enough to investigate further.

The saved files are:

- `threshold_search.csv`: performance for each tested `R`
- `best_threshold_backtest.csv`: daily PnL path for the best `R`
