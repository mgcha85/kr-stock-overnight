# Full Rolling Walk-Forward Strategy Backtest Report

### Backtest Architecture & Dataset Scope
- **Scope**: All historical MarketMosaic datasets (`2026-04-01 ~ 2026-08-03`).
- **Causal Controls**: Look-Ahead Bias 0%, News Cutoff `15:30:00`, Upper Limit (+29.0%) Lock Filter Enabled.
- **Model**: LightGBM + PyTorch Ensemble + Judal Theme Driver + Auxiliary DART/News Bonus.

## 1. 30-Day (Monthly) Rolling Performance

```text
                        label                  period  trading_days  trades  return_pct  bh_return_pct  alpha_pct  win_rate    pf   mdd
April 2026 (In-Sample Warmup) 2026-04-01 ~ 2026-04-30          11.0      33        8.65          24.33     -15.68     63.64  3.29 -0.92
      May 2026 (Train Period) 2026-05-01 ~ 2026-05-31           6.0      18       77.00          -2.38      79.38     83.33  8.59  0.00
 June 2026 (Out-of-Sample M1) 2026-06-01 ~ 2026-06-30          16.0      48      292.71         -10.06     302.77     81.25 12.34 -4.19
 July 2026 (Out-of-Sample M2) 2026-07-01 ~ 2026-07-31          16.0      45       72.86         -27.01      99.86     64.44  2.86 -8.64
      August 2026 (Recent M3) 2026-08-01 ~ 2026-08-03           NaN       0        0.00           0.00       0.00      0.00  0.00  0.00
```

## 2. 7-Day (Weekly) Rolling Performance

```text
 week_idx                  period  trades  return_pct  bh_return_pct  alpha_pct  win_rate    pf   mdd
        1 2026-05-12 ~ 2026-05-28      15       51.53          -3.08      54.60     80.00  6.48  0.00
        2 2026-05-29 ~ 2026-06-05      15       63.10          -4.21      67.31     86.67 12.63 -4.19
        3 2026-06-08 ~ 2026-06-12      15       93.97          11.22      82.76     86.67 57.11  0.00
        4 2026-06-16 ~ 2026-06-26      15       14.43         -10.04      24.47     66.67  2.96 -3.34
        5 2026-06-29 ~ 2026-07-03      15       42.85           1.23      41.62     86.67 67.24  0.00
        6 2026-07-10 ~ 2026-07-20      15       16.99          -6.21      23.20     66.67  3.22  0.00
        7 2026-07-21 ~ 2026-07-27      15       43.46           5.44      38.02     60.00  5.38 -2.07
        8 2026-07-28 ~ 2026-07-30       6       -8.64          -9.12       0.48     50.00  0.41 -6.53
```

