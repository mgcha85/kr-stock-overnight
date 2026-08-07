# Full Rolling Walk-Forward Strategy Backtest Report

### Backtest Architecture & Dataset Scope
- **Scope**: All historical MarketMosaic datasets (`2026-04-01 ~ 2026-08-03`).
- **Causal Controls**: Look-Ahead Bias 0%, News Cutoff `15:30:00`, Upper Limit (+29.0%) Lock Filter Enabled.
- **Model**: LightGBM + PyTorch Ensemble + Judal Theme Driver + Auxiliary DART/News Bonus.

## 1. 30-Day (Monthly) Rolling Performance

```text
                        label                  period  trading_days  trades  return_pct  bh_return_pct  alpha_pct  win_rate    pf   mdd
April 2026 (In-Sample Warmup) 2026-04-01 ~ 2026-04-30          11.0      33        8.65          24.33     -15.68     63.64  3.29 -0.92
      May 2026 (Train Period) 2026-05-01 ~ 2026-05-31           6.0      18       50.77          -2.38      53.15     83.33  6.36  0.00
 June 2026 (Out-of-Sample M1) 2026-06-01 ~ 2026-06-30          16.0      48      223.47         -10.06     233.53     81.25 10.61 -4.19
 July 2026 (Out-of-Sample M2) 2026-07-01 ~ 2026-07-31          16.0      45       52.33         -27.01      79.33     64.44  2.39 -8.64
      August 2026 (Recent M3) 2026-08-01 ~ 2026-08-03           NaN       0        0.00           0.00       0.00      0.00  0.00  0.00
```

## 2. 7-Day (Weekly) Rolling Performance

```text
 week_idx                  period  trades  return_pct  bh_return_pct  alpha_pct  win_rate    pf   mdd
        1 2026-05-12 ~ 2026-05-28      15       37.01          -3.08      40.09     80.00  5.10  0.00
        2 2026-05-29 ~ 2026-06-05      15       53.65          -4.21      57.86     86.67 11.14 -4.19
        3 2026-06-08 ~ 2026-06-12      15       71.62          11.22      60.40     86.67 45.95  0.00
        4 2026-06-16 ~ 2026-06-26      15       14.43         -10.04      24.47     66.67  2.96 -3.34
        5 2026-06-29 ~ 2026-07-03      15       33.00           1.23      31.77     86.67 52.92  0.00
        6 2026-07-10 ~ 2026-07-20      15       16.99          -6.21      23.20     66.67  3.22  0.00
        7 2026-07-21 ~ 2026-07-27      15       26.42           5.44      20.98     60.00  3.72 -2.07
        8 2026-07-28 ~ 2026-07-30       6       -8.64          -9.12       0.48     50.00  0.41 -6.53
```

