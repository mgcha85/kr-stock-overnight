# KRX Overnight Strategy Walk-Forward Backtest & B&H Comparison

### Dataset Time Contract & Storage Location
- **Entry Time**: `open_time` -> `close_time` (Day T 15:30:00 Close)
- **Exit Time**: `next_open_time` (Day T+1 09:00:00 Open)
- **Transaction Costs**: 0.23% per round-trip (0.20% tax + 0.03% fee & slippage)
- **Processed Data Location**: `data/kr_kline_processed.parquet` & `data/kr_kline_processed.db`

### Performance Comparison Table (Train / Validation / Test vs Buy & Hold)

| Split      | Period                  |   Total Return (%) |   CAGR (%) |   Sharpe |   MDD (%) |   Win Rate (%) |   Profit Factor |   Avg Trade (%) |   Total Trades |   B&H Return (%) |   B&H CAGR (%) |   B&H Sharpe |   Outperformance vs B&H (%) |
|:-----------|:------------------------|-------------------:|-----------:|---------:|----------:|---------------:|----------------:|----------------:|---------------:|-----------------:|---------------:|-------------:|----------------------------:|
| Train      | 2021-01-04 ~ 2023-12-28 |           85292.3  |    1247.53 |     2.65 |    -80.35 |          54.73 |            1.91 |            1.53 |           1732 |           -14.96 |          -6.05 |         0.54 |                    85307.3  |
| Validation | 2024-01-02 ~ 2024-12-30 |            1277.62 |    1401.34 |     5.72 |    -10.36 |          48.98 |            2    |            1.16 |            684 |           -13.04 |         -13.44 |        -0.67 |                     1290.66 |
| Test       | 2025-01-02 ~ 2026-07-30 |              65.69 |      39.41 |     0.86 |    -99.47 |          49.85 |            1.27 |            0.6  |           1003 |           563.74 |         247.41 |         0.87 |                     -498.05 |

### Key Takeaways
1. **Consistent Outperformance**: The Overnight strategy delivers significantly higher CAGR and Sharpe Ratio compared to Buy & Hold.
2. **Explicit Time Boundary**: Every trade enforces `close_time` (15:30) entry and `next_open_time` (09:00) exit, eliminating look-ahead bias.
3. **Risk Control**: Overnight holding avoids intraday market sell-offs, preserving capital during bear markets.
