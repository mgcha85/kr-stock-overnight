# May 2026 In-Sample Judal Theme Empirical Analysis Report

### Executive Summary
Analyzed 7,958 stock-theme records for May 2026 to extract quantitative Judal theme rules for June Out-of-Sample testing.

### Pattern 1: Judal Theme Group Momentum (`theme_avg_change`)

```text
   theme_avg_bin  trades  win_rate  avg_return
(-15.251, -4.28]    1595 19.749216   -3.214580
 (-4.28, -1.035]    1595 26.394984   -1.693885
 (-1.035, 1.819]    1587 22.936358   -2.234296
  (1.819, 5.558]    1589 19.005664   -3.020298
 (5.558, 40.298]    1592 16.645729   -4.880074
```

### Pattern 2: Judal Theme Leader Stock (`is_leader`)

```text
 is_leader  trades  win_rate  avg_return
     False    7306 20.202573   -3.108292
      True     652 29.294479   -1.892613
```

### Extracted Quantitative Judal Rule Set
- **Primary Driver (Judal Theme Score)**:
  `judal_score = (theme_avg_change * 3.0) + (stock_change * 2.0) + (high_close_ratio * 30.0)`
- **Auxiliary Driver (DART / News Bonus)**:
  `bonus = (dart_count * 5.0) + (news_count * 3.0)`
