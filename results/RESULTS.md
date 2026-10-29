# Results


## Lit fee 1 bps (risk aversion chosen on training days: gamma* = 0.03)

| Strategy | Mean P&L/h | sd | Sharpe/h | P&L net of client spread | Paired vs Nash | mean abs Q_T (BTC) | Lit volume (BTC/h) |
|---|---|---|---|---|---|---|---|
| Nash broker (closed form) | +550 ± 135 | 1616 | +0.340 | -834 | +0 ± 0 | 64.5 | 219 |
| Nash broker, signal off | +560 ± 135 | 1624 | +0.345 | -824 | +9 ± 2 | 64.5 | 215 |
| Nash broker, informed-inventory channel off | -419 ± 94 | 1132 | -0.370 | -1803 | -969 ± 131 | 3.3 | 356 |
| Internalise, liquidate at close | -902 ± 377 | 4522 | -0.200 | -2286 | -1453 ± 337 | 3.3 | 101 |
| Externalise all client flow | -1038 ± 158 | 1897 | -0.547 | -2422 | -1588 ± 192 | 4.2 | 450 |
| Inventory decay, no signal (tau0=0.05h) | +145 ± 160 | 1924 | +0.075 | -1239 | -406 ± 101 | 15.8 | 103 |

Frontier (mean / sd of hourly P&L): Nash by gamma: 0: +3/4094 (g dt 0.89); 0.0001: +33/4052 (g dt 0.89); 0.0003: +90/3932 (g dt 0.89); 0.001: +255/3491 (g dt 0.89); 0.003: +482/2740 (g dt 0.90); 0.01: +628/1990 (g dt 0.93); 0.03: +550/1616 (g dt 1.01); 0.1: +131/1559 (g dt 1.28); 0.3: -4853/7114 (g dt 1.90)
Inventory decay by tau0 (hours): 0.005: +101/1927; 0.01: +109/1922; 0.02: +120/1918; 0.05: +145/1924; 0.1: +177/1939; 0.2: +232/1959; 0.5: +349/1988; 1: +462/2027

## Lit fee 4 bps (risk aversion chosen on training days: gamma* = 0.1)

| Strategy | Mean P&L/h | sd | Sharpe/h | P&L net of client spread | Paired vs Nash | mean abs Q_T (BTC) | Lit volume (BTC/h) |
|---|---|---|---|---|---|---|---|
| Nash broker (closed form) | +1763 ± 228 | 2731 | +0.645 | -1478 | +0 ± 0 | 57.8 | 255 |
| Nash broker, signal off | +1777 ± 229 | 2749 | +0.646 | -1464 | +14 ± 2 | 57.8 | 253 |
| Nash broker, informed-inventory channel off | +9 ± 89 | 1067 | +0.008 | -3232 | -1754 ± 210 | 3.3 | 374 |
| Internalise, liquidate at close | -72 ± 421 | 5047 | -0.014 | -3313 | -1835 ± 491 | 3.3 | 101 |
| Externalise all client flow | -1407 ± 200 | 2399 | -0.586 | -4648 | -3170 ± 382 | 4.2 | 450 |
| Inventory decay, no signal (tau0=0.05h) | +1924 ± 237 | 2846 | +0.676 | -1317 | +161 ± 81 | 15.8 | 103 |
| Learned broker, open loop (MARL) | +1919 ± 245 | 2938 | +0.653 | -1322 | +156 ± 39 | 52.0 | 223 |
| Learned broker, closed loop (MARL) | +1928 ± 248 | 2977 | +0.648 | -1312 | +166 ± 39 | 54.0 | 228 |

Frontier (mean / sd of hourly P&L): Nash by gamma: 0: +1730/3332 (g dt 0.88); 0.0001: +1790/3284 (g dt 0.88); 0.0003: +1888/3223 (g dt 0.88); 0.001: +2100/3191 (g dt 0.88); 0.003: +2313/3335 (g dt 0.88); 0.01: +2396/3508 (g dt 0.90); 0.03: +2251/3358 (g dt 0.93); 0.1: +1763/2731 (g dt 1.05); 0.3: +457/1705 (g dt 1.36)
Inventory decay by tau0 (hours): 0.005: +1839/2784; 0.01: +1857/2792; 0.02: +1880/2807; 0.05: +1924/2846; 0.1: +1974/2902; 0.2: +2047/2994; 0.5: +2186/3190; 1: +2310/3383

## Lit fee 10 bps (risk aversion chosen on training days: gamma* = 0.1)

| Strategy | Mean P&L/h | sd | Sharpe/h | P&L net of client spread | Paired vs Nash | mean abs Q_T (BTC) | Lit volume (BTC/h) |
|---|---|---|---|---|---|---|---|
| Nash broker (closed form) | +5085 ± 657 | 7890 | +0.644 | -1869 | +0 ± 0 | 59.3 | 211 |
| Nash broker, signal off | +5098 ± 659 | 7907 | +0.645 | -1856 | +13 ± 2 | 59.3 | 210 |
| Nash broker, informed-inventory channel off | +2926 ± 424 | 5084 | +0.576 | -4028 | -2159 ± 260 | 3.3 | 306 |
| Internalise, liquidate at close | +1589 ± 617 | 7405 | +0.215 | -5365 | -3496 ± 845 | 3.3 | 101 |
| Externalise all client flow | -2145 ± 298 | 3575 | -0.600 | -9099 | -7230 ± 895 | 4.2 | 450 |
| Inventory decay, no signal (tau0=0.05h) | +5482 ± 677 | 8126 | +0.675 | -1472 | +397 ± 82 | 15.8 | 103 |

Frontier (mean / sd of hourly P&L): Nash by gamma: 0: +5386/7396 (g dt 0.88); 0.0001: +5455/7534 (g dt 0.88); 0.0003: +5561/7758 (g dt 0.88); 0.001: +5762/8238 (g dt 0.88); 0.003: +5930/8757 (g dt 0.88); 0.01: +5944/9052 (g dt 0.88); 0.03: +5716/8831 (g dt 0.90); 0.1: +5085/7890 (g dt 0.96); 0.3: +3811/6162 (g dt 1.11)
Inventory decay by tau0 (hours): 0.005: +5316/7944; 0.01: +5353/7977; 0.02: +5400/8027; 0.05: +5482/8126; 0.1: +5566/8239; 0.2: +5677/8399; 0.5: +5860/8684; 1: +6006/8922

## Data, proxy and model checks (identical across fee levels)

| Client class (test days) | 10 s markout (bps) | 60 s | 300 s | Volume share | Orders |
|---|---|---|---|---|---|
| informed | +0.37 ± 0.04 | +0.30 ± 0.09 | +0.16 ± 0.14 | 49% | 43,350 |
| uninformed | -0.37 ± 0.00 | -0.39 ± 0.01 | -0.42 ± 0.02 | 51% | 963,061 |

| Flow (test days) | Loading on forward drift (t) | Model predicts | Loading on own inventory (t) | Model predicts |
|---|---|---|---|---|
| informed | -28.6 (-4.3) | positive | +44.9 (+6.8) | negative |
| uninformed | -0.1 (-0.0) | zero | +38.7 (+18.1) | zero |

Signal: R^2 of forward drift 0.0017 (train), -0.0037 (test).

Calibrated base parameters (hours, USDT, BTC): T = 1, a = 0.003865, b = 0.003865, c = 0.003865, Ups = 0.09749, Xi = 1.882, kappa_a = 64.98, sig_a = 119, kappa_x = 178.3, sig_x = 6028, sig_S = 22.91

Learned agents in the calibrated simulator: R2_learnedOL_vs_Nash_broker_in_sim = 0.8095, sim_JB_nash = 61.4915, sim_JB_learnedOL = 103.8566, sim_JB_learnedCL = 103.6117
