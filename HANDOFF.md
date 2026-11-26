# Handoff — Aurum gold (XAUUSD) research, as of 2026-11-26

Paste this into the next chat: "Read HANDOFF.md and continue."

## What this repo is
Aurum is a research and backtesting framework for gold (XAUUSD). It has 17 strategies: trend, mean reversion, intraday and ML. The backtest protocol is strict:
- walk-forward with purge and embargo
- deflated Sharpe ratio (DSR), using `n_trials`
- PBO
- a final holdout set aside (2025-01-01 onward)

Until 2026-11-26 nothing in `aurum/` had been modified; everything added was a config, a tool or data. On 2026-11-26 the `ml_mlp` strategy was added. It is a new module, `aurum/strategies/ml_mlp.py`, plus one line in the loader list of `aurum/strategies/base.py`. `ml_gbm` itself is untouched.

## Data (already downloaded)
- Real H1 bars from Dukascopy, 2013 to 2026, are in `data_store/`. `manifest.json` was rewritten with `build_path: H1`; the M15 cache only starts 2025-05.
- Dukascopy is **blocked on Telkomsel**, so download on uni wifi instead.
- The macro data (DXY, yields and so on) is in `data_store/macro/`.

## Rule: do not touch the holdout again
- The 2025–26 holdout was looked at **once**, on 2026-11-23 by the trend_core run. That look is recorded in `runs/holdout_ledger.jsonl`.
- Every new config must set `data.end: "2024-12-31T23:59:59Z"` and `walkforward.holdout_start: null`.
- Raise `n_trials` by the number of new strategy variants you test. It is at 35 now (34, plus `ml_mlp`).

## Results so far (none beat buy-and-hold)

| Run (config) | Result | Run dir |
|---|---|---|
| trend_core (default) | OOS 2016–24: −0.8%/yr vs gold +6.6%/yr, PBO 0.97. Holdout 2025–26: +10.7%/yr | `runs/walkforward_20261123T074351Z_06a2fca0` |
| short_term (`configs/short_term.yaml`) | orb −2.9%, rsi2 −2.1%, zscore_fade −2.5%, bollinger_revert −3.4%, intraday_seasonality −0.1% (25 trades) | `runs/walkforward_20261123T093410Z_c7287c20` |
| ml (`configs/ml.yaml`) | ml_gbm 0 trades (all 18 folds failed the skill gate, best AUC about 0.56); meta_label −1.2%/yr | `runs/walkforward_20261123T095317Z_f025c92a` |
| ml_news_base (2017–24 data, OOS 2020–24) | ml_gbm −1.4%/yr (1 of 10 folds traded); meta_label −0.6%/yr; gold +6.3%/yr | `runs/walkforward_20261124T085320Z_ddecfe41` |
| ml_news (same, plus news) | ml_gbm 0 trades (10 of 10 gated); meta_label −1.1%/yr | `runs/walkforward_20261124T085518Z_a061fa46` |
| ml_mlp (`configs/ml_mlp.yaml`, 2013–24, OOS 2016–24) | ml_mlp −0.6%/yr (Sharpe −0.61, 598 trades, max DD −7.6%). It was gated in 17 of 18 folds, with validation AUC from 0.44 to 0.55 (median about 0.51). The one fold that traded, 2021 H1, lost 2.7%. ml_gbm made 0 trades again (18 of 18 gated), the same as the `ml` run. Gold +6.8%/yr. PBO 1.0 is not informative here, because the comparison is effectively ml_mlp against a flat ml_gbm. | `runs/walkforward_20261126T072314Z_fad7ec7b` |

`tools/news_importance.py` showed the news features got **zero** importance, and the validation AUC was 0.44.

## News experiment (free step, GDELT)
- `tools/news_gdelt.py` downloads daily GDELT DOC 2.0 timelines for 6 topics (gold, fed, inflation, dollar, conflict, crisis), each as attention share and tone. That is 12 series in total.
- Point-in-time: day D becomes available at D + 1 day 01:00 UTC.
- **Only 4 of the 12 are cached:** gold share, gold tone, fed share and fed tone. GDELT 429-blocked the uni IP. Try another network, such as a phone hotspot.
  - Resume: `python tools/news_gdelt.py`. Cached series are in `cache/gdelt/`.
  - Build from the cache only: `python tools/news_gdelt.py --cached-only`.
- After downloading, rebuild `data_store/macro_news/` by copying `data_store/macro/*.parquet` and `data_store/news/*.parquet` into it. Then rerun:
  `python -m aurum walkforward --config configs/ml_news.yaml`
  and compare against `configs/ml_news_base.yaml`. Check the exact CLI in the README.
- The news was fed in through the **existing** macro feature group, via `data.macro_dir`. The model code itself is unchanged.
- **Paid step, NOT approved:** have Claude score headlines for sentiment. The estimate is about $2–5 with Haiku 5.5, or $50–100 with Opus 5.5 using the batch API. Do not run it without an explicit yes. It also carries a look-ahead risk: an LLM may remember later events.

## Files we added (committed 2026-11-26 to `main` of the fork `leeve-afk/tradingbot`; data, runs, cache, logs and graphify files stay local)
- `configs/short_term.yaml`
- `configs/ml.yaml`
- `configs/ml_news_base.yaml`
- `configs/ml_news.yaml`
- `tools/news_gdelt.py`
- `tools/news_importance.py`
- `data_store/news/`
- `data_store/macro_news/`
- `cache/gdelt/`
- `logs/*.log`
- `graphify-out/`, `CLAUDE.md`, `.claude/` (from the graphify setup; not committed, and `graphify-out/` is 60 MB, so it is now in `.gitignore`)
- `HANDOFF.md` (this file)
- `aurum/strategies/ml_mlp.py`: the `ml_mlp` strategy (new file), plus one line in `aurum/strategies/base.py` so the strategy loader imports it
- `tests/test_strategies_ml_mlp.py` (13 tests), plus `ml_mlp` added to the module list and `TEST_PARAMS` in `tests/test_strategies_leakage.py`
- `configs/ml_mlp.yaml`

## ml_mlp (built 2026-11-26)
- `ml_mlp` subclasses `ml_gbm`. It keeps ml_gbm's features, labels, uniqueness weights, class balancing, calibration, dead-zone, Carver scaling and skill gate exactly as they are, and swaps only the classifier for `CalibratedMLP`.
- The classifier is an ensemble of 3 sklearn `MLPClassifier` networks with hidden layers 32 and 16, ReLU, L2 penalty 1e-3, Adam and batches of 256.
- Each network trains one epoch at a time and keeps the epoch with the lowest log-loss on the purged, time-ordered validation tail (patience 10, at most 100 epochs). sklearn's own early stopping is off, because it would take a shuffled validation split from overlapping labels, which leaks.
- The design follows Gu, Kelly & Xiu (2020): a shallow pyramid of layers, L2, Adam, early stopping and averaging over seeds.
- The tests check: registry and options, point-in-time behaviour (bit-identical past forecasts when the future is perturbed or cut off), determinism, a random-walk negative control (AUC around 0.5, the gate stays flat), and positive controls on a planted trend and a planted feature.

## Open options (the user hasn't picked one yet)
1. ~~Build `ml_mlp`~~ **Done 2026-11-26** (see above). The neural network found no more signal than the boosted trees: the validation AUCs sit around 0.5. The skill gate needs an AUC of roughly 0.57 or more, because overlapping 24-bar labels leave an effective validation sample of only about 220. Do not tune MLP variants on this same data hoping for a pass. Each variant is a new trial, and the AUC spread looks like noise.
2. **CV write-up**: a 1–2 page research summary built from the real results above. Suggested bullet: "…adding daily GDELT news-attention and tone (sentiment) features to a gradient-boosting model…; no out-of-sample improvement under walk-forward validation with deflated Sharpe." A second bullet for the neural network: "…benchmarked a seed-averaged MLP ensemble (Gu–Kelly–Xiu design) against gradient boosting under an identical purged pipeline; neither cleared a significance gate on validation AUC…".
3. **Finish the 8 missing GDELT series** on another network and rerun the news comparison.
4. ~~Commit the work~~ **Done 2026-11-26.** It is on `main` of the public fork `leeve-afk/tradingbot`, the repo `origin` points to. The original repo `zero-was-here/tradingbot` is the remote called `upstream` (MIT license; keep `LICENSE`).

## Terms to know for interviews
- **Walk-forward validation:** train on a past window, test on the next unseen period, roll forward and repeat.
- **Deflated Sharpe ratio:** a Sharpe ratio adjusted down for how many strategies were tried, so that luck isn't mistaken for skill.
- **Look-ahead bias:** using information you wouldn't have had at the time. It is prevented here by `available_at` timestamps on every data point.
