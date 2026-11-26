"""Which clues does ml_gbm lean on? Fit it once on 2019-2023 (with the news macro dir) and print
the permutation importance by feature family, plus the news columns and the validation AUC.

    python tools/news_importance.py [--config configs/ml_news.yaml] [--start 2019-01-01] [--end 2023-12-31]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ml_news.yaml")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2023-12-31T23:59:59Z")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    from aurum.core.config import load_config
    from aurum.strategies.base import get_strategy

    cfg = load_config(args.config, overrides=[f"data.start={args.start}", f"data.end={args.end}"])
    md = cfg.data.load()
    strat = get_strategy("ml_gbm", skill_gate_z=None)
    strat.fit(md)
    rep = strat.fit_report_
    print(f"validation AUC: {rep.get('val_auc', float('nan')):.4f} (0.50 = coin flip)")
    imp = strat.feature_importances_
    by_group = strat.importance_by_group()
    total = float(by_group.clip(lower=0).sum()) or 1.0
    print("\nimportance by feature family (share of total positive importance):")
    for fam, v in by_group.items():
        print(f"  {fam:16s} {v:+.5f}  ({max(v, 0) / total:5.1%})")
    news = imp[imp.index.str.contains("news")].sort_values(ascending=False)
    print(f"\nnews columns ({len(news)}):")
    for c, v in news.items():
        print(f"  {c:40s} {v:+.5f}")
    print("\ntop 10 columns overall:")
    for c, v in imp.sort_values(ascending=False).head(10).items():
        print(f"  {c:40s} {v:+.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
