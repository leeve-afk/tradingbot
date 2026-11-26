"""Download daily gold-relevant news series from the GDELT DOC 2.0 API (free, 2017+).

For each topic query it fetches two daily timelines:

* ``news_{topic}_share`` - share of all monitored articles that match the query
  (``timelinevolraw``: matching count / total count), a measure of how much the world talks
  about the topic;
* ``news_{topic}_tone``  - average tone of the matching articles (``timelinetone``, roughly
  -10 very negative .. +10 very positive).

Point-in-time: a value for UTC day D covers articles published during D, so it is stamped
``available_at = D + 1 day + 1 hour`` (the day must be over; GDELT ingests with ~15 min lag).

Output: one parquet per series in the aurum macro format (``value`` + ``available_at``,
see ``aurum.data.macro.save_macro_dir``), written to ``--out`` (default
``data_store/news``). Raw API responses are cached in ``--cache`` so re-runs resume.

The API asks for at most one request every 5 seconds; this script waits between requests
and backs off when told to slow down.

Usage:
    python tools/news_gdelt.py                       # 2017-01-01 .. 2024-12-31
    python tools/news_gdelt.py --start 2017-01-01 --end 2026-11-20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

API = "https://api.gdeltproject.org/api/v2/doc/doc"

#: topic -> GDELT query (phrases in double quotes, OR needs parentheses)
TOPICS: dict[str, str] = {
    "gold": '("gold price" OR "gold prices" OR "price of gold")',
    "fed": '("federal reserve" OR "interest rate hike" OR "rate cut")',
    "inflation": '("inflation")',
    "dollar": '("us dollar" OR "dollar index")',
    "conflict": '(war OR invasion OR airstrike OR missile)',
    "crisis": '(recession OR "financial crisis" OR "bank collapse")',
}
MODES = {"share": "timelinevolraw", "tone": "timelinetone"}
AVAILABLE_LAG = pd.Timedelta(days=1, hours=1)


def _cache_path(cache: Path, query: str, mode: str, t0: pd.Timestamp, t1: pd.Timestamp) -> Path:
    qh = hashlib.sha1(query.encode()).hexdigest()[:10]
    return cache / f"{mode}_{qh}_{t0:%Y%m%d}_{t1:%Y%m%d}.json"


def _fetch(query: str, mode: str, t0: pd.Timestamp, t1: pd.Timestamp, cache: Path,
           min_gap: float, state: dict) -> dict:
    path = _cache_path(cache, query, mode, t0, t1)
    if path.exists():
        return json.loads(path.read_text())
    params = {"query": query, "mode": mode, "format": "json",
              "startdatetime": f"{t0:%Y%m%d%H%M%S}", "enddatetime": f"{t1:%Y%m%d%H%M%S}"}
    url = f"{API}?{urlencode(params)}"
    wait = min_gap
    for attempt in range(20):
        gap = time.time() - state["last"]
        if gap < wait:
            time.sleep(wait - gap)
        state["last"] = time.time()
        try:
            r = requests.get(url, timeout=120, headers={"User-Agent": "aurum-research/1.0"})
            text = r.text.strip()
        except requests.RequestException as exc:  # network hiccup: back off and retry
            print(f"    network error ({exc.__class__.__name__}); retrying", flush=True)
            wait = min(wait * 2, 600)
            continue
        if r.status_code == 200 and text.startswith("{"):
            data = json.loads(text)
            path.write_text(json.dumps(data))
            return data
        wait = min(wait * 2, 600)
        print(f"    slow-down / HTTP {r.status_code} (attempt {attempt + 1}); waiting {wait:.0f}s",
              flush=True)
    raise RuntimeError(f"GDELT kept refusing: {url}")


def _series(data: dict, mode: str) -> pd.Series:
    tl = data.get("timeline") or []
    if not tl:
        return pd.Series(dtype=float)
    rows = tl[0].get("data") or []
    idx = pd.to_datetime([r["date"] for r in rows], format="%Y%m%dT%H%M%SZ", utc=True)
    if mode == "timelinevolraw":
        vals = [r["value"] / r["norm"] if r.get("norm") else float("nan") for r in rows]
    else:
        vals = [float(r["value"]) for r in rows]
    return pd.Series(vals, index=idx, dtype=float)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--out", default="data_store/news")
    ap.add_argument("--cache", default="cache/gdelt")
    ap.add_argument("--min-gap", type=float, default=15.0, help="seconds between requests")
    ap.add_argument("--cached-only", action="store_true",
                    help="no network: build files only from series already in the cache")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from aurum.data.macro import save_macro_dir

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    # one request per series: the API returns daily resolution even for multi-year spans
    windows = [(start, end + pd.Timedelta(days=1))]
    state = {"last": 0.0}
    frames: dict[str, pd.DataFrame] = {}
    total = len(TOPICS) * len(MODES) * len(windows)
    n = 0
    for topic, query in TOPICS.items():
        for label, mode in MODES.items():
            if args.cached_only and not all(_cache_path(cache, query, mode, a, b).exists()
                                            for a, b in windows):
                print(f"  skip news_{topic}_{label}: not cached", flush=True)
                continue
            parts = []
            for t0, t1 in windows:
                n += 1
                print(f"[{n}/{total}] {topic} {label} {t0:%Y}", flush=True)
                parts.append(_series(_fetch(query, mode, t0, t1, cache, args.min_gap, state), mode))
            s = pd.concat(parts).sort_index()
            s = s[~s.index.duplicated(keep="last")]
            s = s[(s.index >= start.tz_localize("UTC")) & (s.index < (end + pd.Timedelta(days=1)).tz_localize("UTC"))]
            df = pd.DataFrame({"value": s})
            df.index.name = "date"
            df["available_at"] = df.index + AVAILABLE_LAG
            # tone can be negative: difference it (bp-style) instead of log-differencing
            df.attrs["kind"] = "level" if label == "tone" else "price"
            df.attrs["source"] = f"GDELT DOC 2.0 {mode} {query}"
            frames[f"news_{topic}_{label}"] = df
    path = save_macro_dir(frames, args.out)
    for name, df in sorted(frames.items()):
        print(f"  {name}: {len(df)} days {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"saved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
