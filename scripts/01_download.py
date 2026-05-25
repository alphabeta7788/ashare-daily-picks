"""
Parallel baostock downloader for A-share bars.
4 worker PROCESSES (each own baostock login). 5207 stocks / 6 per sec = ~15 min.

For CI: keep only last 200 trading days to keep file small + fast.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

OUT = Path("data/ashare")
BARS_FILE = OUT / "all_a_share_bars.parquet"
NUMERIC_COLS = ["open", "high", "low", "close", "volume", "amount", "turnover_pct", "pct_chg"]


def _bs_symbol(code: str) -> str:
    return f"sh.{code}" if code.startswith(("6", "9")) else f"sz.{code}"


def _worker(args):
    """Each worker process: own baostock login, fetch its chunk."""
    codes, start, end = args
    import baostock as bs
    lg = bs.login()
    if lg.error_code != "0":
        return []
    out = []
    for code in codes:
        try:
            rs = bs.query_history_k_data_plus(
                _bs_symbol(code),
                "date,open,high,low,close,volume,amount,turn,pctChg",
                start_date=start, end_date=end, frequency="d", adjustflag="2",
            )
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if rows:
                out.append((code, rows, rs.fields))
        except Exception:
            pass
    bs.logout()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=(datetime.now() - timedelta(days=300)).strftime("%Y-%m-%d"))
    ap.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    uni = pl.read_parquet(OUT / "all_a_share_universe.parquet")
    codes = [c.split(".")[-1] for c in uni["code"].to_list()]
    print(f"[dl] {len(codes)} stocks, workers={args.workers}, "
          f"date range {args.start}..{args.end}", flush=True)

    # Split codes into workers chunks
    n = args.workers
    chunks = [codes[i::n] for i in range(n)]
    t0 = time.time()

    with mp.Pool(args.workers) as pool:
        results = pool.map(_worker, [(c, args.start, args.end) for c in chunks])

    # Flatten + convert to polars
    parts = []
    fails = 0
    for chunk_res in results:
        for code, rows, fields in chunk_res:
            df = pl.DataFrame(rows, schema=fields, orient="row")
            p = df.with_columns(
                pl.col("date").str.to_datetime("%Y-%m-%d").cast(pl.Datetime("ms")).alias("ts"),
                pl.lit(code).alias("symbol"),
            ).rename({"turn": "turnover_pct", "pctChg": "pct_chg"})
            for c in NUMERIC_COLS:
                if c in p.columns:
                    p = p.with_columns(pl.col(c).cast(pl.Float64, strict=False))
            parts.append(p.select([c for c in ["ts","symbol"] + NUMERIC_COLS if c in p.columns]).drop_nulls(["ts"]))
    got = len(parts)
    fails = len(codes) - got
    print(f"[dl] got {got}/{len(codes)} stocks in {time.time()-t0:.0f}s, {fails} fails", flush=True)

    if not parts:
        print("[dl] nothing downloaded")
        return 1

    merged = pl.concat(parts).unique(subset=["ts","symbol"]).sort(["symbol","ts"])
    merged.write_parquet(BARS_FILE)
    print(f"[dl] saved {merged.select('symbol').n_unique()} stocks, {len(merged)} rows → {BARS_FILE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
