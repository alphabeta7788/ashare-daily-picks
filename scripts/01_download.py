"""
Incremental A-share bars downloader.

Default mode (--incremental): read existing bars parquet, find max date,
only fetch new days from baostock since then. Saves 90%+ time on daily runs.

Full mode (--full): redownload N days from scratch. Use weekly or after data
loss.

Rolling window: keep only last ROLLING_DAYS to prevent unbounded growth.
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
ROLLING_DAYS = 365  # trim to last N days after merge


def _bs_symbol(code: str) -> str:
    return f"sh.{code}" if code.startswith(("6", "9")) else f"sz.{code}"


def _worker(args):
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


def _rows_to_polars(rows, code) -> pl.DataFrame:
    df = pl.DataFrame(rows[1], schema=rows[2], orient="row")
    p = df.with_columns(
        pl.col("date").str.to_datetime("%Y-%m-%d").cast(pl.Datetime("ms")).alias("ts"),
        pl.lit(code).alias("symbol"),
    )
    rename = {"turn": "turnover_pct", "pctChg": "pct_chg"}
    for src, dst in rename.items():
        if src in p.columns:
            p = p.rename({src: dst})
    for c in NUMERIC_COLS:
        if c in p.columns:
            p = p.with_columns(pl.col(c).cast(pl.Float64, strict=False))
    return p.select([c for c in ["ts", "symbol"] + NUMERIC_COLS if c in p.columns]).drop_nulls(["ts"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--incremental", action="store_true", default=True,
                    help="(default) Fetch only since max date in existing bars")
    ap.add_argument("--full", action="store_true",
                    help="Override: redownload N days from scratch")
    ap.add_argument("--full-days", type=int, default=300,
                    help="If --full, how many days back")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--safety-overlap-days", type=int, default=5,
                    help="Re-fetch last N days even in incremental mode (handle late-arriving data)")
    args = ap.parse_args()

    today = datetime.now().strftime("%Y-%m-%d")
    end = today
    uni = pl.read_parquet(OUT / "all_a_share_universe.parquet")
    codes = [c.split(".")[-1] for c in uni["code"].to_list()]

    # Determine start date
    if args.full or not BARS_FILE.exists():
        start = (datetime.now() - timedelta(days=args.full_days)).strftime("%Y-%m-%d")
        mode = f"FULL ({args.full_days}d)"
        existing = pl.DataFrame()
    else:
        existing = pl.read_parquet(BARS_FILE)
        max_ts = existing["ts"].max()
        max_date_str = str(max_ts)[:10]
        # Re-fetch with overlap to handle late corrections
        start_dt = datetime.strptime(max_date_str, "%Y-%m-%d") - timedelta(days=args.safety_overlap_days)
        start = start_dt.strftime("%Y-%m-%d")
        mode = f"INCREMENTAL (since {max_date_str}, overlap {args.safety_overlap_days}d)"

    print(f"[dl] {mode}  range {start} → {end}  workers={args.workers}", flush=True)
    if start >= end:
        print(f"[dl] already up to date, nothing to fetch", flush=True)
        return 0

    # Parallel download
    n = args.workers
    chunks = [codes[i::n] for i in range(n)]
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        results = pool.map(_worker, [(c, start, end) for c in chunks])

    parts = []
    for chunk_res in results:
        for code, rows, fields in chunk_res:
            try:
                parts.append(_rows_to_polars((None, rows, fields), code))
            except Exception:
                pass
    print(f"[dl] fetched {len(parts)} stocks in {time.time()-t0:.0f}s", flush=True)

    if not parts:
        print("[dl] nothing fetched")
        return 1
    new = pl.concat(parts).unique(subset=["ts", "symbol"]).sort(["symbol", "ts"])

    if existing.height > 0:
        # Align schemas
        for c in NUMERIC_COLS:
            if c in existing.columns:
                existing = existing.with_columns(pl.col(c).cast(pl.Float64, strict=False))
        merged = pl.concat([existing, new], how="vertical_relaxed").unique(
            subset=["ts", "symbol"]
        ).sort(["symbol", "ts"])
    else:
        merged = new

    # Trim to rolling window
    cutoff = (datetime.now() - timedelta(days=ROLLING_DAYS)).strftime("%Y-%m-%d")
    cutoff_ts = pl.lit(cutoff).str.to_datetime("%Y-%m-%d").cast(pl.Datetime("ms"))
    n_before = len(merged)
    merged = merged.filter(pl.col("ts") >= cutoff_ts)
    n_after = len(merged)
    if n_before != n_after:
        print(f"[dl] trimmed {n_before-n_after} rows older than {cutoff}", flush=True)

    merged.write_parquet(BARS_FILE)
    print(f"[dl] saved {merged.select('symbol').n_unique()} stocks, "
          f"{len(merged)} rows, file size {BARS_FILE.stat().st_size/1024/1024:.1f}MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
