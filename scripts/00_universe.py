"""Fetch A-share universe (~5200 stocks) via baostock. Standalone, CI-friendly."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import baostock as bs
import polars as pl

OUT = Path("data/ashare")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    lg = bs.login()
    if lg.error_code != "0":
        print(f"login fail: {lg.error_msg}")
        return 1

    # Find latest trading day with data — try recent days backwards
    today = dt.date.today()
    rows = []
    for back in range(14):
        d = (today - dt.timedelta(days=back)).strftime("%Y-%m-%d")
        rs = bs.query_all_stock(day=d)
        tmp = []
        while rs.next():
            tmp.append(rs.get_row_data())
        if tmp:
            rows = tmp
            print(f"using {d} for universe: {len(tmp)} symbols")
            break
    bs.logout()
    if not rows:
        print("could not find universe data")
        return 1

    df = pl.DataFrame(rows, schema=rs.fields, orient="row")
    a = df.filter(
        pl.col("code").str.starts_with("sh.60")
        | pl.col("code").str.starts_with("sh.68")
        | pl.col("code").str.starts_with("sz.00")
        | pl.col("code").str.starts_with("sz.30")
    )
    a.write_parquet(OUT / "all_a_share_universe.parquet")
    print(f"saved {len(a)} A-shares → all_a_share_universe.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
