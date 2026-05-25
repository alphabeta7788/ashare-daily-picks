"""
Generate today's M&A target ranking for A-share CSI300.

Heuristic v1 scorer — combines five validated M&A signals into a composite
ranking. Outputs:
  - data/ashare/ma_targets.json       — machine-readable top-K with reasons
  - data/ashare/ma_targets.md         — human-readable report

Signals:
  1. SMALL CAP            — invert market_cap percentile (smaller = better)
  2. LOW VALUATION        — invert PB percentile + invert |PE| if positive
  3. PRIOR M&A NOISE      — count of M&A-flavored notices for this stock in last 180d
  4. STAGNANT PRICE       — 6m return below median (relative laggard)
  5. INDUSTRY CONSOLIDATION— count of same-industry M&A notices last 180d
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

DATA = Path("data/ashare")
OUT_JSON = DATA / "ma_targets.json"
OUT_MD = DATA / "ma_targets.md"
BARS_FILE = (DATA / "all_a_share_bars.parquet"
             if (DATA / "all_a_share_bars.parquet").exists()
             else DATA / "csi300_bars.parquet")
UNIVERSE_FILE = (DATA / "all_a_share_universe.parquet"
                 if (DATA / "all_a_share_universe.parquet").exists()
                 else DATA / "csi300_universe.parquet")


def _safe_pct_rank(s: pl.Series) -> pl.Series:
    """Cross-section pct rank, NaN-safe (NaNs ranked 0.5)."""
    if s.is_empty():
        return s
    return s.rank(method="average") / max(s.len(), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--lookback-mom", type=int, default=126, help="trading days for stagnant-price signal")
    ap.add_argument("--notice-window-days", type=int, default=180)
    args = ap.parse_args()

    if not BARS_FILE.exists():
        print(f"[ma] missing {BARS_FILE} — run download script first")
        return 1

    bars = pl.read_parquet(BARS_FILE)
    fund = (
        pl.read_parquet(DATA / "csi300_fundamentals.parquet")
        if (DATA / "csi300_fundamentals.parquet").exists()
        else pl.DataFrame()
    )
    notices = (
        pl.read_parquet(DATA / "notices_ma.parquet")
        if (DATA / "notices_ma.parquet").exists()
        else pl.DataFrame()
    )
    universe = pl.read_parquet(UNIVERSE_FILE)
    # Normalize schema: csi300 has (symbol, name); all_a has (code, code_name, tradeStatus)
    if "symbol" not in universe.columns:
        universe = universe.with_columns(
            pl.col("code").map_elements(lambda c: c.split(".")[-1], return_dtype=pl.Utf8).alias("symbol"),
            pl.col("code_name").alias("name"),
        ).select(["symbol", "name"])

    # Build feature panel
    latest_ts = bars["ts"].max()
    n_days = bars.select("ts").n_unique()
    print(f"[ma] universe={universe.height} | bars: {bars.select('symbol').n_unique()} symbols, "
          f"{n_days} days, latest={latest_ts}")
    print(f"[ma] fundamentals: {fund.height} stocks | notices: {notices.height} M&A-flavored")

    # Derive proxy market cap = average daily turnover over last 60 days.
    # In a relative ranking sense, lower turnover → smaller / less liquid stocks,
    # which are precisely the M&A target profile.
    proxy = bars.sort(["symbol", "ts"]).group_by("symbol").tail(60).group_by("symbol").agg(
        pl.col("amount").mean().alias("avg_amount_60d"),
        pl.col("close").last().alias("last_close"),
    ) if "amount" in bars.columns else pl.DataFrame()

    # 1+2. Static features from fundamentals
    feats = universe.select(["symbol", "name"])
    if fund.height:
        feats = feats.join(fund.select(["symbol", "pe", "pb", "market_cap", "industry"]),
                            on="symbol", how="left")
    else:
        feats = feats.with_columns(
            pl.lit(None).alias("pe"), pl.lit(None).alias("pb"),
            pl.lit(None).alias("market_cap"), pl.lit("").alias("industry"),
        )

    # Fallback: use 60d avg amount as a market-cap PROXY when fundamentals missing.
    # Smaller turnover ≈ smaller mcap (loose but useful for full-A ranking).
    if proxy.height:
        feats = feats.join(proxy, on="symbol", how="left")
        # When market_cap missing, fill with proxy
        feats = feats.with_columns(
            pl.when(pl.col("market_cap").is_null())
              .then(pl.col("avg_amount_60d"))
              .otherwise(pl.col("market_cap"))
              .alias("market_cap")
        )

    # 3. Stagnant price signal — 6m return vs median
    sym_returns = []
    for sym in feats["symbol"].to_list():
        s_bars = bars.filter(pl.col("symbol") == sym).sort("ts")
        if s_bars.height < args.lookback_mom + 1:
            sym_returns.append(None)
            continue
        c0 = float(s_bars["close"][-args.lookback_mom - 1])
        c1 = float(s_bars["close"][-1])
        sym_returns.append((c1 / c0 - 1.0) if c0 > 0 else None)
    feats = feats.with_columns(pl.Series("ret_6m", sym_returns))

    # 4. Per-stock prior M&A notice count
    if notices.height and "代码" in notices.columns:
        cutoff = (datetime.now() - timedelta(days=args.notice_window_days)).strftime("%Y-%m-%d")
        recent = notices.with_columns(pl.col("公告日期").cast(pl.Utf8)).filter(
            pl.col("公告日期") >= cutoff
        )
        notice_count = recent.group_by("代码").agg(
            pl.len().alias("ma_notice_count"),
            pl.col("公告标题").str.concat("|").alias("ma_titles_recent"),
        ).rename({"代码": "symbol"})
        feats = feats.join(notice_count, on="symbol", how="left").with_columns(
            pl.col("ma_notice_count").fill_null(0),
            pl.col("ma_titles_recent").fill_null(""),
        )
    else:
        feats = feats.with_columns(
            pl.lit(0).alias("ma_notice_count"),
            pl.lit("").alias("ma_titles_recent"),
        )

    # 5. Industry consolidation signal
    if "industry" in feats.columns:
        ind_counts = feats.group_by("industry").agg(
            pl.col("ma_notice_count").sum().alias("industry_ma_count")
        )
        feats = feats.join(ind_counts, on="industry", how="left")
    else:
        feats = feats.with_columns(pl.lit(0).alias("industry_ma_count"))

    # Compose composite score
    # We invert mcap & pb pctile (smaller = higher M&A appeal),
    # invert ret_6m (more stagnant = more appeal),
    # add log-scaled M&A notice counts.
    import numpy as np
    mcap_rank = _safe_pct_rank(feats["market_cap"]).fill_nan(0.5).fill_null(0.5)
    pb_rank = _safe_pct_rank(feats["pb"]).fill_nan(0.5).fill_null(0.5)
    ret_rank = _safe_pct_rank(feats["ret_6m"]).fill_nan(0.5).fill_null(0.5)
    notice_score = feats["ma_notice_count"].fill_null(0).cast(pl.Float64).log1p()
    ind_score = feats["industry_ma_count"].fill_null(0).cast(pl.Float64).log1p()

    composite = (
        0.30 * (1.0 - mcap_rank)
        + 0.20 * (1.0 - pb_rank)
        + 0.15 * (1.0 - ret_rank)
        + 0.25 * (notice_score / max(float(notice_score.max() or 1), 1))
        + 0.10 * (ind_score / max(float(ind_score.max() or 1), 1))
    )
    feats = feats.with_columns(
        composite.alias("ma_score"),
        mcap_rank.alias("_mcap_pctile"),
        pb_rank.alias("_pb_pctile"),
        ret_rank.alias("_ret6m_pctile"),
        notice_score.alias("_notice_score"),
        ind_score.alias("_ind_score"),
    )

    # Filter to stocks where we have at least mcap data
    candidates = feats.filter(pl.col("market_cap").is_not_null()).sort("ma_score", descending=True)
    topk = candidates.head(args.top_k)

    # JSON output
    out_rows = []
    for r in topk.to_dicts():
        reasons = []
        if r["_mcap_pctile"] < 0.30:
            reasons.append(f"小市值 ({r['_mcap_pctile']*100:.0f}百分位)")
        if r["_pb_pctile"] < 0.30:
            reasons.append(f"低估值 PB {r['pb']:.2f}")
        if r["_ret6m_pctile"] < 0.30:
            reasons.append(f"半年涨幅落后 ({r['ret_6m']*100:+.1f}%)")
        if (r["ma_notice_count"] or 0) > 0:
            reasons.append(f"近180d 有 {r['ma_notice_count']} 条并购相关公告")
        if (r["industry_ma_count"] or 0) > 5:
            reasons.append(f"{r['industry']} 板块在整合 ({r['industry_ma_count']} 条同业公告)")
        is_st = ("ST" in (r["name"] or "")) or ("*ST" in (r["name"] or ""))
        warnings = []
        if "*ST" in (r["name"] or ""):
            warnings.append("⚠️ *ST 高风险 — 有退市可能；并购重组是常见出路但失败也常见")
        elif is_st:
            warnings.append("⚠️ ST 类标的 — 行情会受到风险警示限制，谨慎参与")
        out_rows.append({
            "symbol": r["symbol"],
            "name": r["name"],
            "industry": r["industry"],
            "ma_score": round(r["ma_score"], 4),
            "market_cap_yi": round((r["market_cap"] or 0) / 1e8, 1) if r["market_cap"] else None,
            "pb": round(r["pb"], 2) if r["pb"] else None,
            "pe": round(r["pe"], 2) if r["pe"] else None,
            "ret_6m": round(r["ret_6m"], 4) if r["ret_6m"] is not None else None,
            "ma_notices_180d": r["ma_notice_count"],
            "industry_ma_notices_180d": r["industry_ma_count"],
            "is_st": is_st,
            "reasons": reasons,
            "warnings": warnings,
            "recent_notice_sample": (r["ma_titles_recent"][:200] if r["ma_titles_recent"] else ""),
        })

    OUT_JSON.write_text(json.dumps(
        {"as_of": str(latest_ts), "universe": "CSI300", "top": out_rows},
        ensure_ascii=False, indent=2,
    ))

    # Markdown report
    md = [f"# A股并购标的日榜 — {latest_ts}\n",
          f"Universe: CSI300，Top-{args.top_k} | 基于 5 因子启发式打分（小市值 / 低估值 / 涨幅落后 / 历史并购公告 / 行业整合）\n"]
    md.append("| 排名 | 代码 | 名称 | 行业 | M&A 分 | 市值(亿) | PB | 半年收益 | 公告数 | 主要理由 |")
    md.append("|---:|:--|:--|:--|---:|---:|---:|---:|---:|:--|")
    for i, r in enumerate(out_rows, 1):
        md.append(
            f"| {i} | {r['symbol']} | **{r['name']}** | {r['industry'] or '-'} | "
            f"{r['ma_score']:.3f} | {r['market_cap_yi'] or '?'} | "
            f"{r['pb'] or '?'} | "
            f"{(r['ret_6m']*100):+.1f}% | "
            f"{r['ma_notices_180d']} | {'；'.join(r['reasons']) or '综合评分'} |"
        )
    md.append("\n## 说明\n")
    md.append("- 评分越高越有可能成为并购标的；")
    md.append("- 历史经验：标的公告后 5-20 个交易日通常跳涨 15-50%；")
    md.append("- 本榜单为启发式打分（v1），不构成投资建议；")
    md.append(f"- 数据局限：当前可用样本 {bars.select('symbol').n_unique()} 只（CSI300）。")
    OUT_MD.write_text("\n".join(md))

    print(f"[ma] wrote {OUT_JSON} ({len(out_rows)} entries)")
    print(f"[ma] wrote {OUT_MD}")
    print(f"\nTop 5 preview:")
    for r in out_rows[:5]:
        print(f"  {r['symbol']} {r['name']:8s}  score={r['ma_score']:.3f}  reasons: {'; '.join(r['reasons'][:2])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
