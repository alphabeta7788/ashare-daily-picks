"""
STAR-evolved picker → daily_picks.json.

Replaces the v1 8-factor heuristic with the STAR-audited winner picker.

Reads:
  - data/ashare/all_a_share_bars.parquet  (price panel)
  - data/ashare/star_picker_winner.py     (STAR-evolved picker — has `def pick(bars)`)

Writes:
  - data/ashare/daily_picks.json          (top-3 with buy/target/stop levels)

Industrial filters (same as STAR evaluator):
  - Liquidity: top 2000 by 20d avg amount
  - Tradability: exclude stocks at 涨停 (pct_chg ≥ 9.5%)
  - Sector dedup: prefer ≥2 different code-prefix groups in top-3
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

DATA = Path("data/ashare")
BARS_FILE = (DATA / "all_a_share_bars.parquet"
             if (DATA / "all_a_share_bars.parquet").exists()
             else DATA / "csi300_bars.parquet")
UNIVERSE_FILE = (DATA / "all_a_share_universe.parquet"
                 if (DATA / "all_a_share_universe.parquet").exists()
                 else DATA / "csi300_universe.parquet")
WINNER_FILE = DATA / "star_picker_winner.py"
OUT = DATA / "daily_picks.json"

LIQUIDITY_TOP_N = 2000
COST_BPS_ROUNDTRIP = 6.0
SECTOR_PREFIX = {
    "600": "上海主板", "601": "上海主板", "603": "上海主板", "605": "上海主板",
    "688": "科创板",
    "000": "深圳主板", "001": "深圳主板", "002": "中小板",
    "300": "创业板", "301": "创业板",
}


def load_star_picker():
    """Exec winner code in a clean namespace; return its pick() function."""
    if not WINNER_FILE.exists():
        return None
    code = WINNER_FILE.read_text()
    ns = {}
    exec(compile(code, str(WINNER_FILE), "exec"), ns)
    fn = ns.get("pick")
    if fn is None:
        raise RuntimeError("star_picker_winner.py has no `def pick(bars)`")
    return fn


def industrial_filters(bars: pl.DataFrame, eval_ts) -> pl.DataFrame:
    """Apply liquidity + tradability filters."""
    today = bars.filter(pl.col("ts") == eval_ts)
    if today.is_empty():
        return pl.DataFrame()
    liq = (
        bars.sort(["symbol", "ts"]).group_by("symbol").tail(20)
        .group_by("symbol").agg(pl.col("amount").mean().alias("avg_amount_20d"))
    ).sort("avg_amount_20d", descending=True).head(LIQUIDITY_TOP_N)
    not_limit_up = today.filter(
        (pl.col("pct_chg") < 9.5) & pl.col("pct_chg").is_not_null()
    ).select(["symbol"])
    return liq.join(not_limit_up, on="symbol", how="inner")


def pick_top3_with_dedup(scored: pl.DataFrame, eligible: pl.DataFrame) -> pl.DataFrame:
    j = scored.join(eligible, on="symbol", how="inner").drop_nulls(["score"])
    if j.height == 0:
        return j
    j = j.sort("score", descending=True)
    # Walk top, prefer ≥2 sectors
    picked = []
    sectors = []
    for row in j.iter_rows(named=True):
        sec = SECTOR_PREFIX.get(row["symbol"][:3], "其他")
        # First pick: anything; pick 2: must be different sector if possible;
        # pick 3: any (but try diversity)
        if len(picked) == 0:
            picked.append(row); sectors.append(sec)
        elif len(picked) == 1:
            if sec != sectors[0]:
                picked.append(row); sectors.append(sec)
        elif len(picked) == 2:
            picked.append(row); sectors.append(sec); break
        if len(picked) == 3:
            break
    # If we didn't get 3 with dedup, fall back to top-3
    if len(picked) < 3:
        picked = list(j.head(3).iter_rows(named=True))
    return pl.DataFrame(picked)


def _features_per_stock(bars: pl.DataFrame, sym: str) -> dict | None:
    s = bars.filter(pl.col("symbol") == sym).sort("ts")
    if s.height < 60:
        return None
    closes = s["close"].to_numpy()
    highs = s["high"].to_numpy()
    lows = s["low"].to_numpy()
    vols = s["volume"].to_numpy()
    if not np.isfinite(closes[-1]) or closes[-1] <= 0:
        return None
    daily_ret = np.diff(closes) / closes[:-1]
    vol_20d = float(np.std(daily_ret[-20:]) * np.sqrt(252))
    tr = np.maximum(highs[-20:] - lows[-20:], np.abs(highs[-20:] - closes[-21:-1]))
    atr = float(np.mean(tr))
    return {
        "last_close": float(closes[-1]),
        "atr": atr,
        "vol_20d": vol_20d,
        "ret_1d": float(closes[-1] / closes[-2] - 1) if closes[-2] > 0 else 0.0,
        "ret_5d": float(closes[-1] / closes[-6] - 1) if closes[-6] > 0 else 0.0,
        "ret_20d": float(closes[-1] / closes[-21] - 1) if closes[-21] > 0 else 0.0,
        "ret_60d": float(closes[-1] / closes[-61] - 1) if closes[-61] > 0 else 0.0,
        "above_ma20": closes[-1] > float(np.mean(closes[-20:])),
        "above_ma60": closes[-1] > float(np.mean(closes[-60:])),
        "vol_spike_5_30": float(np.mean(vols[-5:]) / max(np.mean(vols[-30:]), 1.0)),
        "max_60d": float(np.max(closes[-60:])),
        "breakout_20d": closes[-1] > float(np.max(closes[-60:-20])),
        "kline": [
            {"ts": str(s["ts"][i]), "open": float(s["open"][i]),
             "high": float(s["high"][i]), "low": float(s["low"][i]),
             "close": float(s["close"][i]), "volume": float(s["volume"][i] or 0)}
            for i in range(max(0, s.height - 60), s.height)
        ],
    }


def _decide_levels(feat: dict, expected_ret_hint: float = 0.10) -> dict:
    last = feat["last_close"]
    atr = max(feat["atr"], last * 0.01)
    buy = round(last * 1.005, 2)
    stop = round(last - 1.5 * atr, 2)
    # Target: blend STAR's mean-excess hint with ATR-based
    target = round(last + max(2.5 * atr, last * expected_ret_hint), 2)
    expected_ret = (target - buy) / buy
    risk_ret = (buy - stop) / buy
    holding = (3, 10) if expected_ret < 0.10 else (5, 15)
    return {
        "buy": buy, "target": target, "stop": stop,
        "expected_ret_pct": round(expected_ret * 100, 1),
        "risk_pct": round(risk_ret * 100, 1),
        "rr_ratio": round(expected_ret / max(risk_ret, 0.001), 2),
        "holding_days_min": holding[0],
        "holding_days_max": holding[1],
    }


def main():
    if not WINNER_FILE.exists():
        print(f"[picks-star] no STAR winner at {WINNER_FILE} — fall back to v1 heuristic")
        # Fallback to legacy heuristic
        import subprocess
        subprocess.run(["uv", "run", "python", "scripts/ashare/generate_daily_picks.py"], check=True)
        return 0

    bars = pl.read_parquet(BARS_FILE)
    uni = pl.read_parquet(UNIVERSE_FILE)
    if "symbol" in uni.columns:
        name_map = dict(zip(uni["symbol"], uni["name"]))
    else:
        name_map = {c.split(".")[-1]: n for c, n in zip(uni["code"], uni["code_name"])}

    star_pick_fn = load_star_picker()
    print(f"[picks-star] loaded STAR picker from {WINNER_FILE}")

    eval_ts = bars["ts"].max()
    print(f"[picks-star] eval date: {eval_ts}, universe: {bars.select('symbol').n_unique()} stocks")

    # Run STAR picker on full panel (up to today)
    scored = star_pick_fn(bars)
    if scored is None or scored.is_empty():
        print("[picks-star] STAR picker returned empty — falling back to v1")
        import subprocess
        subprocess.run(["uv", "run", "python", "scripts/ashare/generate_daily_picks.py"], check=True)
        return 0
    print(f"[picks-star] STAR scored {scored.height} stocks")

    # Apply industrial filters
    eligible = industrial_filters(bars, eval_ts)
    print(f"[picks-star] eligible (after liquidity + tradability): {eligible.height}")

    top3 = pick_top3_with_dedup(scored, eligible)
    if top3.height < 3:
        print(f"[picks-star] only {top3.height} stocks survived filters")
        return 1

    picks = []
    for r in top3.iter_rows(named=True):
        sym = r["symbol"]
        feat = _features_per_stock(bars, sym)
        if feat is None:
            continue
        lv = _decide_levels(feat)
        sec = SECTOR_PREFIX.get(sym[:3], "其他")
        picks.append({
            "symbol": sym,
            "name": name_map.get(sym, sym),
            "score": round(float(r["score"]), 3),
            "current_price": round(feat["last_close"], 2),
            "sector": sec,
            "levels": lv,
            "opinion": {
                "one_liner": f"押 {name_map.get(sym, sym)} —— {lv['holding_days_min']}-{lv['holding_days_max']} "
                              f"个交易日博 {lv['expected_ret_pct']}% 涨幅",
                "reasons": [
                    f"STAR 评分 {float(r['score']):.3f}（全A股 {scored.height} 只筛选）",
                    f"流动性：20 日均额 ¥{r['avg_amount_20d']/1e8:.1f} 亿",
                    f"近 20 日 {feat['ret_20d']*100:+.1f}%，近 5 日 {feat['ret_5d']*100:+.1f}%",
                    "STAR 多因子复合（趋势 + 量能 + 突破 + 反转防过热）",
                ],
                "risk_warning": f"止损 {lv['stop']} 元（亏损约 {lv['risk_pct']}%），止损后果断离场",
                "position_advice": "3-5 成仓位试探" if lv["expected_ret_pct"] < 8 else "5-7 成仓位重仓博弈",
            },
            "features": {
                "ret_1d_pct": round(feat["ret_1d"]*100, 2),
                "ret_5d_pct": round(feat["ret_5d"]*100, 2),
                "ret_20d_pct": round(feat["ret_20d"]*100, 2),
                "ret_60d_pct": round(feat["ret_60d"]*100, 2),
                "vol_spike_5_30": round(feat["vol_spike_5_30"], 2),
                "vol_20d_ann_pct": round(feat["vol_20d"]*100, 1),
                "bb_z": 0,  # not computed here (STAR picker uses its own)
                "above_ma20": bool(feat["above_ma20"]),
                "above_ma60": bool(feat["above_ma60"]),
                "breakout_20d": bool(feat["breakout_20d"]),
                "dist_from_60d_high_pct": round((feat["last_close"] - feat["max_60d"]) / feat["max_60d"] * 100, 2),
            },
            "kline": feat["kline"],
        })

    out = {
        "as_of": str(eval_ts),
        "universe": "全A股 (STAR-evolved)",
        "n_candidates_scored": scored.height,
        "n_eligible_after_filters": eligible.height,
        "picker": "star_evolved",
        "picker_source": str(WINNER_FILE),
        "picks": picks,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[picks-star] wrote {OUT}")
    for p in picks:
        print(f"  {p['symbol']} {p['name']:8s} ({p['sector']})  score={p['score']:+.3f}  "
              f"buy {p['levels']['buy']} → target {p['levels']['target']} ({p['levels']['expected_ret_pct']:+.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
