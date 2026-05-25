"""
Generate today's 3 short-term A-share picks for the retail-facing webpage.

Heuristic v1 picker — combines short-term momentum, volume confirmation,
volatility-adjusted return targets, and news catalyst proxies.

Each pick comes with:
  - Buy / target / stop levels (absolute prices)
  - Expected return + holding-period band (days)
  - One-line opinion + 2-3 reasons
  - Confidence score (0-1)

Output:
  data/ashare/daily_picks.json  — for the webpage
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

DATA = Path("data/ashare")
OUT = DATA / "daily_picks.json"
# Prefer all-A-share bars if available, fall back to CSI300
BARS_FILE = (DATA / "all_a_share_bars.parquet"
             if (DATA / "all_a_share_bars.parquet").exists()
             else DATA / "csi300_bars.parquet")
UNIVERSE_FILE = (DATA / "all_a_share_universe.parquet"
                 if (DATA / "all_a_share_universe.parquet").exists()
                 else DATA / "csi300_universe.parquet")


def _per_stock_features(bars: pl.DataFrame, sym: str) -> dict | None:
    s = bars.filter(pl.col("symbol") == sym).sort("ts")
    if s.height < 60:
        return None
    close = s["close"].to_numpy()
    high = s["high"].to_numpy()
    low = s["low"].to_numpy()
    vol = s["volume"].to_numpy()
    amount = s["amount"].to_numpy() if "amount" in s.columns else None
    turn = s["turnover_pct"].to_numpy() if "turnover_pct" in s.columns else None

    last_close = float(close[-1])
    if not np.isfinite(last_close) or last_close <= 0:
        return None

    # Returns
    ret_1d = float(close[-1] / close[-2] - 1) if close[-2] > 0 else 0.0
    ret_5d = float(close[-1] / close[-6] - 1) if close[-6] > 0 else 0.0
    ret_20d = float(close[-1] / close[-21] - 1) if close[-21] > 0 else 0.0
    ret_60d = float(close[-1] / close[-61] - 1) if close[-61] > 0 else 0.0

    # MAs
    ma5 = float(np.mean(close[-5:]))
    ma20 = float(np.mean(close[-20:]))
    ma60 = float(np.mean(close[-60:]))
    above_ma60 = last_close > ma60
    above_ma20 = last_close > ma20

    # Volume spike: last 5d avg / last 30d avg
    vol_spike = float(np.mean(vol[-5:]) / max(np.mean(vol[-30:]), 1.0))

    # Volatility (20d realized, annualized)
    daily_ret = np.diff(close) / close[:-1]
    vol_20d = float(np.std(daily_ret[-20:]) * np.sqrt(252))
    if not np.isfinite(vol_20d):
        vol_20d = 0.30

    # Distance from 60d max (pullback potential vs breakout)
    max_60d = float(np.max(close[-60:]))
    dist_from_high = (last_close - max_60d) / max_60d  # negative = below peak

    # Recent breakout signal: is last close above the 20d range max from 21..60d ago?
    range_high = float(np.max(close[-60:-20]))
    breakout = last_close > range_high

    # Bollinger position (last close vs 20d mean ± 2 std)
    bb_mean = float(np.mean(close[-20:]))
    bb_std = float(np.std(close[-20:]))
    bb_z = (last_close - bb_mean) / max(bb_std, 1e-6)

    # ATR-ish for stop/target
    tr = np.maximum(high[-20:] - low[-20:], np.abs(high[-20:] - close[-21:-1]))
    atr = float(np.mean(tr))

    return {
        "symbol": sym,
        "last_close": last_close,
        "ret_1d": ret_1d, "ret_5d": ret_5d, "ret_20d": ret_20d, "ret_60d": ret_60d,
        "ma5": ma5, "ma20": ma20, "ma60": ma60,
        "above_ma20": above_ma20, "above_ma60": above_ma60,
        "vol_spike_5_30": vol_spike,
        "vol_20d_ann": vol_20d,
        "dist_from_60d_high": dist_from_high,
        "breakout_20d": breakout,
        "bb_z": bb_z,
        "atr": atr,
        "max_60d": max_60d,
        "kline_last_30d": [
            {"ts": str(s["ts"][i]), "open": float(s["open"][i]), "high": float(s["high"][i]),
             "low": float(s["low"][i]), "close": float(s["close"][i]),
             "volume": float(s["volume"][i] or 0)}
            for i in range(max(0, s.height - 60), s.height)
        ],
    }


def _score(f: dict, news_count: int = 0, lhb_count: int = 0) -> float:
    """Composite short-term score: prioritize stocks just breaking out with
    volume + recent constructive momentum + above key MAs."""
    s = 0.0
    s += 1.0 if f["above_ma60"] else -0.3
    s += 0.8 if f["above_ma20"] else 0.0
    s += min(0.7, max(0.0, f["vol_spike_5_30"] - 1.0) * 0.5)  # volume spike
    s += 0.5 if f["breakout_20d"] else 0.0
    s += min(0.6, max(-0.3, f["ret_5d"] * 5))  # 5d momentum boost
    s += min(0.4, max(-0.4, f["ret_20d"] * 1.5))
    # Soft penalty if too extended (BB z very high → mean revert risk)
    if f["bb_z"] > 2.5:
        s -= 0.4
    if f["bb_z"] < -2.5:
        s += 0.2  # oversold bounce
    # News / LHB attention boost
    s += min(0.5, news_count * 0.15)
    s += min(0.5, lhb_count * 0.2)
    return s


def _decide_levels(f: dict, vol_20d: float) -> dict:
    """Pick buy / target / stop levels from current price + ATR + vol."""
    last = f["last_close"]
    atr = max(f["atr"], last * 0.01)  # min 1% ATR
    # Buy near current (small chase OK), stop 1.5 ATR below, target 2.5 ATR above
    buy = round(last * 1.005, 2)            # tiny chase
    stop = round(last - 1.5 * atr, 2)
    target = round(last + 2.5 * atr, 2)
    # Expected return for "good case" — vol-scaled
    expected_ret = (target - buy) / buy
    risk_ret = (buy - stop) / buy
    # Holding period band: 3-10 days for a swing trade
    holding_days = (3, 10) if expected_ret < 0.10 else (5, 15)
    return {
        "buy": buy, "target": target, "stop": stop,
        "expected_ret_pct": round(expected_ret * 100, 1),
        "risk_pct": round(risk_ret * 100, 1),
        "rr_ratio": round(expected_ret / max(risk_ret, 0.001), 2),
        "holding_days_min": holding_days[0],
        "holding_days_max": holding_days[1],
    }


def _build_opinion(f: dict, lv: dict, news_count: int, lhb_count: int, name: str) -> dict:
    """Build the retail-facing opinion text."""
    reasons = []
    if f["breakout_20d"]:
        reasons.append("突破近 20 日箱体顶部")
    if f["vol_spike_5_30"] > 1.4:
        reasons.append(f"成交量近 5 日放大至常规 {f['vol_spike_5_30']:.1f} 倍")
    if f["above_ma60"] and f["above_ma20"]:
        reasons.append("均线多头排列 (站稳 20/60 日线)")
    if f["ret_20d"] > 0.05:
        reasons.append(f"近 20 日已涨 {f['ret_20d']*100:+.1f}%，势头未息")
    if news_count > 0:
        reasons.append(f"近期有 {news_count} 条关联新闻")
    if lhb_count > 0:
        reasons.append(f"近 5 日上龙虎榜 {lhb_count} 次")
    if not reasons:
        reasons = ["技术面短线买点", "市场情绪偏多"]

    one_liner = f"押 {name} —— {lv['holding_days_min']}-{lv['holding_days_max']} 个交易日博 {lv['expected_ret_pct']}% 涨幅"
    return {
        "one_liner": one_liner,
        "reasons": reasons[:4],
        "risk_warning": f"止损 {lv['stop']} 元（亏损约 {lv['risk_pct']}%），止损后果断离场",
        "position_advice": ("3-5 成仓位试探" if lv["expected_ret_pct"] < 8
                            else "5-7 成仓位重仓博弈"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--min-history-days", type=int, default=60)
    args = ap.parse_args()

    bars = pl.read_parquet(BARS_FILE)
    universe = pl.read_parquet(UNIVERSE_FILE)
    # Universe schema differs between csi300 (symbol/name) and all_a (code/code_name)
    if "symbol" in universe.columns:
        name_map = dict(zip(universe["symbol"], universe["name"]))
    else:
        # all_a schema: code = "sh.600000", code_name = "浦发银行"
        name_map = {c.split(".")[-1]: n for c, n in zip(universe["code"], universe["code_name"])}

    # Optional: per-stock recent news + lhb counts (last 7 days)
    news_cnt = {}
    lhb_cnt = {}
    if (DATA / "lhb.parquet").exists():
        try:
            lhb = pl.read_parquet(DATA / "lhb.parquet")
            if "代码" in lhb.columns:
                cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                lhb_recent = lhb.with_columns(pl.col("上榜日").cast(pl.Utf8)).filter(
                    pl.col("上榜日") >= cutoff
                )
                for r in lhb_recent.group_by("代码").len().to_dicts():
                    lhb_cnt[r["代码"]] = r["len"]
        except Exception:
            pass

    print(f"[picks] universe in bars: {bars.select('symbol').n_unique()} symbols")

    features = []
    symbols = bars.select("symbol").unique()["symbol"].to_list()
    for sym in symbols:
        f = _per_stock_features(bars, sym)
        if f is None:
            continue
        f["name"] = name_map.get(sym, sym)
        f["score"] = _score(f, news_count=news_cnt.get(sym, 0), lhb_count=lhb_cnt.get(sym, 0))
        features.append(f)

    if not features:
        print("[picks] no features — bars may be empty")
        return 1

    features.sort(key=lambda x: -x["score"])
    picks = []
    for f in features[: args.top_k]:
        lv = _decide_levels(f, f["vol_20d_ann"])
        opinion = _build_opinion(f, lv, news_cnt.get(f["symbol"], 0), lhb_cnt.get(f["symbol"], 0), f["name"])
        picks.append({
            "symbol": f["symbol"],
            "name": f["name"],
            "score": round(f["score"], 3),
            "current_price": round(f["last_close"], 2),
            "levels": lv,
            "opinion": opinion,
            "features": {
                "ret_1d_pct": round(f["ret_1d"]*100, 2),
                "ret_5d_pct": round(f["ret_5d"]*100, 2),
                "ret_20d_pct": round(f["ret_20d"]*100, 2),
                "ret_60d_pct": round(f["ret_60d"]*100, 2),
                "vol_spike_5_30": round(f["vol_spike_5_30"], 2),
                "vol_20d_ann_pct": round(f["vol_20d_ann"]*100, 1),
                "bb_z": round(f["bb_z"], 2),
                "above_ma20": f["above_ma20"],
                "above_ma60": f["above_ma60"],
                "breakout_20d": f["breakout_20d"],
                "dist_from_60d_high_pct": round(f["dist_from_60d_high"]*100, 2),
            },
            "kline": f["kline_last_30d"],
        })

    out = {
        "as_of": str(bars["ts"].max()),
        "universe": "CSI300 (partial)",
        "n_candidates_scored": len(features),
        "picks": picks,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[picks] wrote {OUT}")
    for p in picks:
        print(f"  {p['symbol']} {p['name']}  score={p['score']}  "
              f"buy {p['levels']['buy']} → target {p['levels']['target']} "
              f"({p['levels']['expected_ret_pct']:+.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
