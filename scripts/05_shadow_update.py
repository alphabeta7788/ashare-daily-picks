"""
Shadow portfolio simulator for daily A-share picks.

Tracks: each pick made on day D → bought at D+1 open → held until exit.
Exit rule (priority order):
  1. STOP HIT: any subsequent day's low ≤ stop_price → sell at stop_price
  2. TARGET HIT: any subsequent day's high ≥ target_price → sell at target_price
  3. TIME EXIT: hold_days_max (default 10) trading days reached → sell at exit-day open

Cost: 6 bps round-trip (3 bps each side) baked into pnl.

Operating modes:
  --backfill : replay the STAR picker on each of the last N days, simulating
               the full pick → buy → monitor → exit lifecycle. Use to seed
               the shadow ledger with historical data before going live.
  --update   : add today's picks (from data/ashare/daily_picks.json) as new
               open positions, and tick all existing open positions forward
               by one day using new bar data. Idempotent.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import polars as pl

DATA = Path("data/ashare")
LEDGER = DATA / "shadow_portfolio.parquet"
BARS = DATA / "all_a_share_bars.parquet"
PICKS_JSON = DATA / "daily_picks.json"
WINNER = DATA / "star_picker_winner.py"

# Trading constants
HOLD_DAYS_MAX = 10
COST_BPS_ROUNDTRIP = 6.0  # 3 bps each side


LEDGER_SCHEMA = {
    "entry_decision_date": pl.Utf8,   # date picker selected
    "symbol": pl.Utf8,
    "name": pl.Utf8,
    "sector": pl.Utf8,
    "score": pl.Float64,
    "target_price": pl.Float64,
    "stop_price": pl.Float64,
    "actual_buy_date": pl.Utf8,
    "actual_buy_price": pl.Float64,
    "actual_exit_date": pl.Utf8,
    "actual_exit_price": pl.Float64,
    "exit_reason": pl.Utf8,
    "holding_days": pl.Int64,
    "pnl_per_share": pl.Float64,
    "pnl_pct_gross": pl.Float64,
    "pnl_pct_net": pl.Float64,
    "pnl_yuan_per_lot": pl.Float64,
    "status": pl.Utf8,                # "pending" / "open" / "closed"
}


def _empty_ledger() -> pl.DataFrame:
    return pl.DataFrame(schema=LEDGER_SCHEMA)


def _load_ledger() -> pl.DataFrame:
    if not LEDGER.exists():
        return _empty_ledger()
    return pl.read_parquet(LEDGER)


def _save_ledger(df: pl.DataFrame) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    # Force schema for stability
    for col, dt in LEDGER_SCHEMA.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dt, strict=False))
    df.write_parquet(LEDGER)


def _trading_dates(bars: pl.DataFrame) -> list[str]:
    return [str(d)[:10] for d in bars.select("ts").unique().sort("ts")["ts"].to_list()]


def _bar(bars: pl.DataFrame, sym: str, date_str: str) -> dict | None:
    """Get a single bar (symbol, date) as dict, or None if missing."""
    ts = pl.lit(date_str).str.to_datetime("%Y-%m-%d").cast(pl.Datetime("ms"))
    row = bars.filter((pl.col("symbol") == sym) & (pl.col("ts") == ts))
    if row.is_empty():
        return None
    r = row.to_dicts()[0]
    return r


def _load_star_picker():
    if not WINNER.exists():
        return None
    code = WINNER.read_text()
    ns = {}
    exec(compile(code, str(WINNER), "exec"), ns)
    return ns.get("pick")


def _industrial_filter_top3(scored: pl.DataFrame, bars_up_to: pl.DataFrame,
                             eval_date_str: str) -> list[dict]:
    """Same logic as generate_picks_star.py — top 2000 liquidity + no 涨停 + sector dedup."""
    SECTOR = {"600":"上海主板","601":"上海主板","603":"上海主板","605":"上海主板",
              "688":"科创板","000":"深圳主板","001":"深圳主板","002":"中小板",
              "300":"创业板","301":"创业板"}
    ts = pl.lit(eval_date_str).str.to_datetime("%Y-%m-%d").cast(pl.Datetime("ms"))
    today = bars_up_to.filter(pl.col("ts") == ts)
    if today.is_empty():
        return []
    liq = (bars_up_to.sort(["symbol","ts"]).group_by("symbol").tail(20)
           .group_by("symbol").agg(pl.col("amount").mean().alias("avg_amt_20d"))
           .sort("avg_amt_20d", descending=True).head(2000))
    no_limitup = today.filter((pl.col("pct_chg") < 9.5) & pl.col("pct_chg").is_not_null()).select(["symbol"])
    eligible = liq.join(no_limitup, on="symbol", how="inner")
    j = scored.join(eligible, on="symbol", how="inner").drop_nulls(["score"]).sort("score", descending=True)
    # Top-3 with sector dedup (max 2 same sector)
    picked, sectors = [], []
    for r in j.iter_rows(named=True):
        sec = SECTOR.get(r["symbol"][:3], "其他")
        if len(picked) == 0:
            picked.append((r, sec))
        elif len(picked) == 1 and sec != picked[0][1]:
            picked.append((r, sec))
        elif len(picked) >= 1:
            picked.append((r, sec))
        if len(picked) == 3:
            break
    if len(picked) < 3:
        # Fallback: top-3 ignoring dedup
        picked = [(r, SECTOR.get(r["symbol"][:3], "其他")) for r in j.head(3).iter_rows(named=True)]
    return [{"symbol": r["symbol"], "score": float(r["score"]), "sector": s} for r, s in picked]


def _close_position(bars: pl.DataFrame, row: dict, trading_dates: list[str]) -> dict:
    """Walk forward from buy_date and find exit. Returns updated row dict."""
    sym = row["symbol"]
    buy_date = row["actual_buy_date"]
    buy_px = float(row["actual_buy_price"])
    target = float(row["target_price"])
    stop = float(row["stop_price"])
    try:
        i_buy = trading_dates.index(buy_date)
    except ValueError:
        row["status"] = "open"
        return row
    # Walk forward
    for k in range(1, HOLD_DAYS_MAX + 1):
        if i_buy + k >= len(trading_dates):
            # Hit end of available data — leave open
            row["status"] = "open"
            return row
        d = trading_dates[i_buy + k]
        bar = _bar(bars, sym, d)
        if bar is None:
            continue
        hi, lo = float(bar.get("high") or 0), float(bar.get("low") or 0)
        # Stop check first (conservative)
        if lo <= stop:
            exit_px = stop
            row.update(actual_exit_date=d, actual_exit_price=exit_px,
                        exit_reason="stop_hit", holding_days=k)
            break
        if hi >= target:
            exit_px = target
            row.update(actual_exit_date=d, actual_exit_price=exit_px,
                        exit_reason="target_hit", holding_days=k)
            break
    else:
        # Time exit at open of day +HOLD_DAYS_MAX (use that day's open if available, else close of last filled day)
        if i_buy + HOLD_DAYS_MAX < len(trading_dates):
            d_exit = trading_dates[i_buy + HOLD_DAYS_MAX]
            bar = _bar(bars, sym, d_exit)
            if bar is not None:
                row.update(actual_exit_date=d_exit, actual_exit_price=float(bar["open"] or 0),
                            exit_reason="time_exit", holding_days=HOLD_DAYS_MAX)
            else:
                row["status"] = "open"
                return row
        else:
            row["status"] = "open"
            return row

    pnl_per_sh = row["actual_exit_price"] - buy_px
    pnl_pct_gross = pnl_per_sh / buy_px
    pnl_pct_net = pnl_pct_gross - COST_BPS_ROUNDTRIP / 10000.0
    row["pnl_per_share"] = pnl_per_sh
    row["pnl_pct_gross"] = pnl_pct_gross
    row["pnl_pct_net"] = pnl_pct_net
    row["pnl_yuan_per_lot"] = pnl_per_sh * 100
    row["status"] = "closed"
    return row


def _set_buy_price(bars: pl.DataFrame, row: dict, trading_dates: list[str]) -> dict:
    """For a pending pick, set actual_buy_price = next trading day's open after decision."""
    decision = row["entry_decision_date"]
    sym = row["symbol"]
    try:
        i_dec = trading_dates.index(decision)
    except ValueError:
        return row  # decision date not in our data yet
    if i_dec + 1 >= len(trading_dates):
        return row  # next day not yet available
    buy_date = trading_dates[i_dec + 1]
    bar = _bar(bars, sym, buy_date)
    if bar is None:
        return row
    buy_px = float(bar.get("open") or 0)
    if buy_px <= 0:
        return row
    row["actual_buy_date"] = buy_date
    row["actual_buy_price"] = buy_px
    row["status"] = "open"
    return row


def _decide_levels(buy_px: float, atr: float, vol_20d_pct: float) -> tuple[float, float]:
    """Compute target/stop from buy price + ATR + vol."""
    a = max(atr, buy_px * 0.01)
    stop = round(buy_px - 1.5 * a, 2)
    target = round(buy_px + max(2.5 * a, buy_px * 0.10), 2)
    return target, stop


def _atr_for(bars: pl.DataFrame, sym: str, date_str: str, window: int = 20) -> float:
    ts = pl.lit(date_str).str.to_datetime("%Y-%m-%d").cast(pl.Datetime("ms"))
    s = bars.filter((pl.col("symbol") == sym) & (pl.col("ts") <= ts)).sort("ts").tail(window + 1)
    if s.height < 5:
        return 0.0
    h = s["high"].to_numpy()
    l = s["low"].to_numpy()
    c = s["close"].to_numpy()
    tr = np.maximum(h[1:] - l[1:], np.abs(h[1:] - c[:-1]))
    return float(np.mean(tr[-window:]))


def _make_new_picks(bars: pl.DataFrame, decision_date: str,
                     name_map: dict, pick_fn) -> list[dict]:
    """Run STAR picker on bars up to decision_date, return 3 pick rows ready to insert."""
    ts = pl.lit(decision_date).str.to_datetime("%Y-%m-%d").cast(pl.Datetime("ms"))
    bars_up_to = bars.filter(pl.col("ts") <= ts)
    scored = pick_fn(bars_up_to)
    if scored is None or scored.is_empty():
        return []
    top3 = _industrial_filter_top3(scored, bars_up_to, decision_date)
    rows = []
    for p in top3:
        sym = p["symbol"]
        last_bar = _bar(bars_up_to, sym, decision_date)
        if last_bar is None:
            continue
        last_close = float(last_bar["close"] or 0)
        if last_close <= 0:
            continue
        atr = _atr_for(bars_up_to, sym, decision_date)
        # vol_20d
        recent = bars_up_to.filter(pl.col("symbol") == sym).sort("ts").tail(21)
        if recent.height >= 6:
            closes = recent["close"].to_numpy()
            rets = np.diff(closes) / closes[:-1]
            vol20 = float(np.std(rets[-20:]) * np.sqrt(252))
        else:
            vol20 = 0.3
        # Target / stop use TENTATIVE buy = last_close * 1.005 (small chase)
        tentative_buy = last_close * 1.005
        target, stop = _decide_levels(tentative_buy, atr, vol20)
        rows.append({
            "entry_decision_date": decision_date,
            "symbol": sym,
            "name": name_map.get(sym, sym),
            "sector": p["sector"],
            "score": p["score"],
            "target_price": target,
            "stop_price": stop,
            "actual_buy_date": None,
            "actual_buy_price": None,
            "actual_exit_date": None,
            "actual_exit_price": None,
            "exit_reason": None,
            "holding_days": None,
            "pnl_per_share": None,
            "pnl_pct_gross": None,
            "pnl_pct_net": None,
            "pnl_yuan_per_lot": None,
            "status": "pending",
        })
    return rows


def _process_ledger_forward(ledger: pl.DataFrame, bars: pl.DataFrame,
                              trading_dates: list[str]) -> pl.DataFrame:
    """Move pending → open (set buy price), open → closed (find exit)."""
    if ledger.is_empty():
        return ledger
    rows = ledger.to_dicts()
    for r in rows:
        if r["status"] == "pending":
            _set_buy_price(bars, r, trading_dates)
        if r["status"] == "open":
            _close_position(bars, r, trading_dates)
    return pl.DataFrame(rows, schema=LEDGER_SCHEMA)


def cmd_backfill(args) -> int:
    bars = pl.read_parquet(BARS)
    print(f"[shadow] backfill last {args.days} days. Bars: {bars.select('symbol').n_unique()} stocks", flush=True)
    # Universe name lookup
    uni_p = DATA / "all_a_share_universe.parquet"
    if uni_p.exists():
        u = pl.read_parquet(uni_p)
        name_map = {c.split(".")[-1]: n for c, n in zip(u["code"], u["code_name"])}
    else:
        name_map = {}
    pick_fn = _load_star_picker()
    if pick_fn is None:
        print("[shadow] no STAR winner — cannot backfill")
        return 1
    trading_dates = _trading_dates(bars)
    # Use last N trading dates, but leave HOLD_DAYS_MAX dates at the end so positions can close
    end_idx = max(0, len(trading_dates) - HOLD_DAYS_MAX - 1)
    start_idx = max(0, end_idx - args.days + 1)
    decision_dates = trading_dates[start_idx:end_idx + 1]
    print(f"[shadow] decision dates: {decision_dates[0]} → {decision_dates[-1]} ({len(decision_dates)} days)", flush=True)

    new_rows = []
    for i, d in enumerate(decision_dates, 1):
        picks = _make_new_picks(bars, d, name_map, pick_fn)
        new_rows.extend(picks)
        if i % 10 == 0:
            print(f"  {i}/{len(decision_dates)}: {d} → {len(picks)} picks (cum {len(new_rows)})", flush=True)
    if not new_rows:
        print("[shadow] no picks generated")
        return 1
    ledger = pl.DataFrame(new_rows, schema=LEDGER_SCHEMA)
    print(f"[shadow] generated {len(ledger)} pending picks", flush=True)
    # Walk forward to resolve all
    ledger = _process_ledger_forward(ledger, bars, trading_dates)
    _save_ledger(ledger)
    _summary(ledger)
    return 0


def cmd_update(args) -> int:
    """Daily update: walk existing ledger forward + add today's picks (from daily_picks.json)."""
    bars = pl.read_parquet(BARS)
    trading_dates = _trading_dates(bars)
    ledger = _load_ledger()
    print(f"[shadow-update] existing ledger: {len(ledger)} rows", flush=True)
    # Walk forward
    ledger = _process_ledger_forward(ledger, bars, trading_dates)
    # Add today's picks
    if PICKS_JSON.exists():
        picks_doc = json.loads(PICKS_JSON.read_text())
        decision_date = picks_doc["as_of"][:10]
        if not ledger.filter(pl.col("entry_decision_date") == decision_date).is_empty():
            print(f"[shadow-update] picks for {decision_date} already in ledger; skipping", flush=True)
        else:
            new_rows = []
            for p in picks_doc.get("picks", []):
                lv = p["levels"]
                new_rows.append({
                    "entry_decision_date": decision_date,
                    "symbol": p["symbol"],
                    "name": p.get("name") or p["symbol"],
                    "sector": p.get("sector", "?"),
                    "score": p.get("score", 0.0),
                    "target_price": lv["target"],
                    "stop_price": lv["stop"],
                    "actual_buy_date": None,
                    "actual_buy_price": None,
                    "actual_exit_date": None,
                    "actual_exit_price": None,
                    "exit_reason": None,
                    "holding_days": None,
                    "pnl_per_share": None,
                    "pnl_pct_gross": None,
                    "pnl_pct_net": None,
                    "pnl_yuan_per_lot": None,
                    "status": "pending",
                })
            ledger = pl.concat(
                [ledger, pl.DataFrame(new_rows, schema=LEDGER_SCHEMA)],
                how="vertical_relaxed",
            )
            print(f"[shadow-update] added {len(new_rows)} new picks from {decision_date}", flush=True)
            # Walk forward again so pending picks from yesterday get buy_price
            ledger = _process_ledger_forward(ledger, bars, trading_dates)
    _save_ledger(ledger)
    _summary(ledger)
    return 0


def _summary(ledger: pl.DataFrame) -> None:
    if ledger.is_empty():
        print("[shadow] ledger empty")
        return
    closed = ledger.filter(pl.col("status") == "closed")
    n_closed = len(closed)
    print(f"\n=== Shadow Portfolio Summary ===")
    print(f"  total picks: {len(ledger)}")
    print(f"  open: {len(ledger.filter(pl.col('status') == 'open'))}")
    print(f"  pending: {len(ledger.filter(pl.col('status') == 'pending'))}")
    print(f"  closed: {n_closed}")
    if n_closed == 0:
        return
    rets = closed["pnl_pct_net"].to_numpy()
    hits_target = closed.filter(pl.col("exit_reason") == "target_hit").height
    hits_stop = closed.filter(pl.col("exit_reason") == "stop_hit").height
    time_ex = closed.filter(pl.col("exit_reason") == "time_exit").height
    avg_hold = float(closed["holding_days"].mean() or 0)
    print(f"  hit target: {hits_target} ({hits_target/n_closed*100:.0f}%)")
    print(f"  hit stop: {hits_stop} ({hits_stop/n_closed*100:.0f}%)")
    print(f"  time exit: {time_ex} ({time_ex/n_closed*100:.0f}%)")
    print(f"  avg holding days: {avg_hold:.1f}")
    print(f"  mean net PnL/trade: {float(rets.mean())*100:+.2f}%")
    print(f"  median net PnL/trade: {float(np.median(rets))*100:+.2f}%")
    print(f"  cumulative return (equal-weight, sequential): {float(np.sum(rets))*100:+.1f}%")
    print(f"  win rate: {float((rets > 0).mean())*100:.0f}%")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill", help="generate historical shadow picks")
    b.add_argument("--days", type=int, default=60)
    b.set_defaults(fn=cmd_backfill)
    u = sub.add_parser("update", help="incremental daily update")
    u.set_defaults(fn=cmd_update)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
