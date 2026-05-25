"""
Render the 'past 60 days shadow performance' panel for the webpage.

Honest reporting:
  - Cumulative NAV of equal-weight portfolio of picks (with target/stop)
  - Cumulative NAV of equal-weight UNIVERSE benchmark (same dates)
  - Excess alpha line
  - Summary stats table
  - Recent 10 closed trades table
  - Verdict (signed by data, not by hope)

Generates HTML fragment → /tmp/shadow_panel.html for inclusion in main page.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

DATA = Path("data/ashare")
LEDGER = DATA / "shadow_portfolio.parquet"
BARS = DATA / "all_a_share_bars.parquet"
OUT = Path("viz/shadow_panel.html")


def build_panel() -> str:
    ledger = pl.read_parquet(LEDGER)
    bars = pl.read_parquet(BARS)
    closed = ledger.filter(pl.col("status") == "closed").sort("entry_decision_date")
    if closed.is_empty():
        return "<div class='shadow-panel empty'>No shadow data yet — first 5-10 days needed before stats meaningful.</div>"

    trading_dates = [str(d)[:10] for d in bars.select("ts").unique().sort("ts")["ts"].to_list()]

    # Build per-date aggregated PnL (avg of 3 picks each day) + universe benchmark
    by_date = {}
    for r in closed.iter_rows(named=True):
        d = r["entry_decision_date"]
        by_date.setdefault(d, []).append(r)

    cost_bps = 6.0
    rows_picks = []   # cum pick avg (with target/stop)
    rows_hold = []    # cum pick 5d hold (no target/stop)
    rows_bench = []   # cum universe equal-weight 5d
    daily_records = []

    for d_dec in sorted(by_date.keys()):
        ps = by_date[d_dec]
        pick_avg = float(np.mean([p["pnl_pct_net"] or 0 for p in ps]))
        # Build benchmark: avg universe equal-weight return from each pick's buy_date to exit_date
        bench_rets = []
        hold_rets = []
        for r in ps:
            d_buy = r["actual_buy_date"]
            if not d_buy or d_buy not in trading_dates: continue
            i_buy = trading_dates.index(d_buy)
            if i_buy + 5 >= len(trading_dates): continue
            d_exit5 = trading_dates[i_buy + 5]
            ts_buy = pl.lit(d_buy).str.to_datetime("%Y-%m-%d").cast(pl.Datetime("ms"))
            ts_exit5 = pl.lit(d_exit5).str.to_datetime("%Y-%m-%d").cast(pl.Datetime("ms"))
            # Universe avg over buy→exit5
            b_in = bars.filter(pl.col("ts") == ts_buy).select(["symbol","open"]).rename({"open":"p_in"})
            b_out = bars.filter(pl.col("ts") == ts_exit5).select(["symbol","open"]).rename({"open":"p_out"})
            j = b_in.join(b_out, on="symbol", how="inner").drop_nulls()
            if not j.is_empty():
                bench_rets.append(float(j.with_columns((pl.col("p_out")/pl.col("p_in") - 1).alias("r"))["r"].mean() or 0))
            # 5d hold of THIS pick
            pick_bar_in = bars.filter((pl.col("symbol") == r["symbol"]) & (pl.col("ts") == ts_buy))
            pick_bar_out = bars.filter((pl.col("symbol") == r["symbol"]) & (pl.col("ts") == ts_exit5))
            if not pick_bar_in.is_empty() and not pick_bar_out.is_empty():
                p_in = float(pick_bar_in["open"][0]); p_out = float(pick_bar_out["open"][0])
                if p_in > 0:
                    hold_rets.append(p_out/p_in - 1 - cost_bps/10000)
        bench_avg = float(np.mean(bench_rets)) if bench_rets else 0.0
        hold_avg = float(np.mean(hold_rets)) if hold_rets else pick_avg
        daily_records.append({
            "date": d_dec,
            "pick": pick_avg,
            "hold": hold_avg,
            "bench": bench_avg,
        })

    dates = [r["date"] for r in daily_records]
    pick_returns = np.array([r["pick"] for r in daily_records])
    hold_returns = np.array([r["hold"] for r in daily_records])
    bench_returns = np.array([r["bench"] for r in daily_records])
    pick_nav = np.cumprod(1 + pick_returns)
    hold_nav = np.cumprod(1 + hold_returns)
    bench_nav = np.cumprod(1 + bench_returns)

    # ─── Chart: cum NAV ───
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=(pick_nav-1)*100, mode="lines+markers",
                              name="STAR picks (with target/stop)",
                              line=dict(color="#dc2626", width=3)))
    fig.add_trace(go.Scatter(x=dates, y=(hold_nav-1)*100, mode="lines",
                              name="STAR picks (5d hold only)",
                              line=dict(color="#f59e0b", dash="dash")))
    fig.add_trace(go.Scatter(x=dates, y=(bench_nav-1)*100, mode="lines",
                              name="Universe 5d equal-weight benchmark",
                              line=dict(color="#16a34a", width=2)))
    fig.add_hline(y=0, line_color="#d1d5db", line_dash="dot")
    fig.update_layout(
        height=320, margin=dict(t=10, b=40, l=50, r=20),
        xaxis_title="决策日", yaxis_title="累计 NAV - 1 (%)",
        plot_bgcolor="#fafafa", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="-apple-system, 'PingFang SC'", size=11),
    )
    chart_html = fig.to_html(include_plotlyjs=False, full_html=False,
                              config={"displayModeBar": False})

    # ─── Summary ───
    all_closed = closed
    cost_arr = all_closed["pnl_pct_net"].to_numpy()
    n_target = all_closed.filter(pl.col("exit_reason") == "target_hit").height
    n_stop = all_closed.filter(pl.col("exit_reason") == "stop_hit").height
    n_time = all_closed.filter(pl.col("exit_reason") == "time_exit").height
    hit_rate = (cost_arr > 0).mean()
    avg_hold = float(all_closed["holding_days"].mean() or 0)
    excess = pick_returns - bench_returns
    excess_sharpe = float(excess.mean()/(excess.std()+1e-9) * np.sqrt(252/5))

    verdict_html = ""
    if excess.mean() > 0.005:
        verdict_html = f'<div class="verdict-good">✅ STAR picker 在最近 {len(daily_records)} 天有正 alpha (+{excess.mean()*100:.2f}%/笔) — 信号成立</div>'
    elif excess.mean() > -0.002:
        verdict_html = f'<div class="verdict-marginal">⚠️ STAR picker 与基准持平（{excess.mean()*100:+.2f}%/笔）— 信号偏弱，建议观察</div>'
    else:
        verdict_html = (f'<div class="verdict-bad">❌ STAR picker 跑输基准 {-excess.mean()*100:.2f}%/笔 '
                        f'(excess Sharpe {excess_sharpe:+.2f}) — 该模型当前不应实盘部署。'
                        f'网页 picks 仅作研究展示，不应据此交易。</div>')

    # ─── Recent closes table ───
    recent = all_closed.sort("actual_exit_date", descending=True).head(10).to_dicts()
    rows_html = []
    for r in recent:
        cls_pnl = "up" if (r["pnl_pct_net"] or 0) > 0 else "down"
        reason_emoji = {"target_hit":"🎯", "stop_hit":"🛑", "time_exit":"⏱️"}.get(r["exit_reason"], "❓")
        rows_html.append(f"""
          <tr>
            <td>{r['entry_decision_date']}</td>
            <td><b>{r['name']}</b> <span class="code-tag">{r['symbol']}</span></td>
            <td>{reason_emoji} {r['exit_reason']}</td>
            <td>{r['holding_days']} 天</td>
            <td class="num {cls_pnl}">{(r['pnl_pct_net'] or 0)*100:+.2f}%</td>
            <td class="num {cls_pnl}">¥{(r['pnl_yuan_per_lot'] or 0):+,.0f}</td>
          </tr>
        """)

    panel = f"""
    <div class="shadow-panel">
      <h2>📊 过去 {len(daily_records)} 天 Shadow 实盘模拟</h2>
      <p class="sub">基于 STAR picker 在历史每个交易日产出 3 个 picks，按当日开盘价买入、按 target/stop/时间出场的<b>真实模拟</b>。结果不掩饰。</p>

      {verdict_html}

      <div class="shadow-stats">
        <div class="stat-cell"><div class="sn">{len(all_closed)}</div><div class="sl">已结束笔数</div></div>
        <div class="stat-cell"><div class="sn {'up' if cost_arr.mean()>0 else 'down'}">{cost_arr.mean()*100:+.2f}%</div><div class="sl">单笔均 PnL</div></div>
        <div class="stat-cell"><div class="sn {'up' if hit_rate>0.5 else 'down'}">{hit_rate*100:.0f}%</div><div class="sl">赢率</div></div>
        <div class="stat-cell"><div class="sn">{avg_hold:.1f} 天</div><div class="sl">平均持仓</div></div>
        <div class="stat-cell"><div class="sn">{n_target} | {n_stop} | {n_time}</div><div class="sl">命中目标｜止损｜时间</div></div>
        <div class="stat-cell"><div class="sn {'up' if excess.mean()>0 else 'down'}">{excess.mean()*100:+.2f}%</div><div class="sl">vs 基准超额</div></div>
      </div>

      <h3 style="margin-top:18px;font-size:14px;color:#374151;">📈 累计 NAV 曲线</h3>
      {chart_html}

      <h3 style="margin-top:18px;font-size:14px;color:#374151;">📋 近 10 笔已结束交易</h3>
      <table class="shadow-table">
        <thead><tr><th>决策日</th><th>标的</th><th>出场原因</th><th>持仓</th><th>收益率</th><th>每手 PnL</th></tr></thead>
        <tbody>{''.join(rows_html)}</tbody>
      </table>
    </div>
    """
    return panel


def main():
    panel = build_panel()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(panel, encoding="utf-8")
    print(f"wrote {OUT} ({len(panel)/1024:.1f} KB)")


if __name__ == "__main__":
    raise SystemExit(main())
