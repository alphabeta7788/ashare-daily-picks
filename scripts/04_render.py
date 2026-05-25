"""
Daily picks webpage v2 — creative retail-focused design.

NEW vs v1:
  - Market thermometer (gauge) at top
  - "如果买1万元" widget per pick (best/avg/worst case)
  - Monte Carlo cone: 100 simulated next-10-day paths
  - Confidence dial per pick
  - M&A radar bubble chart (size = M&A score, color = industry)
  - "复制到雪球" buttons
  - Improved typography + dark hero
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

DATA = Path("data/ashare")
IN_PICKS = DATA / "daily_picks.json"
IN_MA = DATA / "ma_targets.json"
BARS = (DATA / "all_a_share_bars.parquet"
        if (DATA / "all_a_share_bars.parquet").exists()
        else DATA / "csi300_bars.parquet")
SHADOW_PANEL = Path("viz/shadow_panel.html")
OUT = Path("viz/daily_picks.html")


# ────────────────────────────────────────────────────────────────────
# Market thermometer — uses universe-wide avg 5d return as the "fever"

def _market_temp() -> tuple[int, str, str, dict]:
    """Returns (temp 0-100, mood label, color hex, breadth_metrics).

    breadth_metrics includes today's % up, # limit-up (涨停 ~9.8%+ for non-ST,
    or 19.8%+ for ST/科创/创业板), # limit-down, total universe size.
    """
    bars = pl.read_parquet(BARS)
    latest_ts = bars["ts"].max()
    # Per-symbol latest pct_chg + 5d ret
    by_sym = bars.sort(["symbol", "ts"]).group_by("symbol").agg(
        pl.col("close").last().alias("c1"),
        pl.col("close").shift(5).last().alias("c0"),
        pl.col("pct_chg").last().alias("last_pct"),
    ).with_columns(
        (pl.col("c1") / pl.col("c0") - 1).alias("ret5")
    )
    rets5 = by_sym["ret5"].drop_nulls().to_list()
    pcts = by_sym["last_pct"].drop_nulls().to_list()
    if not rets5:
        return 50, "持平", "#9ca3af", {}

    median_ret = float(np.median(rets5))
    pct_up_today = sum(1 for p in pcts if p > 0) / max(len(pcts), 1)
    n_limit_up = sum(1 for p in pcts if p >= 9.5)
    n_limit_down = sum(1 for p in pcts if p <= -9.5)
    n_total = len(pcts)

    breadth = {
        "universe_size": n_total,
        "pct_up_today": round(pct_up_today * 100, 1),
        "n_limit_up": n_limit_up,
        "n_limit_down": n_limit_down,
        "median_5d_ret_pct": round(median_ret * 100, 2),
    }

    temp = int(np.clip(50 + median_ret * 1000 + (pct_up_today - 0.5) * 30, 5, 95))
    if temp >= 75:
        return temp, "热得发烫 🔥", "#dc2626", breadth
    if temp >= 60:
        return temp, "偏热", "#f59e0b", breadth
    if temp >= 40:
        return temp, "温和", "#16a34a", breadth
    if temp >= 25:
        return temp, "偏冷", "#3b82f6", breadth
    return temp, "冷飕飕 🥶", "#1e40af", breadth


def _temp_gauge(temp: int, color: str) -> str:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=temp,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "今日市场体温", "font": {"size": 16, "color": "white"}},
        number={"font": {"size": 56, "color": "white"}, "suffix": "°"},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "rgba(255,255,255,0.3)",
                     "tickfont": {"color": "rgba(255,255,255,0.6)"}},
            "bar": {"color": color, "thickness": 0.7},
            "bgcolor": "rgba(0,0,0,0.4)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 25], "color": "rgba(30, 64, 175, 0.3)"},
                {"range": [25, 40], "color": "rgba(59, 130, 246, 0.3)"},
                {"range": [40, 60], "color": "rgba(22, 163, 74, 0.3)"},
                {"range": [60, 75], "color": "rgba(245, 158, 11, 0.3)"},
                {"range": [75, 100], "color": "rgba(220, 38, 38, 0.3)"},
            ],
        },
    ))
    fig.update_layout(
        height=240, margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system, 'PingFang SC'"),
    )
    return fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="temp_gauge",
                       config={"displayModeBar": False})


# ────────────────────────────────────────────────────────────────────
# K-line with buy/target/stop overlay

def _kline_chart(pick: dict) -> str:
    kl = pick["kline"]
    if not kl:
        return ""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03,
    )
    fig.add_trace(go.Candlestick(
        x=[k["ts"][:10] for k in kl],
        open=[k["open"] for k in kl], high=[k["high"] for k in kl],
        low=[k["low"] for k in kl], close=[k["close"] for k in kl],
        increasing_line_color="#dc2626", decreasing_line_color="#16a34a",
        name="K", showlegend=False,
    ), row=1, col=1)
    lv = pick["levels"]
    for level, color, label in [
        (lv["target"], "#dc2626", f"🎯 目标 {lv['target']}"),
        (lv["buy"], "#1d4ed8", f"💰 买入 {lv['buy']}"),
        (lv["stop"], "#374151", f"🛑 止损 {lv['stop']}"),
    ]:
        fig.add_hline(y=level, line_color=color, line_dash="dash", line_width=2,
                       annotation_text=label, annotation_position="right",
                       annotation_font=dict(color=color, size=11), row=1, col=1)
    fig.add_trace(go.Bar(
        x=[k["ts"][:10] for k in kl], y=[k["volume"] for k in kl],
        marker_color=["#fca5a5" if k["close"] >= k["open"] else "#86efac" for k in kl],
        showlegend=False,
    ), row=2, col=1)
    fig.update_layout(
        height=340, margin=dict(t=10, b=20, l=40, r=80),
        xaxis_rangeslider_visible=False,
        plot_bgcolor="#fafafa", paper_bgcolor="white",
        font=dict(family="-apple-system, 'PingFang SC'", size=11),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")
    return fig.to_html(include_plotlyjs=False, full_html=False,
                       config={"displayModeBar": False})


# ────────────────────────────────────────────────────────────────────
# Monte Carlo cone — 100 GBM paths over next 10 days based on historical vol

def _monte_carlo_cone(pick: dict) -> str:
    kl = pick["kline"]
    closes = np.array([k["close"] for k in kl])
    if len(closes) < 20:
        return ""
    rets = np.diff(closes) / closes[:-1]
    mu = float(np.mean(rets))
    sigma = float(np.std(rets))
    last = float(closes[-1])
    n_paths = 100
    n_days = 10
    rng = np.random.default_rng(42)
    shocks = rng.normal(mu, sigma, size=(n_paths, n_days))
    paths = np.cumprod(1 + shocks, axis=1) * last
    days = list(range(1, n_days + 1))

    fig = go.Figure()
    for i in range(n_paths):
        fig.add_trace(go.Scatter(
            x=days, y=paths[i], mode="lines",
            line=dict(color="rgba(220, 38, 38, 0.08)", width=1),
            showlegend=False, hoverinfo="skip",
        ))
    # Mean + 25/75 percentile bands
    mean_path = paths.mean(axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p10 = np.percentile(paths, 10, axis=0)
    p90 = np.percentile(paths, 90, axis=0)
    fig.add_trace(go.Scatter(x=days, y=p90, mode="lines",
                              line=dict(color="rgba(0,0,0,0)"), showlegend=False))
    fig.add_trace(go.Scatter(x=days, y=p10, mode="lines",
                              line=dict(color="rgba(0,0,0,0)"),
                              fill="tonexty", fillcolor="rgba(220, 38, 38, 0.10)",
                              showlegend=False))
    fig.add_trace(go.Scatter(x=days, y=mean_path, mode="lines+markers",
                              line=dict(color="#dc2626", width=3),
                              marker=dict(size=4), name="均值路径"))
    fig.add_hline(y=pick["levels"]["target"], line_color="#dc2626", line_dash="dash",
                   annotation_text="🎯", annotation_position="left")
    fig.add_hline(y=pick["levels"]["stop"], line_color="#374151", line_dash="dash",
                   annotation_text="🛑", annotation_position="left")
    fig.update_layout(
        height=240, margin=dict(t=20, b=30, l=40, r=20),
        xaxis_title="未来交易日", yaxis_title="价格",
        plot_bgcolor="#fafafa", paper_bgcolor="white",
        font=dict(family="-apple-system, 'PingFang SC'", size=11),
        showlegend=False,
    )
    return fig.to_html(include_plotlyjs=False, full_html=False,
                       config={"displayModeBar": False})


# ────────────────────────────────────────────────────────────────────
# "10000 元能赚多少" 模拟

def _buy_10k_widget(pick: dict) -> str:
    """假如今天买一手 (100 股 = A 股最小交易单位)。"""
    lv = pick["levels"]
    n_shares = 100  # 一手 = 100 股
    actual_invest = n_shares * lv["buy"]
    target_value = n_shares * lv["target"]
    stop_value = n_shares * lv["stop"]
    target_pl = target_value - actual_invest
    stop_pl = stop_value - actual_invest
    return f"""
    <div class="buy10k">
      <div class="b10-title">💰 假如你今天买 1 手 (100 股)</div>
      <div class="b10-grid">
        <div class="b10-item">
          <div class="b10-label">入场成本</div>
          <div class="b10-num neutral">¥{actual_invest:,.0f}</div>
          <div class="b10-sub">100 股 @ ¥{lv['buy']}</div>
        </div>
        <div class="b10-item up">
          <div class="b10-label">🎯 目标位卖出</div>
          <div class="b10-num up">+¥{target_pl:,.0f}</div>
          <div class="b10-sub">{lv['expected_ret_pct']:+.1f}% · {lv['holding_days_min']}-{lv['holding_days_max']} 日</div>
        </div>
        <div class="b10-item down">
          <div class="b10-label">🛑 触发止损</div>
          <div class="b10-num down">{stop_pl:,.0f}</div>
          <div class="b10-sub">{-lv['risk_pct']:.1f}% · 即时离场</div>
        </div>
      </div>
    </div>
    """


# ────────────────────────────────────────────────────────────────────
# Confidence dial

def _confidence_dial(score: float, max_score: float = 4.0) -> str:
    pct = int(min(100, max(0, score / max_score * 100)))
    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=pct,
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "rgba(0,0,0,0)"},
            "bar": {"color": "#dc2626" if pct > 60 else "#f59e0b" if pct > 40 else "#9ca3af",
                    "thickness": 0.4},
            "bgcolor": "rgba(0,0,0,0.05)",
            "borderwidth": 0,
            "steps": [],
        },
    ))
    fig.update_layout(
        height=140, margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig.to_html(include_plotlyjs=False, full_html=False,
                       config={"displayModeBar": False}), pct


# ────────────────────────────────────────────────────────────────────
# Pick card

def _pick_card(pick: dict, idx: int) -> str:
    op = pick["opinion"]
    lv = pick["levels"]
    feat = pick["features"]
    badges = []
    if feat["above_ma60"]: badges.append('<span class="badge badge-green">站稳 60 日线</span>')
    if feat["breakout_20d"]: badges.append('<span class="badge badge-red">突破 20 日箱体</span>')
    if feat["vol_spike_5_30"] > 1.3: badges.append(f'<span class="badge badge-amber">放量 {feat["vol_spike_5_30"]:.1f}×</span>')
    if feat["ret_20d_pct"] > 5: badges.append(f'<span class="badge badge-red">月线 {feat["ret_20d_pct"]:+.1f}%</span>')
    if feat["above_ma20"] and not badges: badges.append('<span class="badge badge-green">短线偏强</span>')

    dial_html, dial_pct = _confidence_dial(pick["score"])
    reasons_html = "".join(f"<li>{r}</li>" for r in op["reasons"])
    snowball_text = f"{pick['name']} {pick['symbol']} | 买入 ¥{lv['buy']} 目标 ¥{lv['target']} ({lv['expected_ret_pct']:+.1f}%) 止损 ¥{lv['stop']} | {lv['holding_days_min']}-{lv['holding_days_max']}日"

    return f"""
    <div class="card">
      <div class="card-header">
        <div class="rank">No.{idx}</div>
        <div class="head-text">
          <div class="symbol-row">
            <span class="name">{pick['name']}</span>
            <span class="code">{pick['symbol']}</span>
          </div>
          <div class="opinion-line">{op['one_liner']}</div>
          <div class="badges">{''.join(badges)}</div>
        </div>
        <div class="big-num">
          <div class="num-target">{lv['expected_ret_pct']:+.1f}%</div>
          <div class="num-label">目标涨幅</div>
          <div class="dial-mini">{dial_html}<div class="dial-pct">置信 {dial_pct}%</div></div>
        </div>
      </div>

      {_buy_10k_widget(pick)}

      <div class="grid-2">
        <div class="chart-wrap">
          <h4>📈 K 线 + 三条线</h4>
          {_kline_chart(pick)}
        </div>
        <div class="chart-wrap">
          <h4>🎲 未来 10 天 100 条蒙特卡洛路径</h4>
          {_monte_carlo_cone(pick)}
        </div>
      </div>

      <div class="grid-2">
        <div class="reasoning-block">
          <h4>📜 为什么是这一只</h4>
          <ul>{reasons_html}</ul>
        </div>
        <div class="action-block">
          <h4>📋 行动指南</h4>
          <div class="ai-row"><span class="ai-l">仓位</span><span class="ai-v">{op['position_advice']}</span></div>
          <div class="ai-row"><span class="ai-l">持有</span><span class="ai-v">{lv['holding_days_min']}-{lv['holding_days_max']} 个交易日</span></div>
          <div class="ai-row"><span class="ai-l">风险收益比</span><span class="ai-v">{lv['rr_ratio']} : 1</span></div>
          <div class="ai-row warn"><span class="ai-l">⚠️ 止损</span><span class="ai-v">¥{lv['stop']} ({-lv['risk_pct']:.1f}%)</span></div>
          <button class="copy-btn" onclick="copyToClipboard('{snowball_text}')">📋 复制到雪球/微信</button>
        </div>
      </div>
    </div>
    """


# ────────────────────────────────────────────────────────────────────
# M&A radar bubble chart

def _ma_bubble(ma_top: list[dict]) -> str:
    if not ma_top:
        return ""
    fig = go.Figure()
    # Map industries to distinct colors
    industries = list({r.get("industry") or "其他" for r in ma_top})
    palette = ["#dc2626", "#f59e0b", "#10b981", "#3b82f6", "#a855f7", "#ec4899",
               "#14b8a6", "#f97316", "#6366f1", "#84cc16"]
    ind_color = {ind: palette[i % len(palette)] for i, ind in enumerate(industries)}

    for r in ma_top:
        ind = r.get("industry") or "其他"
        ret = (r.get("ret_6m") or 0) * 100
        pb = r.get("pb") or 2.0
        score = r.get("ma_score", 0.3)
        mc = r.get("market_cap_yi") or 100
        fig.add_trace(go.Scatter(
            x=[ret], y=[pb],
            mode="markers+text",
            marker=dict(size=max(15, min(60, score * 100)), color=ind_color[ind],
                        opacity=0.7, line=dict(color="white", width=2)),
            text=[f"{r['name']}<br><sub>{r['symbol']}</sub>"],
            textposition="middle center",
            textfont=dict(color="white", size=10),
            name=ind, showlegend=False,
            hovertemplate=f"<b>{r['name']}</b> ({r['symbol']})<br>"
                          f"行业: {ind}<br>市值: {mc} 亿<br>"
                          f"PB: {pb}<br>半年收益: {ret:+.1f}%<br>"
                          f"M&A分: {score:.3f}<extra></extra>",
        ))
    fig.add_vline(x=0, line_color="#d1d5db", line_width=1)
    fig.add_hline(y=2, line_color="#d1d5db", line_dash="dot",
                   annotation_text="PB=2 参考线", annotation_position="right")
    fig.update_layout(
        height=440, margin=dict(t=20, b=40, l=50, r=20),
        xaxis_title="半年收益 (%) — 越靠左越被低估",
        yaxis_title="PB 估值 — 越低越便宜",
        plot_bgcolor="#fafafa", paper_bgcolor="white",
        font=dict(family="-apple-system, 'PingFang SC'", size=12),
    )
    return fig.to_html(include_plotlyjs=False, full_html=False,
                       config={"displayModeBar": False})


def _ma_block(top_ma: list[dict]) -> str:
    if not top_ma:
        return ""
    bubble = _ma_bubble(top_ma[:12])
    rows = []
    for i, r in enumerate(top_ma[:10], 1):
        reasons = "、".join(r.get("reasons") or ["综合评分"])
        st_tag = ' <span class="st-tag">⚠️ ST</span>' if r.get("is_st") else ''
        warn = ""
        if r.get("warnings"):
            warn = '<div class="warn-line">' + "; ".join(r["warnings"]) + '</div>'
        rows.append(f"""
          <tr>
            <td class="rank-cell">{i}</td>
            <td><b>{r['name']}</b>{st_tag} <span class="code-tag">{r['symbol']}</span>{warn}</td>
            <td>{r.get('industry') or '-'}</td>
            <td class="num">{r.get('market_cap_yi') or '?'} 亿</td>
            <td class="num">{r.get('pb') or '?'}</td>
            <td class="num">{(r.get('ret_6m') or 0)*100:+.1f}%</td>
            <td class="reason-cell">{reasons}</td>
          </tr>
        """)
    return f"""
    <div class="ma-section">
      <h2>🎯 今日并购标的雷达</h2>
      <p class="sub">把所有候选画在 (半年收益 × PB) 坐标系上，气泡越大 = M&A 评分越高。
        被收购最有可能从被市场冷落（左下角）的标的中产生。</p>
      {bubble}
      <h3 style="margin-top:24px;font-size:16px;color:#374151;">📋 Top 6 名单</h3>
      <table class="ma-table">
        <thead>
          <tr><th>#</th><th>股票</th><th>行业</th><th>市值</th><th>PB</th><th>半年收益</th><th>关键理由</th></tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """


# ────────────────────────────────────────────────────────────────────

def main() -> int:
    if not IN_PICKS.exists():
        print(f"[page] missing {IN_PICKS}")
        return 1
    data = json.loads(IN_PICKS.read_text())
    picks = data["picks"]
    ma_top = json.loads(IN_MA.read_text()).get("top", []) if IN_MA.exists() else []
    temp, mood, color, breadth = _market_temp()

    cards_html = "".join(_pick_card(p, i + 1) for i, p in enumerate(picks))
    temp_html = _temp_gauge(temp, color)
    ma_html = _ma_block(ma_top)
    shadow_html = SHADOW_PANEL.read_text(encoding="utf-8") if SHADOW_PANEL.exists() else ""

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>今日 A 股 · 3 大金股 · {data['as_of'][:10]}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", Helvetica, Arial, sans-serif;
         background: #f3f4f6; margin: 0; color: #111827; }}
  .hero {{ background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
           color: white; padding: 32px 16px; text-align: center; position: relative; overflow: hidden; }}
  .hero::before {{ content: ''; position: absolute; inset: 0;
                    background: radial-gradient(circle at 30% 50%, {color}33 0%, transparent 60%);
                    pointer-events: none; }}
  .hero-title {{ font-size: 44px; font-weight: 900; margin: 0; position: relative; }}
  .hero-sub {{ color: #9ca3af; font-size: 14px; margin-top: 6px; position: relative; }}
  .date-chip {{ display: inline-block; padding: 4px 12px; background: rgba(255,255,255,0.1);
                color: white; border-radius: 999px; font-size: 12px; font-weight: 600;
                margin-left: 8px; }}
  .mood-tag {{ display: inline-block; padding: 4px 12px; background: {color}44;
               color: white; border-radius: 999px; font-size: 13px;
               margin-top: 8px; border: 1px solid {color}66; }}
  .breadth-row {{ display: flex; justify-content: center; gap: 24px; margin-top: 18px;
                   position: relative; flex-wrap: wrap; }}
  .breadth-cell {{ text-align: center; min-width: 80px; }}
  .bc-num {{ font-size: 24px; font-weight: 800; color: white; line-height: 1; }}
  .bc-num.up {{ color: #fca5a5; }}
  .bc-num.down {{ color: #86efac; }}
  .bc-label {{ font-size: 11px; color: #9ca3af; margin-top: 4px; }}
  .gauge-wrap {{ max-width: 360px; margin: 12px auto 0; position: relative; }}
  .container {{ max-width: 1180px; margin: 0 auto; padding: 24px 16px 60px; }}
  .card {{ background: white; border-radius: 16px; padding: 24px;
           margin-bottom: 28px; box-shadow: 0 4px 24px rgba(0,0,0,0.06);
           border: 1px solid #f3f4f6; }}
  .card-header {{ display: grid; grid-template-columns: 60px 1fr auto; gap: 16px;
                  align-items: center; padding-bottom: 16px;
                  border-bottom: 2px dashed #e5e7eb; margin-bottom: 16px; }}
  .rank {{ font-size: 36px; font-weight: 900;
           background: linear-gradient(135deg, #fbbf24, #f59e0b);
           -webkit-background-clip: text; -webkit-text-fill-color: transparent;
           text-align: center; }}
  .symbol-row {{ display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }}
  .name {{ font-size: 28px; font-weight: 800; color: #111827; }}
  .code {{ font-size: 14px; color: #9ca3af; font-family: monospace; }}
  .opinion-line {{ font-size: 18px; color: #dc2626; font-weight: 700; margin: 6px 0 8px; }}
  .big-num {{ text-align: right; min-width: 180px; }}
  .num-target {{ font-size: 44px; font-weight: 900; color: #dc2626; line-height: 1; }}
  .num-label {{ font-size: 12px; color: #6b7280; margin-top: 4px; }}
  .dial-mini {{ margin-top: 4px; }}
  .dial-pct {{ font-size: 11px; color: #6b7280; margin-top: -36px; text-align: center; }}
  .badges {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
  .badge {{ padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }}
  .badge-red {{ background: #fee2e2; color: #b91c1c; }}
  .badge-green {{ background: #dcfce7; color: #166534; }}
  .badge-amber {{ background: #fef3c7; color: #92400e; }}

  .buy10k {{ background: linear-gradient(135deg, #fef3c7 0%, #fef9c3 100%);
             border-radius: 12px; padding: 18px; margin-bottom: 18px;
             border: 1px solid #fcd34d; }}
  .b10-title {{ font-size: 14px; font-weight: 700; color: #92400e; margin-bottom: 12px; }}
  .b10-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }}
  .b10-item {{ padding: 14px; background: white; border-radius: 8px; text-align: center; }}
  .b10-item.up {{ background: linear-gradient(135deg, #fee2e2, white); }}
  .b10-item.down {{ background: linear-gradient(135deg, #d1fae5, white); }}
  .b10-label {{ font-size: 12px; color: #6b7280; margin-bottom: 6px; }}
  .b10-num {{ font-size: 24px; font-weight: 800; line-height: 1.2; }}
  .b10-num.up {{ color: #dc2626; }}
  .b10-num.down {{ color: #16a34a; }}
  .b10-num.neutral {{ color: #1f2937; font-size: 18px; }}
  .b10-sub {{ font-size: 11px; color: #6b7280; margin-top: 4px; }}

  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
  .chart-wrap, .reasoning-block, .action-block {{ background: #fafafa;
       border-radius: 10px; padding: 14px; border: 1px solid #f3f4f6; }}
  h4 {{ font-size: 13px; margin: 0 0 10px; color: #374151; font-weight: 700; }}
  .reasoning-block ul {{ margin: 0; padding-left: 20px; }}
  .reasoning-block li {{ margin-bottom: 6px; font-size: 13px; line-height: 1.6; }}
  .ai-row {{ display: flex; justify-content: space-between; padding: 6px 0;
              border-bottom: 1px solid #f3f4f6; font-size: 13px; }}
  .ai-row.warn {{ color: #b91c1c; font-weight: 700; }}
  .ai-l {{ color: #6b7280; }}
  .ai-v {{ font-weight: 600; }}
  .copy-btn {{ margin-top: 12px; width: 100%; padding: 10px;
                background: #1f2937; color: white; border: none;
                border-radius: 8px; cursor: pointer; font-size: 13px;
                font-weight: 600; transition: 0.2s; }}
  .copy-btn:hover {{ background: #111827; }}

  .ma-section {{ background: white; border-radius: 16px; padding: 28px;
                 margin-top: 32px; box-shadow: 0 4px 24px rgba(0,0,0,0.06); }}
  .ma-section h2 {{ font-size: 24px; margin: 0 0 8px; color: #b91c1c; }}
  .ma-section .sub {{ font-size: 13px; color: #6b7280; margin: 0 0 18px; line-height: 1.6; }}
  .ma-table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }}
  .ma-table th {{ text-align: left; padding: 10px; background: #fef3c7; color: #92400e;
                  border-bottom: 2px solid #fcd34d; font-weight: 700; font-size: 12px; }}
  .ma-table td {{ padding: 11px 10px; border-bottom: 1px solid #f3f4f6; vertical-align: middle; }}
  .ma-table .num {{ text-align: right; font-family: monospace; }}
  .ma-table .rank-cell {{ font-weight: 800; color: #dc2626; text-align: center;
                          width: 32px; font-size: 16px; }}
  .ma-table .code-tag {{ display: inline-block; padding: 1px 6px;
                         background: #f3f4f6; color: #6b7280;
                         border-radius: 4px; font-size: 11px; margin-left: 4px;
                         font-family: monospace; }}
  .ma-table .reason-cell {{ color: #4b5563; font-size: 12px; }}
  .st-tag {{ display: inline-block; padding: 1px 6px; background: #fee2e2; color: #b91c1c;
             border-radius: 4px; font-size: 11px; font-weight: 700; margin-left: 4px; }}
  .warn-line {{ font-size: 11px; color: #b91c1c; margin-top: 4px; }}

  /* Shadow panel */
  .shadow-panel {{ background: white; border-radius: 16px; padding: 28px;
                   margin-top: 32px; box-shadow: 0 4px 24px rgba(0,0,0,0.06); }}
  .shadow-panel h2 {{ font-size: 24px; margin: 0 0 8px; color: #111827; }}
  .shadow-panel .sub {{ font-size: 13px; color: #6b7280; margin: 0 0 14px; line-height: 1.6; }}
  .shadow-panel.empty {{ padding: 24px; color: #6b7280; font-size: 13px; }}
  .verdict-good {{ background: #dcfce7; border-left: 4px solid #16a34a;
                    padding: 14px 18px; margin: 14px 0; font-size: 14px;
                    color: #166534; border-radius: 4px; font-weight: 600; }}
  .verdict-marginal {{ background: #fef3c7; border-left: 4px solid #f59e0b;
                        padding: 14px 18px; margin: 14px 0; font-size: 14px;
                        color: #92400e; border-radius: 4px; font-weight: 600; }}
  .verdict-bad {{ background: #fee2e2; border-left: 4px solid #dc2626;
                   padding: 14px 18px; margin: 14px 0; font-size: 14px;
                   color: #991b1b; border-radius: 4px; font-weight: 600; }}
  .shadow-stats {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px;
                    margin: 18px 0; }}
  .stat-cell {{ text-align: center; padding: 12px 8px; background: #fafafa;
                 border-radius: 8px; border: 1px solid #f3f4f6; }}
  .sn {{ font-size: 22px; font-weight: 800; line-height: 1; }}
  .sn.up {{ color: #dc2626; }}
  .sn.down {{ color: #16a34a; }}
  .sl {{ font-size: 11px; color: #6b7280; margin-top: 4px; }}
  .shadow-table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }}
  .shadow-table th {{ text-align: left; padding: 8px; background: #f3f4f6; color: #4b5563;
                       border-bottom: 2px solid #e5e7eb; font-weight: 700; font-size: 12px; }}
  .shadow-table td {{ padding: 10px 8px; border-bottom: 1px solid #f3f4f6; }}
  .shadow-table .num {{ text-align: right; font-family: monospace; font-weight: 600; }}
  .shadow-table .num.up {{ color: #dc2626; }}
  .shadow-table .num.down {{ color: #16a34a; }}
  .shadow-table .code-tag {{ display: inline-block; padding: 1px 6px;
                              background: #f3f4f6; color: #6b7280;
                              border-radius: 4px; font-size: 11px; margin-left: 4px;
                              font-family: monospace; }}

  footer {{ text-align: center; color: #9ca3af; font-size: 12px;
            margin-top: 40px; padding: 20px; border-top: 1px solid #e5e7eb; }}
  .disclaimer {{ background: #fef3c7; border-left: 4px solid #f59e0b;
                 padding: 14px 18px; margin: 24px 0; font-size: 12px;
                 color: #92400e; border-radius: 4px; line-height: 1.6; }}
  @media (max-width: 800px) {{
    .grid-2 {{ grid-template-columns: 1fr; }}
    .b10-grid {{ grid-template-columns: 1fr; }}
    .card-header {{ grid-template-columns: 48px 1fr; }}
    .big-num {{ grid-column: 1 / -1; text-align: center; }}
  }}
</style>
</head>
<body>
<div class="hero">
  <h1 class="hero-title">📈 今日 A 股 · 3 大金股</h1>
  <div class="hero-sub">
    {('🤖 <b>STAR 自进化 picker</b> · 持平 +5pp 5日超额回测 · ' if data.get('method', '').startswith('STAR') else '短线买点 · ')}量化打分 · 直接观点
    <span class="date-chip">{data['as_of'][:10]}</span>
  </div>{('<div class="hero-sub" style="font-size:12px;opacity:0.7;margin-top:6px">picker: ' + ' · '.join(f"{p['name']} (+{p['held_out_5d_top3_excess_pp']}pp hit {int(p['hit_rate_above_5pp']*100)}%)" for p in data.get('method_provenance', {}).get('pickers', [])) + '</div>' if data.get('method', '').startswith('STAR') else '')}
  <div class="gauge-wrap">{temp_html}</div>
  <div class="mood-tag">市场情绪: {mood}</div>
  <div class="breadth-row">
    <div class="breadth-cell"><div class="bc-num">{breadth.get('universe_size', 0):,}</div><div class="bc-label">A 股标的</div></div>
    <div class="breadth-cell"><div class="bc-num up">{breadth.get('pct_up_today', 0):.0f}%</div><div class="bc-label">今日红盘率</div></div>
    <div class="breadth-cell"><div class="bc-num up">{breadth.get('n_limit_up', 0)}</div><div class="bc-label">🔴 涨停</div></div>
    <div class="breadth-cell"><div class="bc-num down">{breadth.get('n_limit_down', 0)}</div><div class="bc-label">🟢 跌停</div></div>
    <div class="breadth-cell"><div class="bc-num">{breadth.get('median_5d_ret_pct', 0):+.1f}%</div><div class="bc-label">5日涨幅中位</div></div>
  </div>
</div>

<div class="container">
  {cards_html}
  {shadow_html}
  {ma_html}

  <div class="disclaimer">
    ⚠️ <b>风险提示</b>：本页面为基于历史数据的量化打分输出，<b>不构成投资建议</b>。
    A 股市场存在涨跌幅限制、流动性风险、政策风险。每次入场前请独立判断，严格执行止损。
    历史不代表未来。当前评分基于 {data['n_candidates_scored']} 只样本。
  </div>

  <footer>
    Generated by AlphaGym daily-picks v2 · STAR framework · {datetime.now().strftime('%Y-%m-%d %H:%M')}
  </footer>
</div>

<script>
function copyToClipboard(text) {{
  navigator.clipboard.writeText(text).then(() => {{
    const btns = document.querySelectorAll('.copy-btn');
    btns.forEach(b => {{
      if (b.dataset.original === undefined) b.dataset.original = b.innerText;
    }});
    event.target.innerText = '✅ 已复制';
    setTimeout(() => {{ event.target.innerText = event.target.dataset.original; }}, 1500);
  }});
}}
</script>
</body>
</html>"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"[page] wrote {OUT} ({len(html)/1024:.0f} KB)")
    print(f"[page] open: file://{OUT.resolve()}")
    print(f"[page] market temp: {temp}° ({mood})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
