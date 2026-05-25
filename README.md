# A-share Daily Picks (Auto-Updated)

每天北京时间 9:00 自动更新。基于全 A 股 5000+ 只量化打分：

- **每日 3 大金股**：技术面 + 量能 + 资金 + 风控复合打分
- **并购标的雷达**：5 因子（小市值 / 低估值 / 涨幅落后 / 公告动向 / 行业整合）
- **市场温度计**：动态显示当日市场情绪 + 涨停 / 跌停统计

## 网页

👉 **https://alphabeta7788.github.io/ashare-daily-picks/**

## 自动化

GitHub Actions 每天 UTC 01:00 (北京 09:00) 跑：
1. baostock 拉全 A 股 universe + 最近 300 天 K 线
2. 计算 3 picks + Top 10 M&A 候选
3. 渲染 plotly HTML
4. push → GitHub Pages 自动重建

也可在 [Actions tab](https://github.com/alphabeta7788/ashare-daily-picks/actions) 手动点 "Run workflow" 立刻更新。

## ⚠️ 风险提示

本网页仅为基于历史数据的量化打分输出，**不构成投资建议**。
A 股市场存在涨跌幅限制、流动性风险、政策风险。
每次入场前请独立判断，严格执行止损。
