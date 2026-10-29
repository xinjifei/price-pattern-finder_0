# -*- coding: utf-8 -*-
# plot_data.py —— 把 data/ 下每份行情 CSV 画成一张 plotly 折线图，输出到 check/
#
# 用法：uv run python plot_data.py
# 产物：check/<股票代码>.html（彼此共用一份 check/plotly.min.js，离线可打开）

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

##data_dir，数据目录，以脚本位置为基准，不受当前工作目录影响
data_dir = Path(__file__).resolve().parent / "data"
##out_dir，输出目录
out_dir = Path(__file__).resolve().parent / "check"
out_dir.mkdir(exist_ok=True)

##csv_path，单份行情文件
for csv_path in sorted(data_dir.glob("*.csv")):
    ##df，日期 + 收盘价
    df = pd.read_csv(csv_path, parse_dates=["date"])
    ##symbol，股票代码，取文件名去掉 .csv
    symbol = csv_path.stem

    ##fig，单只股票的折线图
    ##date_text，悬停里显示的日期字符串
    date_text = df["date"].dt.strftime("%Y-%m-%d")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        # 横轴用 0,1,2... 的编号（和 matplotlib 图里的 x 对齐），日期放进悬停提示
        x=df.index,
        y=df["close"],
        customdata=date_text,
        mode="lines",
        name=symbol,
        line={"width": 1.5},
        hovertemplate="编号 %{x}<br>%{customdata}<br>close = %{y}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{symbol} 收盘价（{len(df)} 个交易日）",
        xaxis={
            "title": "编号",
            "rangeslider": {"visible": True},   # 底部拖动条，方便看局部
            "showgrid": True,
        },
        yaxis={"title": "close"},
        hovermode="x unified",
        template="plotly_white",
        margin={"l": 60, "r": 30, "t": 60, "b": 60},
    )

    ##path，单个 HTML 产物
    path = out_dir / f"{symbol}.html"
    # "directory"：plotly.min.js 只在 check/ 里存一份，三个 HTML 共用，文件小且断网可看
    fig.write_html(path, include_plotlyjs="directory")
    print(f"{csv_path.name} -> {path.relative_to(out_dir.parent)}")
