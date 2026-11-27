"""计算形态结束后 1-20 个交易日的累计收益路径。

输入 : analysis/data/pattern_hits.csv（由 main.py 生成）
输出 : analysis/forward_returns/output/forward_return_paths.csv
       analysis/forward_returns/output/forward_return_summary.csv
       analysis/forward_returns/output/forward_return_paths.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import PercentFormatter

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
ANALYSIS_DIR = ROOT / "analysis"
DATA_DIR = ROOT / "data"
HITS_FILE = ANALYSIS_DIR / "data" / "pattern_hits.csv"
OUTPUT_DIR = SCRIPT_DIR / "output"
PATHS_FILE = OUTPUT_DIR / "forward_return_paths.csv"
SUMMARY_FILE = OUTPUT_DIR / "forward_return_summary.csv"
PLOT_FILE = OUTPUT_DIR / "forward_return_paths.png"

HORIZONS = list(range(1, 21))
RETURN_COLUMNS = [f"ret_{h}" for h in HORIZONS]


def load_close(stock: str) -> tuple[pd.Series, np.ndarray]:
    """读取单只股票的日期和收盘价。"""
    frame = pd.read_csv(DATA_DIR / f"{stock}.csv", parse_dates=["date"])
    return frame["date"], frame["close"].to_numpy()


def build_paths(hits: pd.DataFrame) -> pd.DataFrame:
    """逐条命中计算形态结束日之后的累计收益。"""
    price_cache = {}
    rows = []

    for _, hit in hits.iterrows():
        stock = hit["stock"]
        if stock not in price_cache:
            price_cache[stock] = load_close(stock)
        dates, closes = price_cache[stock]

        base_index = int(hit["end_index"])
        base_close = float(closes[base_index])
        if not np.isclose(base_close, hit["end_close"]):
            raise ValueError(f"{stock} 第 {base_index} 条收盘价与命中文件不一致")

        row = {
            "stock": stock,
            "end_index": base_index,
            "end_date": hit["end_date"],
            "score": hit["score"],
            "start": hit["start"],
            "w_size": hit["w_size"],
            "t1": hit["t1"],
            "t2": hit["t2"],
            "end_close": base_close,
        }
        for horizon in HORIZONS:
            index = base_index + horizon
            row[f"ret_{horizon}"] = (
                float(closes[index] / base_close - 1) if index < len(closes) else np.nan
            )
        rows.append(row)

    return pd.DataFrame(rows)


def summarize(paths: pd.DataFrame) -> pd.DataFrame:
    """按持有期汇总横截面收益。"""
    long = paths.melt(
        id_vars=["stock", "end_index", "end_date", "score"],
        value_vars=RETURN_COLUMNS,
        var_name="horizon",
        value_name="cum_return",
    )
    long["horizon"] = long["horizon"].str.removeprefix("ret_").astype(int)

    summary = (
        long.groupby("horizon")
        .agg(
            count=("cum_return", "count"),
            mean=("cum_return", "mean"),
            median=("cum_return", "median"),
            std=("cum_return", "std"),
            min=("cum_return", "min"),
            max=("cum_return", "max"),
            positive_rate=(
                "cum_return",
                lambda values: (
                    float((values.dropna() > 0).mean())
                    if values.notna().any()
                    else np.nan
                ),
            ),
        )
        .reset_index()
    )
    return summary


def plot_paths(paths: pd.DataFrame) -> None:
    """绘制每次命中的收益路径及均值、中位数。"""
    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "PingFang SC",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 6))
    values = paths[RETURN_COLUMNS].to_numpy(dtype=float)
    scores = paths["score"].to_numpy(dtype=float)

    low, high = float(scores.min()), float(scores.max())
    if np.isclose(low, high):
        norm = Normalize(vmin=low - 1.0, vmax=high + 1.0)
    else:
        # score 长尾明显，用对数归一化才能拉开低分区间的颜色
        norm = LogNorm(vmin=max(low, np.finfo(float).tiny), vmax=high)
    cmap = plt.get_cmap("RdYlGn")

    # 低分先画、高分后画，避免红花盖住绿花
    for index in np.argsort(scores):
        ax.plot(
            HORIZONS,
            values[index],
            color=cmap(norm(scores[index])),
            alpha=0.75,
            linewidth=0.9,
        )

    ax.plot(
        HORIZONS,
        np.nanmean(values, axis=0),
        color="black",
        linewidth=2,
        label="均值",
    )
    ax.plot(
        HORIZONS,
        np.nanmedian(values, axis=0),
        color="#1f77b4",
        linewidth=2,
        linestyle="--",
        label="中位数",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(HORIZONS)
    ax.set_xlabel("形态结束后交易日数")
    ax.set_ylabel("累计收益（相对结束日收盘价）")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(
        f"形态结束后累计收益路径"
        f"（{paths['stock'].nunique()} 只股票，{len(paths)} 次命中）"
    )

    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])
    colorbar = fig.colorbar(scalar_map, ax=ax, pad=0.02)
    colorbar.set_label("score（越小越红，越大越绿）")

    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not HITS_FILE.exists():
        raise FileNotFoundError(f"请先运行 main.py 生成 {HITS_FILE}")

    hits = pd.read_csv(HITS_FILE)
    if hits.empty:
        raise ValueError("命中文件为空")

    paths = build_paths(hits)
    summary = summarize(paths)

    paths[["stock", "end_index", "end_date", "score", "start", "w_size",
           "t1", "t2", "end_close", *RETURN_COLUMNS]].to_csv(
        PATHS_FILE, index=False, encoding="utf-8-sig"
    )
    summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")
    plot_paths(paths)

    print(f"命中数: {len(paths)}，股票数: {paths['stock'].nunique()}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n收益路径: {PATHS_FILE}")
    print(f"汇总: {SUMMARY_FILE}")
    print(f"图: {PLOT_FILE}")


if __name__ == "__main__":
    main()
