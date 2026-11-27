"""随机 20 日窗口对照组。

随机窗口数量与形态命中数相同；以窗口结束日收盘价为基准，
计算随后 1-20 个交易日的累计收益，并与形态命中结果对比。

输入 : analysis/data/pattern_hits.csv
       analysis/forward_returns/output/forward_return_summary.csv
输出 : analysis/random_control/output/random_control_paths.csv
       analysis/random_control/output/random_control_summary.csv
       analysis/random_control/output/random_control_paths.png
       analysis/random_control/output/comparison_summary.csv
       analysis/random_control/output/hit_vs_control.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
ANALYSIS_DIR = ROOT / "analysis"
DATA_DIR = ROOT / "data"
HITS_FILE = ANALYSIS_DIR / "data" / "pattern_hits.csv"
HIT_SUMMARY_FILE = (
    ANALYSIS_DIR / "forward_returns" / "output" / "forward_return_summary.csv"
)
OUTPUT_DIR = SCRIPT_DIR / "output"
CONTROL_PATHS_FILE = OUTPUT_DIR / "random_control_paths.csv"
CONTROL_SUMMARY_FILE = OUTPUT_DIR / "random_control_summary.csv"
CONTROL_PLOT_FILE = OUTPUT_DIR / "random_control_paths.png"
COMPARISON_FILE = OUTPUT_DIR / "comparison_summary.csv"
COMPARISON_PLOT_FILE = OUTPUT_DIR / "hit_vs_control.png"

WINDOW_SIZE = 20
HORIZONS = list(range(1, 21))
RETURN_COLUMNS = [f"ret_{h}" for h in HORIZONS]
RANDOM_SEED = 20261127


def load_prices(stock: str) -> tuple[pd.Series, np.ndarray]:
    """读取单只股票的日期和收盘价。"""
    frame = pd.read_csv(DATA_DIR / f"{stock}.csv", parse_dates=["date"])
    return frame["date"], frame["close"].to_numpy()


def build_pool(price_cache: dict) -> list[tuple[str, int]]:
    """枚举所有至少还有 1 个后续交易日的 20 日窗口。"""
    pool = []
    for stock, (_, closes) in price_cache.items():
        for end_index in range(WINDOW_SIZE - 1, len(closes) - 1):
            pool.append((stock, end_index))
    return pool


def sample_controls(price_cache: dict, count: int) -> pd.DataFrame:
    """从全市场窗口池中无放回随机抽样，并计算后续收益。"""
    pool = build_pool(price_cache)
    if count > len(pool):
        raise ValueError(f"样本数 {count} 超过候选窗口数 {len(pool)}")

    rng = np.random.default_rng(RANDOM_SEED)
    picked = rng.choice(len(pool), size=count, replace=False)
    picked = sorted(picked, key=lambda i: (pool[i][0], pool[i][1]))

    rows = []
    for pool_index in picked:
        stock, end_index = pool[pool_index]
        dates, closes = price_cache[stock]
        base_close = float(closes[end_index])
        start_index = end_index - WINDOW_SIZE + 1

        row = {
            "stock": stock,
            "start_index": start_index,
            "start_date": dates.iloc[start_index].strftime("%Y-%m-%d"),
            "end_index": end_index,
            "end_date": dates.iloc[end_index].strftime("%Y-%m-%d"),
            "end_close": base_close,
        }
        for horizon in HORIZONS:
            index = end_index + horizon
            row[f"ret_{horizon}"] = (
                float(closes[index] / base_close - 1) if index < len(closes) else np.nan
            )
        rows.append(row)

    return pd.DataFrame(rows)


def summarize(paths: pd.DataFrame) -> pd.DataFrame:
    """按持有期汇总横截面收益。"""
    long = paths.melt(
        id_vars=["stock", "end_index", "end_date"],
        value_vars=RETURN_COLUMNS,
        var_name="horizon",
        value_name="cum_return",
    )
    long["horizon"] = long["horizon"].str.removeprefix("ret_").astype(int)

    return (
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


def set_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "PingFang SC",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def plot_control_paths(paths: pd.DataFrame) -> None:
    """绘制随机窗口的收益路径及均值、中位数。"""
    set_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 6))
    values = paths[RETURN_COLUMNS].to_numpy(dtype=float)

    for path in values:
        ax.plot(HORIZONS, path, color="#4c78a8", alpha=0.35, linewidth=0.9)

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
    ax.set_xlabel("窗口结束后交易日数")
    ax.set_ylabel("累计收益（相对结束日收盘价）")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(
        f"随机 {WINDOW_SIZE} 日窗口累计收益路径"
        f"（{paths['stock'].nunique()} 只股票，{len(paths)} 次抽样）"
    )

    fig.tight_layout()
    fig.savefig(CONTROL_PLOT_FILE, dpi=150)
    plt.close(fig)


def compare(hit_summary: pd.DataFrame, control_summary: pd.DataFrame) -> pd.DataFrame:
    """合并命中与随机对照的逐期统计。"""
    comparison = hit_summary[["horizon", "count", "mean", "median"]].merge(
        control_summary[["horizon", "count", "mean", "median"]],
        on="horizon",
        suffixes=("_hit", "_control"),
    )
    comparison["mean_diff"] = comparison["mean_hit"] - comparison["mean_control"]
    comparison["median_diff"] = comparison["median_hit"] - comparison["median_control"]
    return comparison


def plot_comparison(comparison: pd.DataFrame) -> None:
    """绘制命中与随机对照的均值、中位数对比。"""
    set_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        comparison["horizon"],
        comparison["mean_hit"],
        color="#d62728",
        linewidth=2,
        label="命中 均值",
    )
    ax.plot(
        comparison["horizon"],
        comparison["mean_control"],
        color="#1f77b4",
        linewidth=2,
        label="随机 均值",
    )
    ax.plot(
        comparison["horizon"],
        comparison["median_hit"],
        color="#d62728",
        linewidth=2,
        linestyle="--",
        label="命中 中位数",
    )
    ax.plot(
        comparison["horizon"],
        comparison["median_control"],
        color="#1f77b4",
        linewidth=2,
        linestyle="--",
        label="随机 中位数",
    )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(HORIZONS)
    ax.set_xlabel("结束后交易日数")
    ax.set_ylabel("累计收益（相对结束日收盘价）")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("形态命中与随机 20 日窗口的累计收益对比")

    fig.tight_layout()
    fig.savefig(COMPARISON_PLOT_FILE, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not HITS_FILE.exists():
        raise FileNotFoundError(f"请先运行 main.py 生成 {HITS_FILE}")
    if not HIT_SUMMARY_FILE.exists():
        raise FileNotFoundError(f"请先运行 forward_returns.py 生成 {HIT_SUMMARY_FILE}")

    hits = pd.read_csv(HITS_FILE)
    hit_count = len(hits)

    stocks = sorted(hits["stock"].unique())
    price_cache = {stock: load_prices(stock) for stock in stocks}

    paths = sample_controls(price_cache, hit_count)
    summary = summarize(paths)

    paths[
        ["stock", "start_index", "start_date", "end_index", "end_date",
         "end_close", *RETURN_COLUMNS]
    ].to_csv(CONTROL_PATHS_FILE, index=False, encoding="utf-8-sig")
    summary.to_csv(CONTROL_SUMMARY_FILE, index=False, encoding="utf-8-sig")

    hit_summary = pd.read_csv(HIT_SUMMARY_FILE)
    comparison = compare(hit_summary, summary)
    comparison.to_csv(COMPARISON_FILE, index=False, encoding="utf-8-sig")

    plot_control_paths(paths)
    plot_comparison(comparison)

    print(f"随机窗口数: {len(paths)}，股票数: {paths['stock'].nunique()}")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n对照路径: {CONTROL_PATHS_FILE}")
    print(f"对照汇总: {CONTROL_SUMMARY_FILE}")
    print(f"对照图: {CONTROL_PLOT_FILE}")
    print(f"对比汇总: {COMPARISON_FILE}")
    print(f"对比图: {COMPARISON_PLOT_FILE}")


if __name__ == "__main__":
    main()
