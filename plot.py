"""把每只股票的全部识别结果保存为图片。"""

from pathlib import Path
import contextlib
import importlib.util
import io

import numpy as np

from finder import csv_numpy, fit_line, found


def draw_results(
    data_dir: Path,
    out_root: Path,
    stock_files: tuple,
    window_size: list,
    p: dict,
    results_by_stock: dict | None = None,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    ##exception，导入绘图库失败时的异常信息
    except ImportError as exc:
        print(f"跳过绘图：{exc}")
        return

    # 本文件也叫 plot.py，按路径加载共享绘图函数，避免同名模块冲突
    plot_path = Path.home() / "stu" / "math_practice_file" / "function" / "plot.py"
    spec = importlib.util.spec_from_file_location("shared_plot", plot_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plot = module.plot

    out_root.mkdir(exist_ok=True)

    for filename in stock_files:
        csv_path = data_dir / filename
        stock_data = csv_numpy(str(csv_path), "close")

        if results_by_stock is not None:
            stock_results = results_by_stock[csv_path.stem]
        else:
            # found 会逐窗口打印，批量跑时屏蔽掉
            with contextlib.redirect_stdout(io.StringIO()):
                stock_results = found(stock_data, window_size=window_size, p=p)

        ##ranked，全部合法结果按分数升序（越小越像）
        ranked = sorted(stock_results, key=lambda r: r["score"])
        out_dir = out_root / csv_path.stem
        out_dir.mkdir(exist_ok=True)
        print(f"{csv_path.stem}: {len(ranked)} 个合法窗口 -> {out_dir}")

        ##r，该名次对应的匹配结果
        for rank, r in enumerate(ranked, start=1):
            start, w_size = r["start"], r["w_size"]
            w = stock_data[start:start + w_size]
            x = np.arange(start, start + w_size)

            # 实际收盘价
            layers = [
                {"type": "line", "data": np.column_stack([x, w]),
                 "color": "black", "linestyle": "-", "linewidth": 1},
            ]

            # 三段拟合线
            segs = [
                (w[0:r["t1"] + 1], 0, "tab:blue"),
                (w[r["t1"]:r["t2"] + 1], r["t1"], "tab:orange"),
                (w[r["t2"]:], r["t2"], "tab:green"),
            ]
            for ys, offset_, color in segs:
                xs = np.arange(len(ys))
                k, b = fit_line(ys)
                layers.append({
                    "type": "line",
                    "data": np.column_stack([start + offset_ + xs, k * xs + b]),
                    "color": color,
                    "linestyle": "--",
                    "linewidth": 1.5,
                })

            ##figure，图形对象
            ##axes，坐标轴对象
            fig, ax = plt.subplots(figsize=(8, 6))
            title = (f"{csv_path.stem} 第{rank}名 score={r['score']:.3f} | "
                     f"W={w_size} start={start} t1={r['t1']} t2={r['t2']}")
            plot(layers, title=title, ax=ax, show=False)

            # 两个断点
            ##breakpoint，断点坐标
            for bp in (start + r["t1"], start + r["t2"]):
                ax.axvline(bp, color="gray", linestyle=":", linewidth=1)

            fig.tight_layout()
            path = out_dir / f"rank_{rank:02d}_score_{r['score']:.3f}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  {path.name}")
