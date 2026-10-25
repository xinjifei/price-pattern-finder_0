import numpy as np
import pandas as pd

source_path = "/Users/lyuchanghe/stu/found_0/data/sz000001.csv"


def csv_numpy(csv_path: str, col_name: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    arr = df[col_name].to_numpy()
    return arr


close_data = csv_numpy(source_path, "close")
test_data = close_data[1:20]


def fit_line(ys: np.ndarray):
    xs = np.arange(len(ys))
    k, b = np.polyfit(xs, ys, deg=1)
    return k, b


def score(area_list: list, p: dict):
    """三段特征打分。返回 (total, feat)；A1/A3 不满足硬门槛时返回 None。"""
    a_1, a_2, a_3 = area_list

    s1, _ = fit_line(a_1)
    s2, _ = fit_line(a_2)
    s3, _ = fit_line(a_3)

    # 段内净变化 = 斜率 × 首尾间隔
    A1 = s1 * (len(a_1) - 1)
    A2 = s2 * (len(a_2) - 1)
    A3 = s3 * (len(a_3) - 1)

    # 硬门槛：第一段、第三段必须涨
    if A1 <= 0 or A3 <= 0:
        return None

    feat = [
        abs(A2) / A1,   # 平盘度
        A3 / A1,        # 幅度比
        s3 / s1,        # 斜率比
    ]

    # 偏差归一化，加权平方和（越小越像）
    total = 0.0
    for f, t, e, w in zip(feat, p["target"], p["tol"], p["weight"]):
        total += w * ((f - t) / e) ** 2

    return float(total), [float(x) for x in feat]


def tau_layout(w_size: int, p: dict):
    """返回 (tau_area, t1_point, t2_point, t1_start, t2_start)。"""
    tau_area = round(w_size / 6)
    ratio_sum = sum(p["area_ratio"])
    t1_point = round(w_size * p["area_ratio"][0] / ratio_sum)
    t2_point = round(w_size * (p["area_ratio"][0] + p["area_ratio"][1]) / ratio_sum)
    half = tau_area // 2

    return tau_area, t1_point, t2_point, t1_point - half, t2_point - half


def optimal_tau(w: np.ndarray, p: dict):
    """单窗口内搜两个断点，返回最佳 {'score','t1','t2','feat'}；无合法分段返回 None。"""
    w_size = len(w)
    tau_area, t1_point, _, t1_start, t2_start = tau_layout(w_size, p)

    best = None
    for i in range(tau_area):
        t1 = t1_start + i
        for j in range(tau_area):
            t2 = t2_start + j

            # 边界：t1 < t2，且每段至少 2 个点
            if not (2 <= t1 < t2 and t2 - t1_point >= 2 and w_size - t2 >= 2):
                continue

            res = score([w[0:t1], w[t1_point:t2], w[t2:]], p)
            if res is None:
                continue

            total, feat = res
            if best is None or total < best["score"]:
                best = {"score": total, "t1": t1, "t2": t2, "feat": feat}

    return best


def found(data: np.ndarray, window_size: list, p: dict):
    """逐窗口滑动，每个窗口留一个最佳分段。"""
    day = len(data)
    results = []

    for w_size in range(window_size[0], window_size[1] + 1):
        _, t1_point, t2_point, _, _ = tau_layout(w_size, p)

        # 画图
        print(f"\nwindow size = {w_size} ##########")
        ruler = ["-"] * 100
        for point in (t1_point, t2_point):
            ruler[min(99, round(point / w_size * 100))] = "|"
        print("".join(ruler))

        # 遍历
        for w_nub in range(day - w_size + 1):
            best = optimal_tau(data[w_nub:w_nub + w_size], p)
            if best is None:
                continue

            best.update({"start": w_nub, "w_size": w_size})
            results.append(best)
            print(f"w_nub = {w_nub} | t1 = {best['t1']} ,t2 = {best['t2']} | "
                  f"total = {best['score']:.3f} | feat = {best['feat']}")

    return results


params = {
    "area_ratio": [3, 9, 1],      # 三段时长比
    "target": [0, 1/3, 1],      # [平盘度, 幅度比, 斜率比]
    "tol": [1.0, 1.0, 1.0],       # 各特征容差
    "weight": [1.0, 1.0, 1.0],    # 各特征权重
}

#results = found(test_data, window_size=[10, 12], p=params)
#print(f"\n合法窗口数 = {len(results)}")

results = found(close_data, window_size=[25, 60], p=params)
print(f"\n合法窗口数 = {len(results)}")


# ===== 画图：score 最小的 20 个窗口 =====
from pathlib import Path
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    print(f"跳过绘图：{exc}")
else:
    sys.path.append("/Users/lyuchanghe/stu/math_practice_file/function")
    from plot import plot

    out_dir = Path(__file__).resolve().parent / "tmp"
    out_dir.mkdir(exist_ok=True)

    top = sorted(results, key=lambda r: r["score"])[:20]
    print(f"最佳 {len(top)} 个窗口 -> {out_dir}")

    for rank, r in enumerate(top, start=1):
        start, w_size = r["start"], r["w_size"]
        w = close_data[start:start + w_size]
        x = np.arange(start, start + w_size)

        _, t1_point, _, _, _ = tau_layout(w_size, params)

        # 实际收盘价
        layers = [
            {"type": "line", "data": np.column_stack([x, w]),
             "color": "black", "linestyle": "-", "linewidth": 1},
        ]

        # 三段拟合线
        segs = [
            (w[0:r["t1"]], 0, "tab:blue"),
            (w[t1_point:r["t2"]], t1_point, "tab:orange"),
            (w[r["t2"]:], r["t2"], "tab:green"),
        ]
        for ys, offset, color in segs:
            xs = np.arange(len(ys))
            k, b = fit_line(ys)
            layers.append({
                "type": "line",
                "data": np.column_stack([start + offset + xs, k * xs + b]),
                "color": color,
                "linestyle": "--",
                "linewidth": 1.5,
            })

        fig, ax = plt.subplots(figsize=(8, 4))
        title = (f"#{rank} score={r['score']:.3f} | W={w_size} start={start} "
                 f"t1={r['t1']} t2={r['t2']} | feat={r['feat']}")
        plot(layers, title=title, ax=ax, show=False)

        # 两个断点
        for bp in (start + r["t1"], start + r["t2"]):
            ax.axvline(bp, color="gray", linestyle=":", linewidth=1)

        fig.tight_layout()
        path = out_dir / f"best_{rank:02d}_score_{r['score']:.3f}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {path.name}")