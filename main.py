import numpy as np
import pandas as pd

source_path = "~/stu/found_0/data/sz300750.csv"


def csv_numpy(csv_path: str, col_name: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    arr = df[col_name].to_numpy()
    return arr


close_data = csv_numpy(source_path, "close")
test_data = close_data[1:20]


def fit_line(ys: np.ndarray):
    xs = np.arange(len(ys))
    ##slope，斜率
    ##intercept，截距
    k, b = np.polyfit(xs, ys, deg=1)
    return k, b


def score(area_list: list, p: dict):
    """三段特征打分。返回 (total, feat)；不满足硬门槛时返回 None。"""
    ##parameters，形态参数
    a_1, a_2, a_3 = area_list

    ##slope，第一段斜率
    ##intercept，第一段截距
    s1, b1 = fit_line(a_1)
    s2, b2 = fit_line(a_2)
    s3, b3 = fit_line(a_3)

    # 段内净变化 = 斜率 × 首尾间隔
    ##amplitude，第一段净变化
    A1 = s1 * (len(a_1) - 1)
    A2 = s2 * (len(a_2) - 1)
    A3 = s3 * (len(a_3) - 1)

    # 硬门槛：第一段、第三段必须涨
    if A1 <= 0 or A3 <= 0:
        return None

    # 三段残差的标准差：数据点离拟合线越远，数值越大
    ##x coordinates，第一段横坐标数组
    xs1 = np.arange(len(a_1))
    xs2 = np.arange(len(a_2))
    xs3 = np.arange(len(a_3))

    ##standard deviation，第一段残差标准差
    std1 = np.std(a_1 - (s1 * xs1 + b1))
    std2 = np.std(a_2 - (s2 * xs2 + b2))
    std3 = np.std(a_3 - (s3 * xs3 + b3))

    std = 0.0
    ##weight，单段标准差权重
    for std_i, w in zip([std1, std2, std3], p["std_weight"]):
        std += std_i * w
    std = std / sum(p["std_weight"])
    std = std / A1

    # 相邻拟合线在两个断点处的高度差
    gap1 = abs((s1 * (len(a_1) - 1) + b1) - b2)
    gap2 = abs((s2 * (len(a_2) - 1) + b2) - b3)
    gap = (gap1 + gap2) / A1

    # 硬门槛：拟合线在断点处不能过于割裂
    if gap > p["gap_max"]:
        return None

    ##feature，特征
    feat = [
        abs(A2) / A1,   # 平盘度
        A3 / A1,        # 幅度比
        s3 / s1,        # 斜率比
        std,             # 三段残差标准差的加权均值
        gap,             # 两个断点的拟合线高度差
    ]

    # 偏差归一化，加权平方和（越小越像）
    total = 0.0
    ##feature，当前特征值
    ##target，当前特征的目标值
    ##tolerance，当前特征的容差
    for f, t, e, w in zip(feat, p["target"], p["tol"], p["weight"]):
        total += w * ((f - t) / e) ** 2

    return float(total), [float(x) for x in feat]


def tau_layout(w_size: int, p: dict):
    """返回 (tau_area, t1_point, t2_point, t1_start, t2_start)。"""
    ##tolerance area，断点搜索区间宽度
    tau_area = round(w_size / 6)
    ratio_sum = sum(p["area_ratio"])
    ##turning point，第一个预设断点
    t1_point = round(w_size * p["area_ratio"][0] / ratio_sum)
    t2_point = round(w_size * (p["area_ratio"][0] + p["area_ratio"][1]) / ratio_sum)
    half = tau_area // 2

    return tau_area, t1_point, t2_point, t1_point - half, t2_point - half


def optimal_tau(w: np.ndarray, p: dict):
    """单窗口内搜两个断点，返回最佳 {'score','t1','t2','feat'}；无合法分段返回 None。"""
    ##window size，当前窗口长度
    w_size = len(w)
    tau_area, t1_point, _, t1_start, t2_start = tau_layout(w_size, p)

    best = None
    for i in range(tau_area):
        ##turning point，第一个实际断点
        t1 = t1_start + i
        for j in range(tau_area):
            t2 = t2_start + j

            # 边界：t1 < t2，且每段至少 2 个点
            if not (2 <= t1 < t2 and t2 - t1 >= 2 and w_size - t2 >= 2):
                continue

            ##result，当前分段的打分结果
            res = score([w[0:t1 + 1], w[t1:t2 + 1], w[t2:]], p)
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
        ##window number，窗口编号
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
    "area_ratio": [3, 8, 2],      # 三段时长比
    "target": [0, 1/3, 1, 0, 0],  # [平盘度, 幅度比, 斜率比, 标准差, 断点高度差]
    "tol": [1.0, 1.0, 1.0, 1.0, 1.0],
    "weight": [1.0, 1.0, 1.0, 1.0, 1.0],
    "std_weight": [1.0, 2.0, 1.0],  # 三段标准差的权重
    "gap_max": 0.5,                 # 断点高度差与 A1 的比值上限
}

#results = found(test_data, window_size=[10, 12], p=params)
#print(f"\n合法窗口数 = {len(results)}")

results = found(close_data, window_size=[14, 60], p=params)
print(f"\n合法窗口数 = {len(results)}")


# ===== 画图：score 最小的 20 个窗口 =====
from pathlib import Path
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
##exception，导入绘图库失败时的异常信息
except ImportError as exc:
    print(f"跳过绘图：{exc}")
else:
    # sys.path 不展开 "~"，必须用 Path.home() 拼成绝对路径
    ##plot_dir，共享绘图函数 plot.py 所在目录
    plot_dir = Path.home() / "stu" / "math_practice_file" / "function"
    sys.path.append(str(plot_dir))
    from plot import plot

    out_dir = Path(__file__).resolve().parent / "tmp"
    out_dir.mkdir(exist_ok=True)

    ##ranked，全部合法结果按分数升序（越小越像）
    ranked = sorted(results, key=lambda r: r["score"])

    ##group_size，每组多少个窗口
    ##group_num，抽查的组数
    ##pick，每组画出前几名
    group_size, group_num, pick = 20, 10, 3

    print(f"合法结果 {len(ranked)} 个，按分数分 {group_num} 组、每组 {group_size} 个，"
          f"每组画前 {pick} 名 -> {out_dir}")

    for g in range(group_num):
        ##group_start，本组第一个窗口的全局排名下标
        group_start = g * group_size
        group = ranked[group_start:group_start + group_size]
        if not group:
            print(f"\n第 {g + 1} 组没有数据，抽查结束")
            break

        print(f"\n第 {g + 1} 组（总排名 {group_start + 1}-{group_start + len(group)}）：")

        ##r，本组内的单个匹配结果
        for offset, r in enumerate(group[:pick]):
            ##rank，该窗口的全局名次
            rank = group_start + offset + 1

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
            title = (f"组{g + 1} 第{offset + 1}名（总#{rank}） score={r['score']:.3f} | "
                     f"W={w_size} start={start} t1={r['t1']} t2={r['t2']}")
            plot(layers, title=title, ax=ax, show=False)

            # 两个断点
            ##breakpoint，断点坐标
            for bp in (start + r["t1"], start + r["t2"]):
                ax.axvline(bp, color="gray", linestyle=":", linewidth=1)

            fig.tight_layout()
            path = out_dir / (f"g{g + 1:02d}_top{offset + 1}"
                              f"_rank{rank:03d}_score_{r['score']:.3f}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  {path.name}")
