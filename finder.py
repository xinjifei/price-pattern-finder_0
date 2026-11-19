"""从收盘价序列中寻找符合三段特征的窗口。"""

import numpy as np
import pandas as pd


def csv_numpy(csv_path: str, col_name: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    arr = df[col_name].to_numpy()
    return arr


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

    # 硬门槛1：第一段、第三段必须涨
    if A1 <= 0 or A3 <= 0:
        return None

    # 门槛2:第三段起始点与第二段终点落差，与第三段拟合线落差之比不得超过maxdrop
    # 算两个落差
    drop_0 = ((len(a_2) -1)*s2 + b2) - (b3)
    drop_1 = ((len(a_3) -1)*s3 + b3) - (b3)
    max_drop = 0.35
    if drop_1 / drop_0 >= max_drop:
        return None

    # 门槛3：第二段终点位置不能低于第一段lower_line
    lower_line_coef = 0.66
    lower_line = (b1) + (((len(a_1) -1)*s1+b1) - b1)*lower_line_coef
    if ((len(a_2) -1)*s2 + b2) < lower_line:
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
    """单窗口内固定 t2、搜索 t1，返回最佳分段；无合法分段返回 None。"""
    ##window size，当前窗口长度
    w_size = len(w)
    tau_area, _, t2_point, t1_start, _ = tau_layout(w_size, p)

    best = None
    for i in range(tau_area):
        ##turning point，第一个实际断点
        t1 = t1_start + i
        t2 = t2_point
        
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


def dedup(results: list, overlap_ratio: float = 0.7):
    """重叠达到较短窗口指定比例时，只保留分数更低的窗口。"""
    kept = []

    ##result，按分数从小到大检查的窗口
    for result in sorted(results, key=lambda r: r["score"]):
        start = result["start"]
        end = start + result["w_size"]

        ##saved，已经保留的窗口
        for saved in kept:
            saved_start = saved["start"]
            saved_end = saved_start + saved["w_size"]
            overlap = max(0, min(end, saved_end) - max(start, saved_start))
            if overlap >= overlap_ratio * min(result["w_size"], saved["w_size"]):
                break
        else:
            kept.append(result)

    return kept


def found(data: np.ndarray, window_size: list, p: dict):
    """逐窗口滑动，每个窗口留一个最佳分段，除重后返回。"""
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
    # 除重处理
    results = dedup(results)
    return results


params = {
    "area_ratio": [3, 6, 2],      # 三段时长比
    "target": [0, 1/3, 1, 0, 0],  # [平盘度, 幅度比, 斜率比, 标准差, 断点高度差]
    "tol": [1.0, 1.0, 1.0, 1.0, 1.0],
    "weight": [1.0, 1.0, 1.0, 1.0, 1.0],
    "std_weight": [1.0, 2.0, 1.0],  # 三段标准差的权重
    "gap_max": 0.5,                 # 断点高度差与 A1 的比值上限
}
