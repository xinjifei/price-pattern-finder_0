import numpy as np
import pandas as pd

source_path = "/Users/lyuchanghe/stu/found_0/data/sz000001.csv"

def csv_numpy(csv_path: str, col_name: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    arr = df[col_name].to_numpy()
    return arr

close_data = csv_numpy(source_path, "close")
test_data = close_data[1:20]

# 寻找函数
def found(data:np.array, window_size:list, area_ratio:list):
    day = len(data)
    # 遍历窗口尺寸
    for w_size in range(window_size[0],window_size[1]+ 1):
        tau_area = round(w_size/6)

        #断点起始位置
        t1_point = round(w_size * (area_ratio[0] / sum(area_ratio)))
        t2_point = round(w_size * ((area_ratio[0]+area_ratio[1]) / sum(area_ratio)))
        t1_start = t1_point - round(tau_area/2)
        t2_start = t2_point - round(tau_area/2)
        
        print(f"window size = {w_size} | tau_aSrea = {tau_area} ##########")
        print(f"t1_point={t1_point},t1_start={t1_start} | t2_point={t2_point},t2_start={t2_start}")

        # 字符画
        ruler = ["-"] * 100
        for p in (t1_point, t2_point):
            ruler[min(99, round(p / w_size * 100))] = "|"
        print("".join(ruler))
        
        # 滑动切片
        for w_nub in range(day - w_size+1):
            # 截取数据
            w = data[w_nub : w_nub + w_size]
            print(f"w_nub = {w_nub}")
            # 断点遍历
            for i in range(tau_area):
                t1 = t1_start + i
                # 切割第一区间
                a_1 = w[0:t1]
                for j in range(tau_area):
                    t2 = t2_start + j
                    # 边界：每段至少 2 个点
                    if t1 < 2 or t1 >= t2 or t2 - t1_point < 2 or w_size - t2 < 2:
                        continue
                    # 切割第二、三区间
                    a_2 = w[t1_point:t2]
                    a_3 = w[t2:]
                    print(f"    t1 = {t1} ,t2 = {t2}")
                    

            
            

found(test_data, window_size=[10,12], area_ratio=[3,9,1])

    