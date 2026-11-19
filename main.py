"""运行收盘价形态识别，并保存调试图片。"""

from pathlib import Path

from finder import csv_numpy, found, params
from plot import draw_results


def main():
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    window_size = [20, 60]

    close_data = csv_numpy(str(data_dir / "sz300750.csv"), "close")
    results = found(close_data, window_size=window_size, p=params)
    print(f"\n合法窗口数 = {len(results)}")

    stock_files = ("sz000001.csv", "sz000858.csv", "sz300750.csv")
    draw_results(data_dir, root / "tmp", stock_files, window_size, params)


if __name__ == "__main__":
    main()
