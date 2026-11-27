"""运行收盘价形态识别，把去重后的命中写入 analysis/，并保存调试图片。"""

import contextlib
import io
from pathlib import Path

import pandas as pd

from finder import found, params
from plot import draw_results

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ANALYSIS_DIR = ROOT / "analysis"
HITS_FILE = ANALYSIS_DIR / "data" / "pattern_hits.csv"
WINDOW_SIZE = [20, 60]


def scan_stock(csv_path: Path) -> tuple[list[dict], list[dict]]:
    """识别单只股票，返回命中记录和原始结果。"""
    frame = pd.read_csv(csv_path, parse_dates=["date"])
    closes = frame["close"].to_numpy()

    # found 会逐窗口打印，批量扫描时屏蔽
    with contextlib.redirect_stdout(io.StringIO()):
        results = found(closes, window_size=WINDOW_SIZE, p=params)

    hits = []
    for result in results:
        start = result["start"]
        w_size = result["w_size"]
        end_index = start + w_size - 1
        row = {
            "stock": csv_path.stem,
            "start": start,
            "start_date": frame["date"].iloc[start].strftime("%Y-%m-%d"),
            "w_size": w_size,
            "t1": result["t1"],
            "t2": result["t2"],
            "end_index": end_index,
            "end_date": frame["date"].iloc[end_index].strftime("%Y-%m-%d"),
            "end_close": float(closes[end_index]),
            "score": result["score"],
        }
        for i, value in enumerate(result["feat"], start=1):
            row[f"feat_{i}"] = value
        hits.append(row)

    return hits, results


def main() -> None:
    HITS_FILE.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    results_by_stock = {}
    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        hits, results = scan_stock(csv_path)
        rows.extend(hits)
        results_by_stock[csv_path.stem] = results
        print(f"{csv_path.stem}: {len(hits)} 个命中")

    hits_frame = pd.DataFrame(rows)
    hits_frame.to_csv(HITS_FILE, index=False, encoding="utf-8-sig")
    print(f"\n命中合计 {len(hits_frame)} 个，已保存: {HITS_FILE}")

    stock_files = ("sz000001.csv", "sz000858.csv", "sz300750.csv")
    draw_results(
        DATA_DIR,
        ROOT / "tmp",
        stock_files,
        WINDOW_SIZE,
        params,
        results_by_stock={
            Path(name).stem: results_by_stock[Path(name).stem]
            for name in stock_files
        },
    )


if __name__ == "__main__":
    main()
