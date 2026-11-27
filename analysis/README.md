# analysis

每个分析单独一个子目录，代码和输出放在该子目录内；多个分析共用的中间结果放在 `data/`。

```text
analysis/
├── data/
│   └── pattern_hits.csv          # main.py 生成的去重命中缓存
├── forward_returns/
│   ├── forward_returns.py        # 计算形态结束后 1-20 日累计收益
│   ├── conclusion.md
│   └── output/
│       ├── forward_return_paths.csv
│       ├── forward_return_summary.csv
│       └── forward_return_paths.png
└── random_control/
    ├── random_control.py         # 随机 20 日窗口对照组
    └── output/
        ├── random_control_paths.csv
        ├── random_control_summary.csv
        ├── random_control_paths.png
        ├── comparison_summary.csv
        └── hit_vs_control.png
```
