# CRIA Criterion-to-Anomaly 黑盒实验

本项目使用每帧的 Criterion Judgment 预测同一帧的 `frame_label`：

- 输入：各 Criterion 的 Judgment，`0=No`、`1=Yes`；代码兼容 `-1=Unknown`，但仓库现有数据仅包含 `0/1`。
- 监督与输出类别：`0=anomaly`、`1=success`。
- `success` 表示当前帧正常，不表示整条 episode 已完成任务。

项目包含两种方法：

1. `blackbox_experiment.py`：只使用当前帧 Judgment 的单帧基线。
2. `history_window_experiment.py`：使用同一 episode 内截至当前帧的因果历史统计，不读取未来帧。

详细方法和实验结果见 `report.md`，输入 JSON 格式见 `schema.md`。

## 环境

需要 Python 3.10 或更高版本，并安装：

```bash
pip install numpy torch
```

## 仓库内容

仓库已经包含可直接运行的五个任务输入：

```text
inputs/
├── criterionANDjudgment/  # Criterion Judgment
└── frameAnomalyLabels/    # 逐帧 anomaly/success 标签
```

程序按 JSON 顶层的 `task` 字段配对 Criterion 与标签文件，而不是按文件名配对。两侧的 task、episode ID、split、帧数必须一致。

每个任务使用 `train`、`val`、`test` 三个 episode 级划分。训练和评估不会把同一 episode 的帧拆到不同划分中。

## 运行单帧实验

使用默认配置：

```bash
python blackbox_experiment.py
```

等价于：

```bash
python blackbox_experiment.py --train-steps 10,20,30,40,50 --repeats 5 --hidden-dim 16 --epochs 100 --learning-rate 0.01 --seed 42
```

单帧方法将每个 Criterion 编码为 `state` 和 `observable`，因此任务包含 \(K\) 个 Criterion 时，输入维度为 \(2K\)。

## 运行历史实验

默认比较 10 帧、30 帧和完整因果历史：

```bash
python history_window_experiment.py
```

等价于：

```bash
python history_window_experiment.py --windows 10,30,full --train-steps 10,20,30,40,50 --repeats 5 --hidden-dim 16 --epochs 100 --learning-rate 0.01 --seed 42
```

`--windows` 接受逗号分隔的正整数和 `full`，例如：

```bash
python history_window_experiment.py --windows 5,15,60,full
```

整数 `N` 表示只使用同一 episode 的最近 N 帧 `[t-N+1,t]`；`full` 表示使用 `[0,t]`。每个 Criterion 被编码为 6 个固定维度特征：当前状态、当前可观测性、窗口 Yes 比例、是否曾出现 Yes、距最近一次 Yes 的归一化时间和 Unknown 比例。

## 常用参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--criterion-dir` | `inputs/criterionANDjudgment` | Criterion Judgment JSON 目录 |
| `--label-dir` | `inputs/frameAnomalyLabels` | 逐帧标签 JSON 目录 |
| `--output-dir` | `outputs` | 结果与模型目录 |
| `--train-steps` | `10,20,30,40,50` | 每轮使用的完整训练 episode 数 |
| `--repeats` | `5` | 独立 episode 排列次数，必须至少为 2 |
| `--hidden-dim` | `16` | MLP 隐藏维度 |
| `--epochs` | `100` | 每个模型的 Full-batch 训练轮数 |
| `--learning-rate` | `0.01` | AdamW 学习率 |
| `--seed` | `42` | 基础随机种子 |
| `--windows` | `10,30,full` | 仅历史实验使用的因果窗口 |

自定义输入和输出示例：

```bash
python blackbox_experiment.py --criterion-dir inputs/criterionANDjudgment --label-dir inputs/frameAnomalyLabels --output-dir outputs/run_single

python history_window_experiment.py --criterion-dir inputs/criterionANDjudgment --label-dir inputs/frameAnomalyLabels --output-dir outputs/run_history --windows 10,30,full
```

## 输出

单帧实验生成：

```text
outputs/
├── frame_rolling_backtest_results.json
├── frame_predictions.json
└── <task>_frame_model_repeat_<NN>.pt
```

历史实验生成：

```text
outputs/
├── history_window_backtest_results.json
├── history_window_predictions.json
└── <task>_history_w<window>_model_repeat_<NN>.pt
```

`outputs/` 是运行时生成目录，已被 `.gitignore` 忽略，因此克隆仓库后不会看到已有实验输出，运行脚本后才会在本地创建。预测 JSON 和模型文件可能较大；如需保留不同参数的多组结果，请为每次运行指定不同的 `--output-dir`。

## 评估协议

- 划分单位是完整 episode，训练和预测单位是 frame。
- 每次 repeat 都重新排列训练 episode；同一 repeat 的训练规模是该排列的嵌套前缀。
- `remaining-train` 回测使用当前训练规模之外的训练 episode。
- 使用全部训练 episode 的最终模型在 val 和 test 上评估。
- 50 个训练 episode 全部使用时没有剩余训练 episode，因此不定义 `remaining-train` 指标。
- 主要指标包括 frame Accuracy、frame Macro-F1 和 episode-macro Accuracy。

默认实验会训练多个任务、训练规模、repeat 和历史窗口，运行时间和输出体积可能较大。
