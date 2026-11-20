# Preliminary Experiments

基于现有数据集 CRIA 和框架 MAIA，探索具有可复用性的 Criterion-conditioned Verifier。模型根据给定的任务及Criterion，利用视觉、动作、力觉等交互信息输出 Judgment、Evidence 和 Confidence，并进一步研究如何将多个 Criterion 的判断组合为整体任务 Outcome。初步实验分为 A. Criterion-level Verification 和 B. Criterion Composition 两部分。

## Data

任务选择： 从 CRIA 的五类操作中各选择一个任务，覆盖插入、擦拭、按压、约束路径运动及关节物体操作。

数据规模：每个任务 train 50、val 10、test 15，尽量覆盖 human/pi05；实验监督统一解释为 anomaly/success。

Criterion 设计：

### 1. 简易版（用于A）

| Task | Goal | Interaction |
| ---- | ---- | ----------- |
| open_box2 | Is the box lid open enough to expose the interior?<br>Is the box lid sliding leftward? | |
| wipe_board | Is the target region of the whiteboard free of markings? | Is the eraser sliding along the whiteboard while maintaining contact?<br>擦除需要表面运动和持续接触两个条件，单纯看到末端在移动，无法确认橡皮擦是否真正接触白板。可以利用末端运动、视觉中的橡皮擦位置以及法向力共同辅助标注。 |
| press_switch2 | Is the LED strip illuminated?<br>Is the red button depressed? | 按钮可能被按下但尚未成功触发灯带；如果 LED 在按钮松开后仍保持点亮，C1 和 C2 也会产生不同的结果。 |
| pushi | Is the block within the yellow target area?<br>Is the block moving along the constrained track? | |
| insert_plug_2 | Is the plug fully inserted into the socket?<br>Is the plug advancing into the socket? | |

### 2. 详细版（用于A、B）

| Task | Goal | Interaction |
| ---- | ---- | ----------- |
| open_box2 | Is the box lid open enough to expose the interior? | Is the box lid sliding leftward?<br>盖子是否向左运动<br>Is the gripper securely holding the lid handle?<br>保持抓取关系 |
| wipe_board | Is the target region of the whiteboard free of markings? | Is the eraser in contact with the whiteboard?<br>是否接触白板<br>Is the eraser applying sufficient pressure against the whiteboard?<br>接触压力是否足以进行擦拭<br>Is the eraser moving over the marked region?<br>是否沿目标擦拭区域运动 |
| press_switch2 | Is the LED strip illuminated? | Is the gripper pressing the red button?<br>按钮状态<br>Is the red button fully depressed?<br>有效按压事件 |
| pushi | Is the block within the yellow target area? | Is the block moving along the constrained track?<br>沿轨道实际运动<br>Is the gripper securely holding the block?<br>保持抓取关系<br>Is the block jammed against the track boundary?<br>是否发生卡阻 |
| insert_plug_2 | Is the plug fully inserted into the socket? | Is the plug advancing into the socket?<br>实际插入运动<br>Are the plug prongs aligned with the socket openings?<br>插入对齐<br>Is the plug blocked during an insertion attempt?<br>插入卡阻 |

## A. Criterion-conditioned Verification

目标： 验证预训练多模态交互表征能否支持不同 Criterion 的判断及证据定位。

模型： 冻结现有 VAF-WM，利用其多模态 Interaction Representation，通过 Criterion-conditioned Cross-Attention 提取相关特征，训练轻量级 Judgment Head 和 Evidence Head。

输入与输出： 输入局部多模态观测窗口、任务描述和 Criterion；输出 Judgment（Yes/No）、Evidence（相关时间片段？）及 Confidence。

## B. Criterion Composition

目标： 研究如何将多个 Criterion 的局部判断及其时间关系组合为整体任务 Outcome。

输入与输出： 输入同一轨迹中多个 Criterion 的 Judgment 序列及相关时间信息，输出轨迹级任务成功失败判断或正常异常判断。

候选方法： 比较基于人工规则的组合、Graph/Tree 结构化组合，以及轻量级神经网络学习组合。根据具体任务定义 Criterion 之间的逻辑关系与时间依赖。

## C. Black-box ML 实现

`blackbox_experiment.py` 实现逐帧 Criterion Judgment 到逐帧 anomaly/success 的轻量级神经网络基线。第 \(i\) 帧的每个 Criterion 编码为两个通道：二值状态和可观测掩码；`-1` unknown 编码为 `(state=0, observable=0)`，确定的 `0/1` 编码为 `(state=value, observable=1)`。模型为单隐藏层 MLP，默认隐藏维度 16，使用带类别权重的交叉熵处理异常帧较少的问题。

本实验只使用 `inputs/` 作为模型数据和监督入口，不引入 Graph、规则答案或其他外部监督。Criterion Judgment 位于 `inputs/criterionANDjudgment/`；逐帧监督位于 `inputs/frameAnomalyLabels/`，由 `E:/DenoteDatabase/CRIA/CRIA` 中对应 episode 的 Parquet `frame_label` 原样提取，其中 `0 → anomaly`、`1 → success`。数据加载器按 task、episode ID 和零基 frame index 严格对齐，并检查帧数、split 和标签范围。`episode_label` 不参与训练。

### 滚动训练与回测

抽样单位仍是完整 episode，而模型样本单位是 frame。仅使用官方 `train` split，默认执行 5 次独立重复；每次用不同且被记录的 seed 随机排列训练 episode，并依次获取 10、20、30、40、50 条。因此同一抽样条数会得到 5 个不同的 episode 子集，而不是只报告单一集合。每轮使用已获取 episode 的全部帧重新训练，并在尚未获取的完整 train episode 上回测；同一 episode 绝不会同时进入训练和回测。每个任务保留全部重复的 episode 顺序、各轮抽样 ID、逐次指标、逐帧预测和模型，同时汇总 Accuracy、Macro-F1、episode macro accuracy 的均值、标准差、最小值和最大值。官方 `val` 始终独立；官方 `test` 由每次使用全部 train episode 的最终模型分别评估。每个任务独立训练，避免不同 Criterion 空间混用。

运行：

```bash
python blackbox_experiment.py
```

可调整参数：

```bash
python blackbox_experiment.py --train-steps 10,20,30,40,50 --repeats 5 --hidden-dim 16 --epochs 100 --seed 42
```

输入目录为 `inputs/criterionANDjudgment/` 和 `inputs/frameAnomalyLabels/`。

输出目录：`outputs/`，包括：

- `frame_rolling_backtest_results.json`：每个任务全部重复的抽样集合、逐次指标，以及按抽样规模计算的均值和标准差；
- `frame_predictions.json`：每次重复的验证集与测试集逐帧真实标签、预测标签和概率；
- `<task>_frame_model_repeat_<NN>.pt`：每个任务每次重复的最终逐帧模型、seed 和训练 episode 顺序。

主要指标为 frame Accuracy、Macro-F1、逐类 Precision/Recall/F1、Confusion Matrix，以及先按 episode 计算再等权平均的 `episode_macro_accuracy`。固定 seed 只保证相同软件与硬件环境下尽量可复现；正式结果建议使用多个 seed 汇总均值和方差。
