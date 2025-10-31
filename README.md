基于Transformer的智能交通系统设计与实现

### 1. 三个子任务分工

系统设计分为三个核心任务，每人负责一个：

1. **车辆检测（Vehicle Detection）**

   * 基于音频特征识别是否有车辆出现。
   * 需要二分类模型：有车/无车。
2. **交通状态监测（Traffic Status Monitoring）**

   * 利用音频特征监测道路通行状态，如畅通、轻度拥堵、严重拥堵。
   * 需要多分类模型。
3. **车辆类型识别（Vehicle Type Classification）**

   * 识别车辆种类（自行车、摩托车、轿车、卡车、公交车、有轨电车等）。
   * 需要多分类模型。

---

### 2. 数据与方法

* 数据集：MELAUDIS（提供了车辆类型、数量、交通状态、背景噪声等音频样本）。
* 方法：基于 Transformer 模型（可用 Audio Spectrogram + Transformer Encoder）。
* 输出：各任务的训练模型、分类结果及性能评估（准确率、混淆矩阵等）。

#### 交通状态监测训练脚本

仓库新增了一个轻量级的多分类训练脚本 `traffic_status_monitoring/train.py`，用于基于音频的路况识别。它会自动将音频转换为对数 Mel 频谱，并训练一个卷积神经网络来区分“畅通”“轻度拥堵”“严重拥堵”等类别。

脚本同时支持两种数据布局：

1. **`ImageFolder` 结构** —— 已经人工划分好的 `train/val[/test]` 目录，每个目录下以子文件夹表示类别。
2. **MELAUDIS 原始结构** —— 直接指向 `MELAUDIS_Vehicles/Final_Veh` 等包含 WAV 文件的根目录，脚本会根据文件名中的 `_FF_`、`_SF_`、`_HF_`、`_TJ_` 等标记自动推断交通状态，并按给定比例分层划分训练/验证/测试集。

   这些缩写来源于数据集作者对交通状态的英文描述，对应关系如下：

   | 标记 | 英文含义 | 中文说明 |
   | --- | --- | --- |
   | `_FF_` | Free Flow | 畅通 / 自由流 |
   | `_SF_`, `_SL_`, `_LF_` | Slow Flow / Light Flow | 轻度拥堵 / 车速减慢 |
   | `_HF_`, `_HC_`, `_TJ_`, `_TJF_`, `_TJN_` | Heavy Flow / Traffic Jam | 严重拥堵 / 交通阻塞 |

运行示例：

```
# 针对 ImageFolder 布局
python -m traffic_status_monitoring.train \
    --data-root dataset_root \
    --output-dir experiments/traffic_status \
    --epochs 50 --batch-size 32

# 针对 MELAUDIS 数据集（自动划分 70/15/15）
python -m traffic_status_monitoring.train \
    --data-root data/MELAUDIS_Vehicles/Final_Veh \
    --output-dir experiments/melaudis_status \
    --val-ratio 0.15 --test-ratio 0.15
```

训练完成后会在输出目录生成：

* `best_model.pt`：验证集表现最好的模型参数；
* `label_mapping.json`：类别索引与标签名的映射；
* `metrics.json`：训练/验证/测试阶段的损失与准确率记录。

---

### 3. 系统整合

* 最终成果需要将三个任务整合成**一个完整的智能交通系统**。
* **设计 GUI 界面**，展示：

  * 实时或离线的音频检测结果；
  * 三个子任务的预测输出（是否有车、道路状态、车辆类型）；
  * 可视化图表或状态指示。

---

### 5. 进度与结题安排

1. **前期准备（数据+文献学习）**：熟悉数据集、阅读参考文献。
2. **中期开发（模型训练+单任务实现）**：完成你负责的子任务模型设计、训练与结果验证。
3. **后期整合（系统集成+GUI设计）**：和团队成员合并各自模块，完成整体系统设计。
4. **结题成果**：
   * 系统源码（含模型代码、GUI程序）。
   * 实验报告与答辩展示文档。
   * 可运行的系统 Demo。
