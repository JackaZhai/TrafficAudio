基于 Transformer 的智能交通音频识别平台
=================================

本仓库聚焦于利用音频信号理解道路交通状态，并围绕 MELAUDIS 数据集构建了一套完整的训练与评估流程。项目的最新版本已经将
[AST：Audio Spectrogram Transformer](https://github.com/YuanGongND/ast)
引入交通状态监测任务，通过 Transformer 对声学频谱建模以获得更高的识别精度。

项目结构
--------

* `ast-master/`：原始 AST 代码与预训练权重下载脚本。
* `traffic_status_monitoring/`：交通状态分类训练脚本与实用函数。
* `data/`：建议放置本地的 MELAUDIS 音频数据（仓库中不包含真实音频）。

核心任务：交通状态监测
----------------------

`traffic_status_monitoring/train.py` 提供了一个端到端的训练脚本，用于将原始 WAV 音频转换成对数 Mel 频谱并输入 AST 模型进行分类。
该脚本支持两种常见的数据集组织方式：

1. **`ImageFolder` 布局** —— 数据按 `train/val[/test]/<类别>` 目录存放，适合已经人工划分的自建数据集。
2. **MELAUDIS 原始布局** —— 指向 `MELAUDIS_Vehicles/Final_Veh` 等包含 WAV 文件的根目录，脚本会依据文件名中 `_FF_`、`_SF_`、`_HF_`、
   `_TJ_` 等片段推断交通状态标签，并按照给定比例进行分层划分。

这些片段的含义如下：

| 片段 | 英文含义 | 中文说明 |
| --- | --- | --- |
| `_FF_` | Free Flow | 畅通 / 自由流 |
| `_SF_`, `_SL_`, `_LF_` | Slow/Light Flow | 轻度拥堵 / 低速通行 |
| `_HF_`, `_HC_`, `_TJ_`, `_TJF_`, `_TJN_` | Heavy Flow / Traffic Jam | 严重拥堵 / 交通阻塞 |

训练脚本亮点
------------

* **AST 模型集成**：默认启用 ImageNet + AudioSet 预训练的 AST Base 模型，结合 Transformer 的全局注意力建模能力。
* **自动特征提取**：使用 Torchaudio 生成 128 维对数 Mel 频谱，自动裁剪/填充到固定时长，并在训练阶段随机平移与加性噪声增强。
* **自适应数据拆分**：对 MELAUDIS 原始文件结构执行分层抽样，保证各类样本在训练/验证/测试集中的占比稳定。
* **训练记录齐全**：保存最佳模型权重、标签映射与完整的指标曲线，便于后续部署与分析。

快速上手
--------

1. **准备环境**：安装 PyTorch、Torchaudio、timm==0.4.5 等依赖，并确保可以访问 GPU（若可选）。
2. **下载 MELAUDIS 数据集**：将 WAV 文件置于 `data/` 目录或任意路径。
3. **运行训练脚本**：

```bash
# 针对 ImageFolder 布局
python -m traffic_status_monitoring.train \
    --data-root dataset_root \
    --output-dir experiments/traffic_status_ast \
    --epochs 30 \
    --batch-size 16

# 针对 MELAUDIS 原始布局（自动划分 70/15/15）
python -m traffic_status_monitoring.train \
    --data-root data/MELAUDIS_Vehicles/Final_Veh \
    --output-dir experiments/melaudis_ast \
    --val-ratio 0.15 --test-ratio 0.15 \
    --audioset-pretrain
```

常用参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--sample-rate` | 16000 | 统一的音频采样率（Hz）。 |
| `--duration` | 5.0 | 裁剪/填充后的音频长度（秒）。 |
| `--n-mels` | 128 | Mel 频谱通道数，AST 默认支持 128。 |
| `--fstride` / `--tstride` | 10 / 10 | AST Patch 的频率/时间步长，可根据时频分辨率调整。 |
| `--imagenet-pretrain` | 启用 | 是否使用 ImageNet 预训练权重。 |
| `--audioset-pretrain` | 关闭 | 是否加载提供的 AudioSet 预训练权重（需额外下载，建议开启）。 |

训练结束后输出目录包含：

* `best_model.pt`：验证集表现最佳的 AST 模型参数。
* `label_mapping.json`：类别索引与语义标签的映射关系。
* `metrics.json`：每个 Epoch 的损失、准确率以及可选的测试集评估结果。

下一步工作
----------

* **车辆检测 / 车辆类型识别**：可复用相同的数据处理与 AST 建模框架，构建二分类或多分类任务。
* **系统集成**：结合三个子模型以及 GUI，构建完整的智能交通监测系统，实现实时或离线的音频识别展示。
* **可视化与报告**：基于 `metrics.json` 绘制训练曲线，并输出混淆矩阵、样例预测等分析结果。
