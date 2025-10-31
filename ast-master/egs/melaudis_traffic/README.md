### MELAUDIS 交通状态监测（AST）

本目录包含使用 AST 模型进行交通状态监测的最小配置与数据文件。

#### 1) 生成数据文件
- 已提供脚本：`ast-master/egs/prep_melaudis_traffic.py`
- 作用：扫描 `data/` 下的 `.wav` 文件，依据以下规则推断标签并生成数据文件：
  - 背景目录 `data/MELAUDIS_ BG/_BG_Final` → `bg`
  - 车辆目录中文件名包含 `_FF_` → `free_flow`
  - 车辆目录中文件名包含 `TJN`、`TJF` 或 `_TJ_`，以及未包含 `_FF_` 的车辆样本 → `traffic_jam`
  - `class_labels_indices.csv`
  - `datafiles_train.json`
  - `datafiles_val.json`

注意：脚本会生成三类标签（`bg`/`free_flow`/`traffic_jam`）。可查看 `split_stats.json` 了解三类的样本数量与划分情况。

#### 2) 训练命令（PowerShell）
在项目根目录执行（不会自动执行、需手动运行）：

```powershell
python ast-master/src/run.py `
  --data-train ast-master/egs/melaudis_traffic/datafiles_train.json `
  --data-val   ast-master/egs/melaudis_traffic/datafiles_val.json `
  --label-csv  ast-master/egs/melaudis_traffic/class_labels_indices.csv `
  --n_class 3 `
  --model ast `
  --dataset melaudis_traffic `
  --exp-dir ast-master/egs/exp_melaudis_traffic `
  --metrics acc `
  --loss CE `
  --imagenet_pretrain True `
  --audioset_pretrain False `
  --freqm 48 --timem 48 `
  -b 12 --num-workers 4 `
  --n-epochs 20 --lr 5e-5 --warmup False
```

说明：
- `--n_class` 应与 `class_labels_indices.csv` 中的类别数一致。
- 可根据硬件适当调整 `-b`（batch size）、`--num-workers`、`--n-epochs` 等参数。

#### 3) 结果
- 训练过程会在 `ast-master/egs/exp_melaudis_traffic` 下生成：
  - `models/`：各 epoch 的权重与最佳模型
  - `result.csv`：各 epoch 的指标
  - `predictions/`：验证集预测与集成预测


