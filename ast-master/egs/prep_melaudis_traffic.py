import os
import json
import csv
import random
from pathlib import Path

# 说明：
# - 最小侵入式脚本：仅生成 AST 训练所需的 label CSV 与 train/val JSON。
# - 不进行任何训练或编译。

# 数据根目录（相对项目根）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / 'data'

# 搜索路径：车辆与背景音频
VEH_ROOT = DATA_ROOT / 'MELAUDIS_Vehicles' / 'Final_Veh'
BG_ROOT = DATA_ROOT / 'MELAUDIS_ BG' / '_BG_Final'

# 输出目录：与 AST egs 一致
OUT_DIR = PROJECT_ROOT / 'ast-master' / 'egs' / 'melaudis_traffic'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 三类定义：
# - bg：无车（背景目录）
# - free_flow：畅通（文件名含 _FF_）
# - traffic_jam：拥堵（文件名含 TJ/TJN/TJF；或车辆目录但未标记 FF 时作为回退）
FF_TOKENS = ['_FF_']
TJ_TOKENS = ['_TJ_', 'TJN', 'TJF']

def infer_label_from_name(name: str, is_bg: bool):
    base = os.path.basename(name)
    if is_bg:
        return 'bg'
    # 车辆：优先判断 FF
    for tk in FF_TOKENS:
        if tk in base:
            return 'free_flow'
    # 车辆：判断 TJ 家族
    for tk in TJ_TOKENS:
        if tk in base:
            return 'traffic_jam'
    # 车辆但无 FF 显式标注时，按定义回退为拥堵
    return 'traffic_jam'

def collect_wavs():
    items = []
    # 车辆数据
    if VEH_ROOT.exists():
        for city_dir, _, files in os.walk(VEH_ROOT.as_posix()):
            for f in files:
                if f.lower().endswith('.wav'):
                    full = os.path.join(city_dir, f)
                    label = infer_label_from_name(full, is_bg=False)
                    items.append((full, label))
    # 背景数据（若需要，可作为 free_flow 或忽略，这里不自动赋值，除非名称中含 FF/SF/HF）
    if BG_ROOT.exists():
        for root, _, files in os.walk(BG_ROOT.as_posix()):
            for f in files:
                if f.lower().endswith('.wav'):
                    full = os.path.join(root, f)
                    label = infer_label_from_name(full, is_bg=True)
                    items.append((full, label))
    return items

def build_label_csv(labels, csv_path):
    # AST 需要的列：index, mid, display_name
    # 这里 mid 使用与 display_name 相同的可读字符串
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'mid', 'display_name'])
        for idx, name in enumerate(labels):
            writer.writerow([idx, name, name])

def save_json_split(items, label_to_index, json_path):
    data = []
    for wav, label in items:
        idx_name = list(label_to_index.keys())[list(label_to_index.values()).index(label_to_index[label])]
        # AST 的 labels 字段使用 mid/display_name 文本
        data.append({'wav': wav, 'labels': label})
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'data': data}, f, ensure_ascii=False)

def main():
    items = collect_wavs()
    if len(items) == 0:
        print('未找到可用于交通状态的带标签音频（需文件名包含 _FF_/_SF_/_HF_）。')
        return

    # 统计标签
    labels = sorted(list({label for _, label in items}))
    print('发现类别：', labels)

    # 写出标签 CSV
    label_csv = OUT_DIR / 'class_labels_indices.csv'
    build_label_csv(labels, label_csv.as_posix())
    print('写出标签文件：', label_csv)

    # 划分训练/验证（80/20）
    random.seed(42)
    random.shuffle(items)
    split = int(len(items) * 0.8)
    train_items = items[:split]
    val_items = items[split:]

    # 构建 label->index
    label_to_index = {name: i for i, name in enumerate(labels)}

    # 写出 JSON
    train_json = OUT_DIR / 'datafiles_train.json'
    val_json = OUT_DIR / 'datafiles_val.json'
    save_json_split(train_items, label_to_index, train_json.as_posix())
    save_json_split(val_items, label_to_index, val_json.as_posix())
    print('写出训练/验证 JSON：', train_json, val_json)

    # 额外写出类别统计
    def count_by(items_):
        c = {}
        for _, lb in items_:
            c[lb] = c.get(lb, 0) + 1
        return c
    stats_path = OUT_DIR / 'split_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({'total': len(items), 'train': count_by(train_items), 'val': count_by(val_items)}, f, ensure_ascii=False, indent=2)
    print('写出划分统计：', stats_path)

if __name__ == '__main__':
    main()


