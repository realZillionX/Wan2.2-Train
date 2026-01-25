#!/bin/bash
set -e

export DIFFSYNTH_SKIP_DOWNLOAD=True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PROJECT_BASE="/inspire/hdd/project/embodied-multimodality/tongjingqi-CZXS25110029/chj_code/Wan2.2-Train"

MODEL_BASE_PATH="/inspire/hdd/project/embodied-multimodality/public/downloaded_ckpts/Wan2.2-TI2V-5B"
TOKENIZER_PATH="${MODEL_BASE_PATH}/google/umt5-xxl"

MODEL_PATHS='[
    [
        "'${MODEL_BASE_PATH}'/diffusion_pytorch_model-00001-of-00003.safetensors",
        "'${MODEL_BASE_PATH}'/diffusion_pytorch_model-00002-of-00003.safetensors",
        "'${MODEL_BASE_PATH}'/diffusion_pytorch_model-00003-of-00003.safetensors"
    ],
    "'${MODEL_BASE_PATH}'/models_t5_umt5-xxl-enc-bf16.pth",
    "'${MODEL_BASE_PATH}'/Wan2.2_VAE.pth"
]'
# ============ 数据集配置 ============
DATASET_BASE_PATH=""
METADATA_PATH="${PROJECT_BASE}/wan_train.csv"

# 自动合并 CSV
python3 -c "
import csv, os
files = [
    '/inspire/hdd/project/embodied-multimodality/public/hjchen/MazeSquare/part_7x7.csv',
    '/inspire/hdd/project/embodied-multimodality/public/hjchen/MazeSquare/part_9x9.csv',
    '/inspire/hdd/project/embodied-multimodality/public/hjchen/MazeSquare/part_11x11.csv'
]
output_file = '${PROJECT_BASE}/wan_train.csv'
print(f'Merging {len(files)} CSV files into {output_file}...')
header = None
total_rows = 0
try:
    with open(output_file, 'w', newline='') as fout:
        writer = csv.writer(fout)
        for f_path in files:
            if not os.path.exists(f_path):
                print(f'Warning: File not found: {f_path}')
                continue
            with open(f_path, 'r') as fin:
                reader = csv.reader(fin)
                try:
                    h = next(reader)
                    if header is None:
                        header = h
                        writer.writerow(header)
                    for row in reader:
                        writer.writerow(row)
                        total_rows += 1
                except StopIteration:
                    pass
    print(f'Success! Merged {total_rows} rows.')
except Exception as e:
    print(f'Error merging CSVs: {e}')
    exit(1)
"

# ============ 视频配置 ============
NUM_FRAMES=223
HEIGHT=873
WIDTH=480

# ============ 训练配置 ============
DATASET_REPEAT=1
LEARNING_RATE=1e-4
NUM_EPOCHS=3
LORA_RANK=32
GRADIENT_ACCUMULATION=1

# ============ 输出配置 ============
OUTPUT_PATH="${PROJECT_BASE}/output/rank32_multi"
SAVE_STEPS=250

# ============ 路径配置 ============
DIFFSYNTH_PATH="/inspire/hdd/project/embodied-multimodality/tongjingqi-CZXS25110029/chj_code/DiffSynth-Studio"
CONFIG_FILE="$(dirname "$0")/accelerate_config_multi_node.yaml"
TRAIN_SCRIPT="${PROJECT_BASE}/train.py"

echo "========================================"
echo "Wan2.2-TI2V-5B LoRA 多机分布式训练 (Rank=32)"
echo "========================================"
echo "  LoRA Rank: ${LORA_RANK}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Resolution: ${WIDTH}x${HEIGHT}"
echo "  Frames: ${NUM_FRAMES}"
echo "  输出路径: ${OUTPUT_PATH}"
echo "========================================"



cd ${DIFFSYNTH_PATH}

accelerate launch \
  --config_file "${CONFIG_FILE}" \
  ${TRAIN_SCRIPT} \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${METADATA_PATH}" \
  --height ${HEIGHT} \
  --width ${WIDTH} \
  --num_frames ${NUM_FRAMES} \
  --dataset_repeat ${DATASET_REPEAT} \
  --model_paths "${MODEL_PATHS}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --learning_rate ${LEARNING_RATE} \
  --num_epochs ${NUM_EPOCHS} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank ${LORA_RANK} \
  --extra_inputs "input_image" \
  --use_gradient_checkpointing \
  --save_steps ${SAVE_STEPS}

echo "训练完成！模型保存在: ${OUTPUT_PATH}"
