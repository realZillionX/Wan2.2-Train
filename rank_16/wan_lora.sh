#!/bin/bash
set -e

# ============ 离线模式配置 ============
export DIFFSYNTH_SKIP_DOWNLOAD=True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_P2P_DISABLE=1

# ============ 模型路径配置 ============
MODEL_BASE_PATH="/inspire/hdd/project/embodied-multimodality/public/downloaded_ckpts/Wan2.2-TI2V-5B"
TOKENIZER_PATH="${MODEL_BASE_PATH}/google/umt5-xxl"

# 模型文件路径（JSON 格式）
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
METADATA_PATH="/inspire/hdd/project/embodied-multimodality/tongjingqi-CZXS25110029/chj_code/wan_train.csv"

# ============ 视频配置 ============
NUM_FRAMES=249     
HEIGHT=480
WIDTH=832

# ============ 训练配置 ============
DATASET_REPEAT=1
LEARNING_RATE=1e-4
NUM_EPOCHS=3
LORA_RANK=16
GRADIENT_ACCUMULATION=1

# ============ 输出配置 ============
OUTPUT_PATH="/inspire/hdd/project/embodied-multimodality/tongjingqi-CZXS25110029/chj_code/Wan2.2-TI2V-5B_lora_rank16"
SAVE_STEPS=250

# ============ DiffSynth-Studio 路径 ============
DIFFSYNTH_PATH="/inspire/hdd/project/embodied-multimodality/tongjingqi-CZXS25110029/chj_code/DiffSynth-Studio"
TRAIN_SCRIPT="/inspire/hdd/project/embodied-multimodality/tongjingqi-CZXS25110029/chj_code/train.py"

# ============ 训练命令 ============
cd ${DIFFSYNTH_PATH}

echo "========================================"
echo "训练配置（离线模式）- Rank=16："
echo "  - 模型路径: ${MODEL_BASE_PATH}"
echo "  - Tokenizer: ${TOKENIZER_PATH}"
echo "  - 数据集: ${METADATA_PATH}"
echo "  - 帧数: ${NUM_FRAMES}"
echo "  - LoRA Rank: ${LORA_RANK}"
echo "  - 学习率: ${LEARNING_RATE}"
echo "  - Epochs: ${NUM_EPOCHS}"
echo "  - 输出路径: ${OUTPUT_PATH}"
echo "========================================"
echo ""

accelerate launch ${TRAIN_SCRIPT} \
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

echo ""
echo "训练完成！模型保存在: ${OUTPUT_PATH}"
