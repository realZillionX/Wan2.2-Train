#!/bin/bash
set -e

export DIFFSYNTH_SKIP_DOWNLOAD=True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_P2P_DISABLE=1

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

DATASET_BASE_PATH=""
METADATA_PATH="/inspire/hdd/project/embodied-multimodality/tongjingqi-CZXS25110029/chj_code/wan_train.csv"

NUM_FRAMES=249     
HEIGHT=480
WIDTH=832

DATASET_REPEAT=1
LEARNING_RATE=1e-4
NUM_EPOCHS=3
LORA_RANK=64
GRADIENT_ACCUMULATION=1

OUTPUT_PATH="/inspire/hdd/project/embodied-multimodality/tongjingqi-CZXS25110029/chj_code/Wan2.2-TI2V-5B_lora_rank64"
SAVE_STEPS=250

DIFFSYNTH_PATH="/inspire/hdd/project/embodied-multimodality/tongjingqi-CZXS25110029/chj_code/DiffSynth-Studio"
TRAIN_SCRIPT="/inspire/hdd/project/embodied-multimodality/tongjingqi-CZXS25110029/chj_code/train.py"

cd ${DIFFSYNTH_PATH}

echo "训练配置 - Rank=64, Epochs=${NUM_EPOCHS}"

accelerate launch ${TRAIN_SCRIPT} \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${METADATA_PATH}" \
  --height ${HEIGHT} --width ${WIDTH} --num_frames ${NUM_FRAMES} \
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
