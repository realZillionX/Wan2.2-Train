# Wan2.2-TI2V-5B LoRA Training

本仓库包含 **Wan2.2-TI2V-5B** 模型的 LoRA 微调训练脚本 (rank_32)。

## 目录结构

```
├── rank_32/    # LORA_RANK=32
│   ├── wan_lora_multi_node.sh  # 5节点启动脚本
│   ├── wan_lora.sh             # 单机启动脚本
│   └── accelerate_config_multi_node.yaml
└── train.py    # 训练主程序 (支持自动续训)
```

## 数据集

脚本启动时会自动合并以下三个 CSV 文件到项目根目录的 `wan_train.csv`：
1. `/inspire/hdd/project/embodied-multimodality/public/hjchen/MazeSquare/part_7x7.csv`
2. `/inspire/hdd/project/embodied-multimodality/public/hjchen/MazeSquare/part_9x9.csv`
3. `/inspire/hdd/project/embodied-multimodality/public/hjchen/MazeSquare/part_11x11.csv`

## 训练配置

| 参数 | 值 |
|------|-----|
| NUM_EPOCHS | 3 |
| SAVE_STEPS | 250 |
| 多节点配置 | 5节点 × 8 GPU (40进程) |
| 学习率 | 1e-4 |
| **视频帧数** | **223** |
| **分辨率** | **480×873** (Width × Height, 竖屏) |
| LoRA Rank | 32 |

## 使用方法

### 5 节点分布式训练

```bash
cd rank_32
bash wan_lora_multi_node.sh
```

**自动续训说明**：
如果训练中断，直接再次运行脚本即可。
- 脚本会自动检测 `output/rank32_multi` 目录下最新的 checkpoint（支持 step 或 epoch 格式）。
- 自动加载权重，并恢复训练进度条（Epoch/Steps 计数）。
- 输出文件会接着上次的编号继续保存，不会覆盖。

## 依赖

- DiffSynth-Studio (离线模式)
- DeepSpeed
- Accelerate
- Pandas (用于合并 CSV)
