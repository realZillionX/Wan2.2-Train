# Wan2.2-TI2V-5B LoRA Training

本仓库包含 **Wan2.2-TI2V-5B** 模型的 LoRA 微调训练脚本，支持单机和多机分布式训练。

## 目录结构

```
├── rank_16/    # LORA_RANK=16
├── rank_32/    # LORA_RANK=32
├── rank_64/    # LORA_RANK=64
```

每个文件夹包含：
- `wan_lora.sh` - 单机训练脚本
- `wan_lora_multi_node.sh` - 多机分布式训练 (2节点×8卡)
- `accelerate_config_multi_node.yaml` - DeepSpeed ZeRO-2 配置

## 统一训练配置

| 参数 | 值 |
|------|-----|
| NUM_EPOCHS | 3 |
| SAVE_STEPS | 250 |
| 多节点配置 | 2节点 × 8 GPU (16进程) |
| 学习率 | 1e-4 |
| 视频帧数 | 249 |
| 分辨率 | 480×832 |

## 使用方法

### 多节点分布式训练 (推荐)

```bash
cd rank_32
bash wan_lora_multi_node.sh
```

### 单机训练

```bash
cd rank_32
bash wan_lora.sh
```

## 依赖

- DiffSynth-Studio (离线模式)
- DeepSpeed
- Accelerate

## 输出路径

训练产出保存在：
```
/inspire/hdd/project/embodied-multimodality/tongjingqi-CZXS25110029/chj_code/Wan2.2-TI2V-5B_lora_rank{16,32,64}[_multi]
```
