#!/usr/bin/env python3
"""
将 jsonl 格式的数据转换为 DiffSynth-Studio 训练所需的 csv 格式

数据处理规则：
- dataset_maze/ 开头的视频：prompt 保持原样
- dataset_eyeballing/ 开头的视频：删除 prompt 后两句

输入格式 (jsonl):
{"video": "/path/to/video.mp4", "prompt": "xxx"}

输出格式 (csv):
video,text
/path/to/video.mp4,xxx
"""

import argparse
import csv
import json
import os
from pathlib import Path


def process_prompt(video_path: str, prompt: str) -> str:
    """
    根据视频路径处理 prompt
    - dataset_maze/: 保持原样
    - dataset_eyeballing/: 删除后两句
    """
    if "dataset_eyeballing" in video_path:
        # 按句号分割，删除后两句
        sentences = prompt.split("。")
        # 过滤掉空字符串
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) > 2:
            # 保留前 n-2 句，重新用句号连接
            processed = "。".join(sentences[:-2])
            if processed and not processed.endswith("。"):
                processed += "。"
            return processed
        else:
            # 如果只有 2 句或更少，返回空字符串或原样（根据需求调整）
            return prompt
    else:
        # dataset_maze 或其他：保持原样
        return prompt


def convert_jsonl_to_csv(input_path: str, output_path: str, verbose: bool = True):
    """将 jsonl 文件转换为 csv 文件"""
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    # 统计信息
    total_count = 0
    maze_count = 0
    eyeballing_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8', newline='') as f_out:
        
        writer = csv.writer(f_out)
        # 写入表头
        writer.writerow(['video', 'text'])
        
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行 JSON 解析失败: {e}")
                continue
            
            video_path = data.get('video', '')
            prompt = data.get('prompt', '')
            
            if not video_path:
                print(f"警告: 第 {line_num} 行缺少 video 字段")
                continue
            
            # 处理 prompt
            processed_prompt = process_prompt(video_path, prompt)
            
            # 写入 csv
            writer.writerow([video_path, processed_prompt])
            total_count += 1
            
            # 统计
            if "dataset_maze" in video_path:
                maze_count += 1
            elif "dataset_eyeballing" in video_path:
                eyeballing_count += 1
    
    if verbose:
        print(f"转换完成!")
        print(f"  - 总数据条数: {total_count}")
        print(f"  - maze 数据: {maze_count}")
        print(f"  - eyeballing 数据: {eyeballing_count}")
        print(f"  - 输出文件: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="将 jsonl 转换为 DiffSynth-Studio csv 格式")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="/inspire/hdd/project/embodied-multimodality/public/VLMPuzzle/dataset/combined_dataset.jsonl",
        help="输入 jsonl 文件路径"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="/inspire/hdd/project/embodied-multimodality/tongjingqi-CZXS25110029/chj_code/Wan2.2-Train/wan_train.csv",
        help="输出 csv 文件路径"
    )
    
    args = parser.parse_args()
    
    convert_jsonl_to_csv(args.input, args.output)


if __name__ == "__main__":
    main()
