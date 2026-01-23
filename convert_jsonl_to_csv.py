#!/usr/bin/env python3
"""
将 jsonl 格式的数据转换为 DiffSynth-Studio 训练所需的 csv 格式
同时对视频进行抽帧（30fps -> 6fps），保留首尾帧

数据处理规则：
- dataset_maze/ 开头的视频：prompt 保持原样
- dataset_eyeballing/ 开头的视频：删除 prompt 后两句

抽帧规则：
- 保留首帧和末帧
- 中间帧均匀采样
- 30fps -> 6fps（帧数降为约 1/5）

输入格式 (jsonl):
{"video": "/path/to/video.mp4", "prompt": "xxx"}

输出格式 (csv):
video,text
/path/to/new_video.mp4,xxx
"""

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def get_video_info(video_path: str) -> tuple:
    """获取视频的帧数和帧率"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-count_packets',
        '-show_entries', 'stream=nb_read_packets,r_frame_rate',
        '-of', 'csv=p=0',
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parts = result.stdout.strip().split(',')
        fps_parts = parts[0].split('/')
        fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
        frame_count = int(parts[1])
        return frame_count, fps
    except FileNotFoundError:
        print(f"错误: ffprobe 未安装或不在 PATH 中")
        return None, None
    except subprocess.CalledProcessError as e:
        print(f"错误: ffprobe 执行失败 {video_path}: {e.stderr}")
        return None, None
    except Exception as e:
        print(f"错误: 获取视频信息失败 {video_path}: {e}")
        return None, None


def calculate_frame_indices(total_frames: int, target_fps: float = 6, source_fps: float = 30) -> list:
    """
    计算需要保留的帧索引
    规则：保留首帧、末帧，中间帧均匀采样
    """
    if total_frames <= 2:
        return list(range(total_frames))
    
    duration_seconds = total_frames / source_fps
    target_frame_count = max(2, int(duration_seconds * target_fps))
    
    if target_frame_count >= total_frames:
        return list(range(total_frames))
    
    indices = [0]  # 首帧
    
    if target_frame_count > 2:
        middle_count = target_frame_count - 2
        for i in range(1, middle_count + 1):
            idx = int(1 + (total_frames - 2) * i / (middle_count + 1))
            indices.append(idx)
    
    indices.append(total_frames - 1)  # 末帧
    
    return sorted(set(indices))


def downsample_video(input_path: str, output_path: str, target_fps: float = 6) -> bool:
    """对单个视频进行抽帧"""
    frame_count, source_fps = get_video_info(input_path)
    if frame_count is None:
        return False
    
    frame_indices = calculate_frame_indices(frame_count, target_fps, source_fps)
    select_expr = '+'.join([f'eq(n\\,{idx})' for idx in frame_indices])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-i', input_path,
        '-vf', f"select='{select_expr}',setpts=N/({target_fps}*TB)",
        '-r', str(target_fps),
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '18',
        '-an',
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except FileNotFoundError:
        print(f"错误: ffmpeg 未安装或不在 PATH 中")
        return False
    except subprocess.CalledProcessError as e:
        print(f"错误: ffmpeg 执行失败 {input_path}: {e.stderr}")
        return False


def process_prompt(video_path: str, prompt: str) -> str:
    """根据视频路径处理 prompt"""
    if "dataset_eyeballing" in video_path:
        sentences = prompt.split("。")
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) > 2:
            processed = "。".join(sentences[:-2])
            if processed and not processed.endswith("。"):
                processed += "。"
            return processed
    return prompt


def process_single_item(args):
    """处理单个数据项（用于并行）"""
    data, output_video_dir, target_fps = args
    
    video_path = data.get('video', '')
    prompt = data.get('prompt', '')
    
    if not video_path:
        return {'error': '缺少 video 字段'}
    
    if not os.path.exists(video_path):
        return {'error': f'视频不存在: {video_path}'}
    
    # 构建输出路径（保持相对目录结构）
    # 从 video_path 中提取 dataset_xxx/xxx.mp4 部分
    parts = video_path.split('/')
    for i, p in enumerate(parts):
        if p.startswith('dataset_'):
            rel_path = '/'.join(parts[i:])
            break
    else:
        rel_path = os.path.basename(video_path)
    
    output_video_path = os.path.join(output_video_dir, rel_path)
    
    # 抽帧
    success = downsample_video(video_path, output_video_path, target_fps)
    
    if not success:
        return None
    
    # 处理 prompt
    processed_prompt = process_prompt(video_path, prompt)
    
    return {
        'video': output_video_path,
        'text': processed_prompt
    }


def convert_jsonl_to_csv(
    input_path: str,
    output_csv_path: str,
    output_video_dir: str,
    target_fps: float = 6,
    num_workers: int = 16
):
    """将 jsonl 文件转换为 csv 文件，同时进行视频抽帧"""
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    # 读取所有数据
    data_list = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data_list.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    print(f"读取 {len(data_list)} 条数据")
    print(f"目标帧率: {target_fps} fps")
    print(f"输出视频目录: {output_video_dir}")
    print(f"输出 CSV: {output_csv_path}")
    print()
    
    # 准备任务
    tasks = [(data, output_video_dir, target_fps) for data in data_list]
    
    # 并行处理
    results = []
    errors = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_item, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理视频"):
            result = future.result()
            if result:
                if 'error' in result:
                    errors.append(result['error'])
                else:
                    results.append(result)
    
    # 显示错误样本
    if errors:
        print(f"\n前 5 个错误示例:")
        for err in errors[:5]:
            print(f"  - {err}")
    
    # 写入 CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video', 'text'])
        for item in results:
            writer.writerow([item['video'], item['text']])
    
    print(f"\n转换完成!")
    print(f"  成功处理: {len(results)} 条")
    print(f"  失败/跳过: {len(data_list) - len(results)} 条")


def main():
    parser = argparse.ArgumentParser(description="将 jsonl 转换为 csv，同时抽帧视频 (30fps -> 6fps)")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="/inspire/hdd/project/embodied-multimodality/public/VLMPuzzle/dataset/combined_dataset.jsonl",
        help="输入 jsonl 文件路径"
    )
    parser.add_argument(
        "--output-csv", "-o",
        type=str,
        default="/inspire/hdd/project/embodied-multimodality/tongjingqi-CZXS25110029/chj_code/Wan2.2-Train/wan_train.csv",
        help="输出 csv 文件路径"
    )
    parser.add_argument(
        "--output-video-dir", "-v",
        type=str,
        default="/inspire/hdd/project/embodied-multimodality/public/hjchen/VLMPuzzle/",
        help="抽帧后视频输出目录"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=6,
        help="目标帧率 (默认: 6)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="并行工作进程数 (默认: 16)"
    )
    
    args = parser.parse_args()
    
    convert_jsonl_to_csv(
        input_path=args.input,
        output_csv_path=args.output_csv,
        output_video_dir=args.output_video_dir,
        target_fps=args.fps,
        num_workers=args.workers
    )


if __name__ == "__main__":
    main()
