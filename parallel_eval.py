
import os
import glob
import json
import torch
import argparse
import logging
import multiprocessing
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video

# ================= VLMPuzzle Import Setup =================
import sys
# Server path to VLMPuzzle
VLMPUZZLE_PATH = "/inspire/hdd/project/embodied-multimodality/tongjingqi-CZXS25110029/chj_code/VLMPuzzle"
sys.path.append(VLMPUZZLE_PATH)
try:
    from puzzle.maze_square.evaluator import MazeEvaluator
except ImportError:
    # Fallback or try adding subdirectory if needed, but sys.path.append should work
    pass

# ================= Configuration =================
MODEL_BASE_PATH = "/inspire/hdd/project/embodied-multimodality/public/downloaded_ckpts/Wan2.2-TI2V-5B"
TOKENIZER_PATH = f"{MODEL_BASE_PATH}/google/umt5-xxl"
DEFAULT_PROMPT = "Draw a red path connecting two red dots without touching the black walls. Static camera."

# Global worker variables
pipe = None
evaluator = None

def setup_logger(output_dir):
    logging.basicConfig(
        filename=os.path.join(output_dir, "evaluation.log"),
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

def init_worker(gpu_id, model_base_path, tokenizer_path, lora_ckpt, metadata_path):
    global pipe, evaluator
    device = f"cuda:{gpu_id}"
    print(f"[Worker {gpu_id}] Initializing on {device}...")

    # 1. Initialize Pipeline
    dit_files = sorted(glob.glob(os.path.join(model_base_path, "diffusion_pytorch_model*.safetensors")))
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=os.path.join(model_base_path, "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(path=dit_files),
            ModelConfig(path=os.path.join(model_base_path, "Wan2.2_VAE.pth")),
        ],
        tokenizer_config=ModelConfig(path=tokenizer_path),
        audio_processor_config=None,
    )

    # 2. Load LoRA
    if lora_ckpt and os.path.exists(lora_ckpt):
        print(f"[Worker {gpu_id}] Loading LoRA: {lora_ckpt}")
        pipe.load_lora(pipe.dit, lora_ckpt, alpha=1.0)
    
    # 3. Initialize Evaluator
    if metadata_path and os.path.exists(metadata_path):
        try:
           # MazeEvaluator needs base_dir to find/save things if needed, but mainly metadata.
           # It inherits from MazePuzzleEvaluator.
           evaluator = MazeEvaluator(metadata_path, base_dir=os.path.dirname(metadata_path))
        except Exception as e:
           print(f"[Worker {gpu_id}] Warning: Failed to init evaluator: {e}")
           evaluator = None

def process_item(args):
    """
    Args:
        puzzle_path: Path to input image
        output_dir: Directory to save results
        prompt: Text prompt
        gpu_id: Assigned GPU
    """
    puzzle_path, output_dir, prompt, width, height, num_frames = args
    global pipe, evaluator
    
    puzzle_id = os.path.basename(puzzle_path).replace("_puzzle.png", "")
    video_filename = f"{puzzle_id}_solution.mp4"
    video_path = os.path.join(output_dir, video_filename)
    frame_path = os.path.join(output_dir, f"{puzzle_id}_last_frame.png")
    
    # 1. Inference
    try:
        input_image = Image.open(puzzle_path).convert("RGB")
        input_image = input_image.resize((width, height))
        
        video = pipe(
            prompt=prompt,
            negative_prompt="",
            input_image=input_image,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=42,
            tiled=True
        )
        
        save_video(video, video_path, fps=15, quality=5)
        
        # 2. Extract Last Frame for Evaluation
        # WanVideoPipeline output via save_video logic returns a list of PIL Images (T frames).
        # So video is [PIL_0, PIL_1, ..., PIL_N].
        last_frame = video[-1] 
        last_frame.save(frame_path)
        
        # 3. Evaluate
        result_dict = {}
        if evaluator:
            try:
                # MazeEvaluator.evaluate(record_id, candidate_path)
                result = evaluator.evaluate(puzzle_id, frame_path)
                # result is MazeEvaluationResult object
                result_dict = {
                    'status': 'success',
                    'puzzle_id': puzzle_id,
                    'connected': result.connected,
                    'hit_wall': result.hit_wall,
                    'is_valid': result.is_valid
                }
            except Exception as e:
                result_dict = {'status': 'eval_error', 'message': str(e), 'puzzle_id': puzzle_id}
        else:
             result_dict = {'status': 'no_evaluator', 'puzzle_id': puzzle_id}

        return result_dict

    except Exception as e:
        print(f"Error processing {puzzle_id}: {e}")
        return {'status': 'error', 'message': str(e), 'puzzle_id': puzzle_id}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing _puzzle.png files")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--lora_ckpt", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7", help="Comma separated GPU IDs")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=896)
    parser.add_argument("--num_frames", type=int, default=81)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logger(args.output_dir)
    
    # Locate Data
    puzzle_files = sorted(glob.glob(os.path.join(args.input_dir, "*_puzzle.png")))
    if not puzzle_files:
        print("No puzzles found!")
        return
        
    print(f"Found {len(puzzle_files)} puzzles.")
    
    # Locate Metadata (Assume data.json is in the parent of input_dir)
    # input_dir: .../mazes/maze_square/puzzles
    # metadata: .../mazes/maze_square/data.json
    parent_dir = os.path.dirname(os.path.normpath(args.input_dir))
    metadata_path = os.path.join(parent_dir, "data.json")
    if not os.path.exists(metadata_path):
        print(f"Warning: Metadata not found at {metadata_path}. Evaluation will be skipped (only generation).")
        metadata_path = None
    else:
        print(f"Using metadata: {metadata_path}")
    
    # Prepare Workers
    gpu_list = [int(x) for x in args.gpu_ids.split(",")]
    num_gpus = len(gpu_list)
    
    # Manual Process spawning for exact GPU control
    ctx = multiprocessing.get_context('spawn')
    
    # Manual Process spawning for exact GPU control
    chunk_size = int(np.ceil(len(puzzle_files) / num_gpus))
    chunks = [puzzle_files[i:i + chunk_size] for i in range(0, len(puzzle_files), chunk_size)]
    
    processes = []
    results_queue = ctx.Queue()
    
    for i, gpu_id in enumerate(gpu_list):
        if i >= len(chunks): break
        chunk = chunks[i]
        p = ctx.Process(
            target=worker_process,
            args=(gpu_id, chunk, args, MODEL_BASE_PATH, TOKENIZER_PATH, metadata_path, results_queue)
        )
        p.start()
        processes.append(p)
        print(f"Started worker for GPU {gpu_id} with {len(chunk)} tasks.")
    
    # Collect results
    results = []
    total_tasks = len(puzzle_files)
    with tqdm(total=total_tasks, desc="Evaluating") as pbar:
        for _ in range(total_tasks):
            res = results_queue.get()
            results.append(res)
            pbar.update(1)
            
            if res.get('status') == 'success':
                msg = f"Puzzle {res['puzzle_id']}: Connected={res.get('connected')}"
                logging.info(msg)
            else:
                logging.error(f"Failed {res.get('puzzle_id')}: {res.get('message')}")

    for p in processes:
        p.join()
        
    # Calculate Stats
    process_stats(results)

def worker_process(gpu_id, puzzle_paths, args, model_base, tokenizer_path, metadata_path, result_queue):
    init_worker(gpu_id, model_base, tokenizer_path, args.lora_ckpt, metadata_path)
    
    for puzzle_path in puzzle_paths:
        task_args = (puzzle_path, args.output_dir, args.prompt, args.width, args.height, args.num_frames)
        res = process_item(task_args)
        result_queue.put(res)

def process_stats(results):
    total = len(results)
    success = 0
    connected = 0
    
    for r in results:
        if r.get('status') == 'success':
            success += 1
            if r.get('connected'):
                connected += 1
                
    accuracy = (connected / total) * 100 if total > 0 else 0
    print("\n" + "="*40)
    print(f"Total Puzzles: {total}")
    print(f"Successful Gens: {success}")
    print(f"Connected Paths: {connected}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("="*40)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
