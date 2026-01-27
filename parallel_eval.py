
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
from puzzle.maze_base import MazePuzzleEvaluator, MazeEvaluationResult

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
    # Evaluator needs to know where puzzles are relative to metadata? 
    # Actually MazePuzzleEvaluator usually takes metadata_path.
    # We pass base_dir as the parent of puzzles dir to help it resolve relative paths if needed.
    # Assuming metadata.json is at metadata_path
    if metadata_path and os.path.exists(metadata_path):
        # Create a dummy class or use properly if metadata is loaded per call?
        # MazePuzzleEvaluator loads metadata in __init__
        try:
           evaluator = MazePuzzleEvaluator(metadata_path, base_dir=os.path.dirname(metadata_path))
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
        last_frame = video[0][-1] # [C, F, H, W] -> video[0] gives list/tensor? 
        # DiffSynth save_video handles list of tensors or PIL images.
        # video is typically a list of PIL Images if not raw tensor.
        # Check WanVideoPipeline output: usually list of PIL Images per prompt.
        # pipe() returns 'video' which is usually a list of VideoData or similar?
        # Let's assume standard DiffSynth output.
        # Actually save_video source: def save_video(video_data, ...):
        # If it's a list containing a list of PIL images (batch size 1).
        
        # WanVideoPipeline output inspection from inference.py:
        # video = pipe(...) -> save_video(video, ...)
        
        # Let's extract the last frame.
        # If video is list of list of PIL:
        pil_last_frame = video[0][-1]
        pil_last_frame.save(frame_path)
        
        # 3. Evaluate
        result_dict = {}
        if evaluator:
            # We need to make sure the evaluator can find the original record?
            # Evaluator looks up record by puzzle_id in metadata.
            # candidate needs to be the path to the extracted frame.
            try:
                result = evaluator.evaluate(puzzle_id, frame_path)
                result_dict = result.to_dict()
                result_dict['status'] = 'success'
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
    
    # Multiprocessing Setup
    ctx = multiprocessing.get_context('spawn')
    pool = ctx.Pool(
        processes=num_gpus,
        initializer=init_worker,
        initargs=(None, MODEL_BASE_PATH, TOKENIZER_PATH, args.lora_ckpt, metadata_path)
    )
    
    # Note: init_worker inside Pool doesn't easily allow passing different GPU IDs to different workers
    # standard Pool doesn't support 'worker_id'. 
    # Trick: A queue or managing process ID.
    # Better: Use a custom worker function that grabs a GPU ID from a Queue?
    # Or just spawn N processes manually without Pool if needed?
    # Actually, we can assume process rank? 
    # Simpler: Split tasks into N chunks, run each chunk in a separate Process that initializes one GPU.
    
    pool.close() # Cancel the previous pool attempt
    
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
