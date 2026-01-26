import torch, os, argparse
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video, VideoData
from PIL import Image

# Default Offline Paths
MODEL_BASE_PATH = "/inspire/hdd/project/embodied-multimodality/public/downloaded_ckpts/Wan2.2-TI2V-5B"
TOKENIZER_PATH = f"{MODEL_BASE_PATH}/google/umt5-xxl"

def parse_args():
    parser = argparse.ArgumentParser(description="Wan2.2 LoRA Inference Script")
    parser.add_argument("--lora_ckpt", type=str, required=True, help="Path to LoRA checkpoint (.safetensors)")
    parser.add_argument("--prompt", type=str, default="A majestic lion standing on a rock", help="Text prompt")
    parser.add_argument("--negative_prompt", type=str, default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走", help="Negative prompt")
    parser.add_argument("--input_image", type=str, default=None, help="Input image path (optional, for I2V)")
    parser.add_argument("--height", type=int, default=896, help="Video height (must be multiple of 32)")
    parser.add_argument("--width", type=int, default=480, help="Video width (must be multiple of 32)")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames ((n-1)%4==0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output filename")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    # Advanced paths
    parser.add_argument("--model_base_path", type=str, default=MODEL_BASE_PATH, help="Base model path")
    parser.add_argument("--tokenizer_path", type=str, default=TOKENIZER_PATH, help="Tokenizer path")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Initializing Pipeline...")
    print(f"  Model Base: {args.model_base_path}")
    print(f"  Tokenizer: {args.tokenizer_path}")
    
    # 1. Initialize Pipeline (Offline Mode)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern=os.path.join(args.model_base_path, "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern=os.path.join(args.model_base_path, "diffusion_pytorch_model*.safetensors")),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern=os.path.join(args.model_base_path, "Wan2.2_VAE.pth")),
        ],
        tokenizer_config=ModelConfig(args.tokenizer_path),
        audio_processor_config=None, # Disable audio processor download
    )
    
    # 2. Load LoRA
    if args.lora_ckpt:
        print(f"Loading LoRA from: {args.lora_ckpt}")
        pipe.load_lora(pipe.dit, args.lora_ckpt, alpha=1.0)
    
    # 3. Prepare Inputs
    input_image = None
    if args.input_image:
        print(f"Loading input image: {args.input_image}")
        # Wan2.2-TI2V usually takes a PIL image as conditioning
        input_image = Image.open(args.input_image).convert("RGB")
        # Resize to target resolution to avoid internal resizing mismatch
        input_image = input_image.resize((args.width, args.height))
    
    # 4. Generate
    print(f"Generating video...")
    print(f"  Prompt: {args.prompt}")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Frames: {args.num_frames}")
    
    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_image=input_image, 
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        seed=args.seed,
        tiled=True, # Enable tiled decoding for memory efficiency
    )
    
    # 5. Save
    print(f"Saving to {args.output}")
    save_video(video, args.output, fps=15, quality=5)
    print("Done!")

if __name__ == "__main__":
    main()
