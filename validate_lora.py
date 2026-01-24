import torch, os, argparse
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.core.data.operators import ToAbsolutePath, LoadVideo, ImageCropAndResize

# Global Offline Config
MODEL_BASE_PATH = "/inspire/hdd/project/embodied-multimodality/public/downloaded_ckpts/Wan2.2-TI2V-5B"
TOKENIZER_PATH = f"{MODEL_BASE_PATH}/google/umt5-xxl"

def parse_args():
    parser = argparse.ArgumentParser(description="Wan2.2 LoRA Validation Script")
    parser.add_argument("--lora_ckpt", type=str, required=True, help="Path to the LoRA checkpoint (.safetensors)")
    parser.add_argument("--prompt", type=str, default="A cinematic shot of a cat boxing", help="Prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走", help="Negative prompt")
    parser.add_argument("--input_image", type=str, default=None, help="Path to input image/video for I2V")
    parser.add_argument("--output", type=str, default="validation_output.mp4", help="Output filename")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--model_base_path", type=str, default=MODEL_BASE_PATH, help="Base path for Wan2.2 models")
    parser.add_argument("--tokenizer_path", type=str, default=TOKENIZER_PATH, help="Base path for Tokenizer")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA alpha/scale")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--width", type=int, default=832, help="Width")
    parser.add_argument("--height", type=int, default=480, help="Height")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading base model from: {args.model_base_path}")
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    
    # Initialize Pipeline with Offline Paths
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern=os.path.join(args.model_base_path, "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern=os.path.join(args.model_base_path, "diffusion_pytorch_model*.safetensors")),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern=os.path.join(args.model_base_path, "Wan2.2_VAE.pth")),
        ],
        tokenizer_config=ModelConfig(args.tokenizer_path),
        # Ensure no download for audio processor
        audio_processor_config=None, 
    )
    
    # Load LoRA
    print(f"Loading LoRA from: {args.lora_ckpt} (alpha={args.lora_alpha})")
    pipe.load_lora(pipe.dit, args.lora_ckpt, alpha=args.lora_alpha)
    
    # Prepare Input
    input_image = None
    if args.input_image:
        print(f"Loading input image from: {args.input_image}")
        # Use DiffSynth operators or PIL
        # Example uses VideoData for video input, let's support image too
        if args.input_image.endswith(('.mp4', '.avi', '.mov')):
             # Load video first frame? Or use it as video-to-video?
             # Wan2.2-TI2V is Text-Image-to-Video. Usually takes first frame.
             # DiffSynth example uses VideoData[0] logic.
             from diffsynth.utils.data import VideoData
             input_image = VideoData(args.input_image, height=args.height, width=args.width)[0] # Get first frame
        else:
             input_image = Image.open(args.input_image).convert("RGB")
             input_image = input_image.resize((args.width, args.height))

    # Generate
    print("Generating video...")
    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_image=input_image,
        num_frames=args.num_frames,
        seed=1, 
        tiled=True,
        width=args.width,
        height=args.height,
    )
    
    # Save
    print(f"Saving to {args.output}")
    save_video(video, args.output, fps=15, quality=5)
    print("Done!")

if __name__ == "__main__":
    main()
