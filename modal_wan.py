import modal, subprocess, os
from pathlib import Path

app = modal.App("wan2-minimal")

HF_REPO = "Wan-AI/Wan2.2-S2V-14B"
MODEL_DIR = "/models/Wan2.2-S2V-14B"
WAN_REPO_DIR = "/wan22"
CACHE_VOL = modal.Volume.from_name("wan22-shelly", create_if_missing=True)

# Hard-coded inputs
PROMPT = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard"
REF_IMAGE = "/wan22/examples/i2v_input.JPG"
AUDIO = "/wan22/examples/talk.wav"
OUT_PATH = "/models/out.mp4"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .uv_pip_install(
        "torch>=2.4.0",
        "torchvision>=0.19.0", 
        "torchaudio",
        "numpy>=1.23.5,<2"
    )
    .uv_pip_install(
        "flash_attn"
    )
    .uv_pip_install(
        "opencv-python>=4.9.0.80",
        "diffusers>=0.31.0",
        "transformers>=4.49.0,<=4.51.3",
        "tokenizers>=0.20.3",
        "accelerate>=1.1.1",
        "tqdm",
        "imageio[ffmpeg]",
        "easydict",
        "ftfy",
        "dashscope",
        "imageio-ffmpeg"
    )
    .run_commands("git clone https://github.com/Wan-Video/Wan2.2.git /wan22")
)

@app.function(
    image=image,
    gpu="H200",
    volumes={"/models": CACHE_VOL},
    timeout=60 * 60,
)
def generate_video():
    """Generate video using Wan2.2 model with direct repository approach."""
    import subprocess
    # Generate video using the Wan2.2 generate.py script
    print(f"Generating video with prompt: {PROMPT}")

    generate_cmd = [
        "python", f"{WAN_REPO_DIR}/generate.py",
        "--task", "s2v-14B",
        "--size", "1024*704",
        "--ckpt_dir", MODEL_DIR,
        "--offload_model", "True",
        "--convert_model_dtype",
        "--prompt", PROMPT,
        "--image", REF_IMAGE,
        "--audio", AUDIO
    ]

    # Change to Wan2.2 directory and run generation
    result = subprocess.run(
        generate_cmd,
        cwd=WAN_REPO_DIR,
        capture_output=True,
        text=True,
        check=True
    )

    print("Generation output:")
    print(result.stdout)
    if result.stderr:
        print("Errors/Warnings:")
        print(result.stderr)

    # Find the generated video file (Wan2.2 typically saves to a results directory)
    results_dir = Path(WAN_REPO_DIR) / "results"
    if results_dir.exists():
        video_files = list(results_dir.glob("*.mp4"))
        if video_files:
            generated_video = video_files[-1]  # Get the most recent
            # Copy to our output path
            subprocess.run(["cp", str(generated_video), OUT_PATH], check=True)
            print(f"Video saved to: {OUT_PATH}")
            return OUT_PATH

    print("Warning: No output video found in results directory")
    return None


if __name__ == "__main__":
    with app.run():
        result = generate_video.remote()
        print(f"Video generated at: {result}")


