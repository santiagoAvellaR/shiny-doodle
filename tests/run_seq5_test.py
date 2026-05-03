from pathlib import Path
from src.pipelines.pipeline_seq5 import run_seq5

if __name__ == "__main__":
    run_seq5(
        input_video=Path("data/raw/seq5.mp4"),
        overlay_image=Path("data/assets/jaguar.jpg"),
        output_video=Path("results/videos/seq5_final.mp4"),
        max_frames=155,
        display=True
    )