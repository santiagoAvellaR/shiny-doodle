from pathlib import Path
from src.pipelines.pipeline_seq1 import run_seq1

if __name__ == "__main__":
    run_seq1(
        input_video=Path("data/raw/seq1.mp4"),
        overlay_image=Path("data/raw/pika.png"), # Wait, let me check the real pika image path
        output_video=Path("results/seq1_robust_test.mp4"),
        max_frames=155,
        display=False
    )
