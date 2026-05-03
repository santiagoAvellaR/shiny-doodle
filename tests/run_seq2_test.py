from pathlib import Path
from src.pipelines.pipeline_seq2 import run_seq2

if __name__ == "__main__":
    run_seq2(
        input_video=Path("data/raw/seq2.mp4"),        
        overlay_image=Path("data/assets/jaguar.jpg"),      
        output_video=Path("results/videos/seq2_final.mp4"), 
        max_frames=155,                               
        display=True                                  
    )