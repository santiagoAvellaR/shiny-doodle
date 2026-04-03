# SY32 - Planar Image Insertion by Homography

Computer vision project for inserting an image onto a planar surface in a video sequence using homographies.

## Project structure

    sy32-homography-project/
    ├─ data/
    │  ├─ raw/
    │  └─ assets/
    ├─ results/
    ├─ src/
    │  ├─ main.py
    │  ├─ geometry/
    │  └─ pipelines/
    ├─ environment.yml
    └─ README.md

## Create the environment with Conda

Create the environment from the `environment.yml` file:

    conda env create -f environment.yml

Activate the environment:

    conda activate sy32-project-planar-overlay

## Run the project

Example for sequence 1:

    python -m src.main --seq 1 --input data/raw/seq1.mp4 --overlay data/assets/jaguar.jpg --output results/videos/seq1_result.mp4 --display

## Arguments

- `--seq`: sequence number to process
- `--input`: path to the input video
- `--overlay`: image to insert
- `--output`: path to the output video
- `--display`: shows the result while processing

## Output

The generated video is saved in the `results/videos/` folder.

## Notes

- The project uses OpenCV and NumPy.
- Sequence 1 is the first implemented baseline.