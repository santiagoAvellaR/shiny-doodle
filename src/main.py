from __future__ import annotations

import argparse
from pathlib import Path

from src.pipelines.pipeline_seq1 import run_seq1
from src.pipelines.pipeline_seq3 import run_seq3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SY32 - Incrustation planaire par homographie"
    )
    parser.add_argument(
        "--seq",
        type=int,
        required=True,
        help="Numéro de séquence à traiter (pour l'instant: 1).",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/seq1.mp4",
        help="Chemin vers la vidéo d'entrée.",
    )
    parser.add_argument(
        "--overlay",
        type=str,
        default="data/assets/jaguar.jpg",
        help="Chemin vers l'image à incruster.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/videos/seq1_result.mp4",
        help="Chemin vers la vidéo de sortie.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Afficher la vidéo pendant le traitement.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limiter le nombre d'images traitées (utile pour les tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_video = Path(args.input)
    overlay_image = Path(args.overlay)
    output_video = Path(args.output)

    output_video.parent.mkdir(parents=True, exist_ok=True)

    if args.seq == 1:
        run_seq1(
            input_video=input_video,
            overlay_image=overlay_image,
            output_video=output_video,
            display=args.display,
            max_frames=args.max_frames,
        )
    elif args.seq == 3:
        run_seq3(
            input_video=input_video,
            overlay_image=overlay_image,
            output_video=output_video,
            display=args.display,
            max_frames=args.max_frames,
        )
    else:
        raise NotImplementedError(
            f"La séquence {args.seq} n'est pas encore implémentée."
        )


if __name__ == "__main__":
    main()