from __future__ import annotations

import argparse
from pathlib import Path

import torch

from coolchic.data.image_io import save_image
from coolchic.models.codec import COOLCHICModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct an image from a saved COOL-CHIC research checkpoint."
    )
    parser.add_argument("checkpoint", type=Path, help="Path to model.pt")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <checkpoint-dir>/decoded.png.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used for reconstruction.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = COOLCHICModel(
        height=int(checkpoint["height"]),
        width=int(checkpoint["width"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        dense_latents = model.latents.quantized_forward()
        reconstruction = model.synthesis(dense_latents).clamp(0.0, 1.0)

    output_path = args.output or args.checkpoint.parent / "decoded.png"
    save_image(reconstruction.cpu(), output_path)

    print(f"decoded | checkpoint={args.checkpoint} | output={output_path} | device={device}")


if __name__ == "__main__":
    main()
