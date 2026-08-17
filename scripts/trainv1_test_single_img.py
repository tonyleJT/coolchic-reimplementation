import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from coolchic.data.image_io import load_image, save_image
from coolchic.models.codec import COOLCHICModel
from coolchic.training.trainerv1_test_single_img import train_single_image


'''
@brief Create the command-line argument parser for single-image training.

@return Configured argument parser.
'''

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overfit COOL-CHIC to one image."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input PNG image.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of optimization steps.",
    )

    parser.add_argument(
        "--lambda-rd",
        type=float,
        default=2e-4,
        help="Rate-distortion lambda.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Adam learning rate.",
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Progress printing interval.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="PyTorch random seed.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Training device, for example cuda or cpu.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output directory.",
    )

    return parser


'''
@brief Overfit COOL-CHIC to one real image and save the Stage 9 outputs.

@return None.
'''

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    image_path = Path(args.image)

    if args.output is None:
        output_dir = PROJECT_ROOT / "results" / image_path.stem
    else:
        output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    target = load_image(image_path).to(device)

    _, _, height, width = target.shape

    model = COOLCHICModel(
        height=height,
        width=width,
    ).to(device)

    print(
        "Stage 9 training | "
        f"image={image_path.name} | "
        f"shape={tuple(target.shape)} | "
        f"device={device} | "
        f"steps={args.steps} | "
        f"lambda={args.lambda_rd:g} | "
        f"lr={args.lr:g}"
    )

    final_output = train_single_image(
        model=model,
        target=target,
        lambda_rd=args.lambda_rd,
        steps=args.steps,
        learning_rate=args.lr,
        log_every=args.log_every,
    )

    reconstruction_path = output_dir / "reconstruction.png"
    checkpoint_path = output_dir / "model.pt"
    metrics_path = output_dir / "metrics.json"

    save_image(
        final_output["reconstruction"].detach().cpu(),
        reconstruction_path,
    )

    checkpoint = {
        "model_state_dict": {
            name: parameter.detach().cpu()
            for name, parameter in model.state_dict().items()
        },
        "height": height,
        "width": width,
        "lambda_rd": args.lambda_rd,
        "steps": args.steps,
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    metrics = {
        "image": image_path.name,
        "height": height,
        "width": width,
        "steps": args.steps,
        "lambda": args.lambda_rd,
        "learning_rate": args.lr,
        "mse": final_output["mse"].item(),
        "psnr": final_output["psnr"].item(),
        "latent_bits": final_output["latent_bits"].item(),
        "latent_bpp": final_output["latent_bpp"].item(),
        "evaluation": "rounded_latents_full_precision_mlps",
        "rate_scope": "estimated_latent_rate_only",
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    print(
        "Stage 9 saved | "
        f"reconstruction={reconstruction_path} | "
        f"checkpoint={checkpoint_path} | "
        f"metrics={metrics_path}"
    )


if __name__ == "__main__":
    main()