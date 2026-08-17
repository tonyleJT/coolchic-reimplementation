import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(PROJECT_ROOT / "src"),
)

from coolchic.data import load_image, save_image
from coolchic.models.codec import COOLCHICModel
from coolchic.training.reproducibility import set_seed
from coolchic.training.trainer import (
    quantize_mlp_parameters,
    train_single_image,
)


'''
@brief Parse command-line arguments for single-image COOL-CHIC optimization.

@return Parsed command-line arguments.
'''
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overfit COOL-CHIC on one image "
            "and estimate its final bitrate."
        )
    )

    parser.add_argument(
        "image",
        type=str,
        help="Input PNG image.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--lambda-rd",
        type=float,
        default=2.0e-4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1.0e-2,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory. "
            "Default: results/<image-stem>."
        ),
    )

    return parser.parse_args()


'''
@brief Train COOL-CHIC, quantize its MLPs, and save final Stage 10 artifacts.

@return None.
'''
def main() -> None:
    args = parse_args()

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    image_path = Path(
        args.image
    )

    image = load_image(
        image_path
    ).to(
        device
    )

    _, _, height, width = image.shape

    model = COOLCHICModel(
        height,
        width,
    ).to(
        device
    )

    print(
        "Stage 10 start | "
        f"device={device} | "
        f"image={image_path.name} | "
        f"shape={tuple(image.shape)} | "
        f"steps={args.steps} | "
        f"lambda={args.lambda_rd:g}"
    )

    # ---------------------------------------------------------
    # Stage 9: optimize one image.
    # ---------------------------------------------------------

    train_single_image(
        model=model,
        image=image,
        steps=args.steps,
        learning_rate=args.lr,
        lambda_rd=args.lambda_rd,
        log_every=args.log_every,
    )

    # ---------------------------------------------------------
    # Stage 10: quantize both MLPs and calculate final rate.
    # ---------------------------------------------------------

    final_output = quantize_mlp_parameters(
        model=model,
        image=image,
        lambda_rd=args.lambda_rd,
    )

    if args.output_dir is None:
        output_dir = (
            PROJECT_ROOT
            / "results"
            / image_path.stem
        )
    else:
        output_dir = Path(
            args.output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reconstruction_path = (
        output_dir
        / "reconstruction.png"
    )

    checkpoint_path = (
        output_dir
        / "model.pt"
    )

    metrics_path = (
        output_dir
        / "metrics.json"
    )

    # This reconstruction is now the real Stage 10 result:
    # rounded latents + quantized MLP parameters.
    save_image(
        final_output["reconstruction"],
        reconstruction_path,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "height": height,
        "width": width,
        "lambda_rd": args.lambda_rd,
        "steps": args.steps,
        "theta_quantization_step": (
            final_output[
                "theta_quantization_step"
            ]
        ),
        "psi_quantization_step": (
            final_output[
                "psi_quantization_step"
            ]
        ),
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

        "loss": (
            final_output["loss"].item()
        ),

        "mse": (
            final_output["mse"].item()
        ),

        "psnr": (
            final_output["psnr"].item()
        ),

        "latent_bits": (
            final_output["latent_bits"].item()
        ),

        "latent_bpp": (
            final_output["latent_bpp"].item()
        ),

        "theta_bits": (
            final_output["theta_bits"].item()
        ),

        "psi_bits": (
            final_output["psi_bits"].item()
        ),

        "mlp_bits": (
            final_output["mlp_bits"].item()
        ),

        "total_estimated_bits": (
            final_output[
                "total_estimated_bits"
            ].item()
        ),

        "estimated_bpp": (
            final_output[
                "estimated_bpp"
            ].item()
        ),

        "theta_quantization_step": (
            final_output[
                "theta_quantization_step"
            ]
        ),

        "psi_quantization_step": (
            final_output[
                "psi_quantization_step"
            ]
        ),

        "evaluation": (
            "rounded latents + "
            "quantized synthesis/probability MLPs"
        ),

        "rate": (
            "entropy-model-estimated bitrate; "
            "no real entropy-coded bitstream"
        ),
    }

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print(
        "Stage 10 saved | "
        f"reconstruction={reconstruction_path} | "
        f"checkpoint={checkpoint_path} | "
        f"metrics={metrics_path}"
    )


if __name__ == "__main__":
    main()