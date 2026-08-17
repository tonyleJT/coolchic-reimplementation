import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


'''
# /**
#  * @brief Parse command-line arguments for the RD plotting script.
#  *
#  * @return Parsed command-line arguments.
#  */
'''

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the averaged Kodak COOL-CHIC rate-distortion curve."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/kodak_summary.csv"),
        help="Input Kodak summary CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/kodak_rd_curve.png"),
        help="Output rate-distortion PNG.",
    )
    return parser.parse_args()


'''
# /**
#  * @brief Load and validate averaged Kodak rate-distortion results.
#  *
#  * @param[in] csv_path Path to results/kodak_summary.csv.
#  * @return Summary table sorted by estimated bitrate.
#  */
'''

def load_summary(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Kodak summary CSV not found: {csv_path}"
        )

    summary = pd.read_csv(csv_path)

    required = [
        "lambda",
        "mean_psnr",
        "mean_estimated_bpp",
        "num_images",
    ]

    missing = [
        column
        for column in required
        if column not in summary.columns
    ]

    if missing:
        raise ValueError(
            f"Summary CSV is missing columns: "
            f"{', '.join(missing)}"
        )

    if summary.empty:
        raise ValueError(
            "Kodak summary CSV contains no RD points."
        )

    return summary.sort_values(
        "mean_estimated_bpp",
        ignore_index=True,
    )


'''
# /**
#  * @brief Draw and save the Kodak PSNR-versus-estimated-bpp curve.
#  *
#  * @param[in] summary Averaged Kodak rate-distortion table.
#  * @param[in] output_path Destination PNG path.
#  * @return None.
#  */
'''

def plot_rd_curve(summary, output_path):
    sns.set_theme(
        style="whitegrid",
        context="notebook",
    )

    figure, axis = plt.subplots(
        figsize=(7.5, 5.2)
    )

    sns.lineplot(
        data=summary,
        x="mean_estimated_bpp",
        y="mean_psnr",
        marker="o",
        linewidth=2.0,
        markersize=8,
        ax=axis,
    )

    axis.set_xlabel(
        "Estimated bitrate (bpp)",
        fontsize=12,
    )

    axis.set_ylabel(
        "PSNR (dB)",
        fontsize=12,
    )

    axis.set_title(
        "COOL-CHIC Rate-Distortion Performance on Kodak",
        fontsize=13,
        pad=12,
    )

    axis.tick_params(
        axis="both",
        labelsize=10,
    )

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


'''
# /**
#  * @brief Generate the Stage 12 Kodak rate-distortion figure.
#  *
#  * @return None.
#  */
'''

def main():
    args = parse_args()

    root = Path(__file__).resolve().parents[1]

    input_path = (
        args.input
        if args.input.is_absolute()
        else root / args.input
    )

    output_path = (
        args.output
        if args.output.is_absolute()
        else root / args.output
    )

    summary = load_summary(input_path)

    plot_rd_curve(
        summary,
        output_path,
    )

    print(
        f"RD curve saved | "
        f"points={len(summary)} | "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()