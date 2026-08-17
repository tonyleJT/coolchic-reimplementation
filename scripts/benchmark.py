import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml


RESULT_COLUMNS = [
    "image",
    "lambda",
    "steps",
    "mse",
    "psnr",
    "latent_bits",
    "mlp_bits",
    "estimated_bpp",
    "training_seconds",
]


'''
# /**
#  * @brief Parse command-line arguments for the Kodak benchmark.
#  *
#  * @return Parsed command-line arguments.
#  */
'''

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run COOL-CHIC independently on several Kodak images and lambdas."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmark.yaml"),
        help="Path to the benchmark YAML configuration.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Repeat runs that already exist in the result CSV.",
    )
    return parser.parse_args()


'''
# /**
#  * @brief Load the Kodak benchmark configuration.
#  *
#  * @param[in] path Path to the YAML configuration file.
#  * @return Parsed configuration dictionary.
#  */
'''

def load_config(path):
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    required = [
        "dataset_dir",
        "images",
        "lambdas",
        "steps",
        "learning_rate",
        "seed",
        "log_every",
        "runs_dir",
        "results_csv",
        "summary_csv",
    ]

    missing = [key for key in required if key not in config]

    if missing:
        raise ValueError(
            f"Missing benchmark configuration fields: {', '.join(missing)}"
        )

    return config


'''
# /**
#  * @brief Load existing detailed benchmark results.
#  *
#  * @param[in] csv_path Path to the detailed result CSV.
#  * @return Existing result table or an empty result table.
#  */
'''

def load_results(csv_path):
    if not csv_path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)

    results = pd.read_csv(csv_path)

    missing = [
        column
        for column in RESULT_COLUMNS
        if column not in results.columns
    ]

    if missing:
        raise ValueError(
            f"Existing result CSV is missing columns: {', '.join(missing)}"
        )

    return results[RESULT_COLUMNS]


'''
# /**
#  * @brief Check whether one image/lambda/step experiment already exists.
#  *
#  * @param[in] results Existing benchmark result table.
#  * @param[in] image_name Kodak image filename.
#  * @param[in] lambda_rd Rate-distortion lambda.
#  * @param[in] steps Number of optimization steps.
#  * @return True when the experiment is already recorded.
#  */
'''

def result_exists(results, image_name, lambda_rd, steps):
    if results.empty:
        return False

    lambda_difference = (
        results["lambda"].astype(float) - float(lambda_rd)
    ).abs()

    matches = (
        (results["image"] == image_name)
        & (results["steps"].astype(int) == int(steps))
        & (lambda_difference < 1e-15)
    )

    return bool(matches.any())


'''
# /**
#  * @brief Build the command used to optimize one Kodak image.
#  *
#  * @param[in] root Project repository root.
#  * @param[in] image_path Input Kodak image path.
#  * @param[in] output_dir Directory for this optimized representation.
#  * @param[in] lambda_rd Rate-distortion lambda.
#  * @param[in] config Benchmark configuration.
#  * @return Command-line argument list for scripts/train.py.
#  */
'''

def build_train_command(
    root,
    image_path,
    output_dir,
    lambda_rd,
    config,
):
    train_script = root / "scripts" / "train.py"

    return [
        sys.executable,
        str(train_script),

        # train.py expects the image as a positional argument.
        str(image_path),

        "--steps",
        str(config["steps"]),

        "--lambda-rd",
        str(lambda_rd),

        "--lr",
        str(config["learning_rate"]),

        "--log-every",
        str(config["log_every"]),

        "--seed",
        str(config["seed"]),

        "--output-dir",
        str(output_dir),
    ]


'''
# /**
#  * @brief Read final quantized COOL-CHIC metrics from one training run.
#  *
#  * @param[in] metrics_path Path to the run metrics.json file.
#  * @return Final metric dictionary.
#  */
'''

def load_run_metrics(metrics_path):
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Training finished without metrics file: {metrics_path}"
        )

    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    required = [
        "mse",
        "psnr",
        "latent_bits",
        "mlp_bits",
        "estimated_bpp",
    ]

    missing = [key for key in required if key not in metrics]

    if missing:
        raise KeyError(
            "metrics.json is missing final Stage 10/11 metrics: "
            + ", ".join(missing)
        )

    return metrics


'''
# /**
#  * @brief Insert or replace one completed benchmark row.
#  *
#  * @param[in] results Existing detailed benchmark results.
#  * @param[in] row Newly completed experiment row.
#  * @return Updated benchmark result table.
#  */
'''

def update_results(results, row):
    if not results.empty:
        lambda_difference = (
            results["lambda"].astype(float) - float(row["lambda"])
        ).abs()

        same_run = (
            (results["image"] == row["image"])
            & (results["steps"].astype(int) == int(row["steps"]))
            & (lambda_difference < 1e-15)
        )

        results = results.loc[~same_run].copy()

    new_row = pd.DataFrame([row], columns=RESULT_COLUMNS)

    results = pd.concat(
        [results, new_row],
        ignore_index=True,
    )

    return results.sort_values(
        by=["lambda", "image"],
        ignore_index=True,
    )


'''
# /**
#  * @brief Save the detailed Kodak experiment CSV.
#  *
#  * @param[in] results Detailed benchmark result table.
#  * @param[in] csv_path Output CSV path.
#  * @return None.
#  */
'''

def save_results(results, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(csv_path, index=False)


# /**
#  * @brief Average PSNR and estimated bpp for every lambda.
#  *
#  * @param[in] results Detailed per-image benchmark results.
#  * @return Summary table containing one RD point per lambda.
#  */
def build_summary(results):
    summary = results.groupby(
        "lambda",
        as_index=False,
    ).agg(
        mean_psnr=("psnr", "mean"),
        mean_estimated_bpp=("estimated_bpp", "mean"),
        num_images=("image", "count"),
    )

    summary = summary.sort_values(
        by="lambda",
    ).reset_index(drop=True)

    return summary


'''
# /**
#  * @brief Run the complete Stage 12 Kodak experiment.
#  *
#  * @return None.
#  */

'''

def main():
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = (
        args.config
        if args.config.is_absolute()
        else root / args.config
    )

    config = load_config(config_path)

    dataset_dir = root / config["dataset_dir"]
    runs_dir = root / config["runs_dir"]
    results_csv = root / config["results_csv"]
    summary_csv = root / config["summary_csv"]

    runs_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_csv)

    images = config["images"]
    lambdas = config["lambdas"]
    steps = int(config["steps"])

    total_runs = len(images) * len(lambdas)

    print(
        f"Stage 12 Kodak benchmark | "
        f"images={len(images)} | "
        f"lambdas={len(lambdas)} | "
        f"runs={total_runs} | "
        f"steps={steps}"
    )

    completed = 0

    for image_name in images:
        image_path = dataset_dir / image_name

        if not image_path.exists():
            raise FileNotFoundError(
                f"Kodak image not found: {image_path}"
            )

        for lambda_rd in lambdas:
            completed += 1

            print()
            print(
                f"[{completed}/{total_runs}] "
                f"image={image_name} | "
                f"lambda={lambda_rd:g}"
            )

            if (
                not args.force
                and result_exists(
                    results,
                    image_name,
                    lambda_rd,
                    steps,
                )
            ):
                print("Already completed - skipping.")
                continue

            lambda_name = f"{float(lambda_rd):.0e}"
            output_dir = (
                runs_dir
                / Path(image_name).stem
                / f"lambda_{lambda_name}"
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            command = build_train_command(
                root=root,
                image_path=image_path,
                output_dir=output_dir,
                lambda_rd=lambda_rd,
                config=config,
            )

            start_time = time.perf_counter()

            subprocess.run(
                command,
                cwd=root,
                check=True,
            )

            training_seconds = (
                time.perf_counter() - start_time
            )

            metrics = load_run_metrics(
                output_dir / "metrics.json"
            )

            row = {
                "image": image_name,
                "lambda": float(lambda_rd),
                "steps": steps,
                "mse": float(metrics["mse"]),
                "psnr": float(metrics["psnr"]),
                "latent_bits": float(
                    metrics["latent_bits"]
                ),
                "mlp_bits": float(
                    metrics["mlp_bits"]
                ),
                "estimated_bpp": float(
                    metrics["estimated_bpp"]
                ),
                "training_seconds": training_seconds,
            }

            results = update_results(
                results,
                row,
            )

            # Save after every run so a long benchmark can
            # safely continue after an interruption.
            save_results(
                results,
                results_csv,
            )

            print(
                "completed | "
                f"psnr={row['psnr']:.3f} | "
                f"estimated_bpp="
                f"{row['estimated_bpp']:.6f} | "
                f"time={training_seconds:.1f}s"
            )

    if results.empty:
        raise RuntimeError(
            "No Kodak experiment results were produced."
        )

    summary = build_summary(results)

    summary_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        summary_csv,
        index=False,
    )

    print()
    print(
        f"Stage 12 benchmark complete | "
        f"results={results_csv} | "
        f"summary={summary_csv}"
    )


if __name__ == "__main__":
    main()