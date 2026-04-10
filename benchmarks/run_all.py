"""Run all model benchmarks sequentially.

Models are classified into three tiers:
  - **small**: All-linear or mostly-linear models where ghost coverage can reach
    ~100%.  These serve as *correctness validation* — Spearman ρ ≈ 1.0 is expected.
  - **medium**: Models with convolutional, recurrent, or attention layers where ghost
    coverage is often partial (<100%).  Production proxies for realistic deployments.
  - **large**: ResNet50-scale (~20-25M params) models.  Stress-test ghost
    at production size.

All tiers are run; ``summarize_all.py`` produces one cross-model figure.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.summarize_all import (
    DEFAULT_MODELS,
    LARGE_MODELS,
    MEDIUM_MODELS,
    SMALL_MODELS,
    summarize,
)

# ---------------------------------------------------------------------------
# Model → tier directory mapping (filesystem: testModels/small|medium|large/)
# ---------------------------------------------------------------------------
_MODEL_TIER_DIR: dict[str, str] = {}
for _m in SMALL_MODELS:
    _MODEL_TIER_DIR[_m] = "small"
for _m in MEDIUM_MODELS:
    _MODEL_TIER_DIR[_m] = "medium"
for _m in LARGE_MODELS:
    _MODEL_TIER_DIR[_m] = "large"


def _argv_for_benchmark(
    name: str,
    script_path: Path,
    output_base: Path,
    data_root: str | None,
    device: str,
    scale: str,
) -> list[str]:
    out_dir = output_base / name
    argv = [str(script_path), "--output-dir", str(out_dir), "--device", device]

    # ── scale presets ──
    n_train_models = {
        "synth_regression", "mnist_autoencoder", "transformer_lm", "vae_fashion",
        "encoder_transformer", "gru_lm", "unet_tiny",
        "transformer_lm_large",
    }
    per_class_models = {
        "mnist", "cifar10_cnn", "linear_logistic", "resnet_cifar100",
        "vit_cifar10", "multi_task", "mlp_mixer_cifar10",
        "resnet50_cifar100", "vit_large_cifar10",
    }
    needs_data_root = {
        "mnist", "cifar10_cnn", "mnist_autoencoder", "linear_logistic",
        "resnet_cifar100", "vit_cifar10", "transformer_lm", "vae_fashion",
        "encoder_transformer", "multi_task", "mlp_mixer_cifar10",
        "gru_lm", "unet_tiny",
        "resnet50_cifar100", "transformer_lm_large", "vit_large_cifar10",
    }

    if scale == "smoke":
        if name in n_train_models:
            argv.extend(["--n-train", "200", "--epochs", "10"])
        elif name == "resnet_cifar100":
            argv.extend(["--per-class", "5", "--epochs", "10"])
        elif name in ("resnet50_cifar100", "vit_large_cifar10"):
            # Large models: fewer samples in smoke to keep GPU time reasonable
            argv.extend(["--per-class", "5", "--epochs", "5"])
        else:
            argv.extend(["--per-class", "20", "--epochs", "10"])
    elif scale == "full":
        if name == "synth_regression":
            argv.extend(["--n-train", "50000", "--epochs", "40"])
        elif name == "mnist_autoencoder":
            argv.extend(["--n-train", "60000", "--epochs", "50"])
        elif name == "transformer_lm":
            argv.extend(["--n-train", "50000", "--epochs", "30"])
        elif name == "vae_fashion":
            argv.extend(["--n-train", "60000", "--epochs", "50"])
        elif name == "encoder_transformer":
            argv.extend(["--n-train", "50000", "--epochs", "30"])
        elif name == "gru_lm":
            argv.extend(["--n-train", "50000", "--epochs", "30"])
        elif name == "unet_tiny":
            argv.extend(["--n-train", "50000", "--epochs", "30"])
        elif name == "transformer_lm_large":
            argv.extend(["--n-train", "20000", "--epochs", "15"])
        elif name == "resnet_cifar100":
            argv.extend(["--per-class", "500", "--epochs", "30"])
        elif name == "resnet50_cifar100":
            argv.extend(["--per-class", "200", "--epochs", "15"])
        elif name == "vit_large_cifar10":
            argv.extend(["--per-class", "500", "--epochs", "15"])
        elif name in ("cifar10_cnn", "vit_cifar10", "mlp_mixer_cifar10"):
            argv.extend(["--per-class", "5000", "--epochs", "30"])
        else:
            argv.extend(["--per-class", "6000", "--epochs", "30"])
    # scale == "default": run_benchmark.py defaults only

    if data_root and name in needs_data_root:
        argv.extend(["--data-root", data_root])
    return argv


def main() -> None:
    p = argparse.ArgumentParser()
    ALL_MODELS = SMALL_MODELS + MEDIUM_MODELS + LARGE_MODELS
    p.add_argument(
        "--model",
        choices=(*ALL_MODELS, "all"),
        default="all",
    )
    p.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "benchmarks"),
        help="Base directory; each model writes to <output-dir>/<model_name>/",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="Passed to MNIST/CIFAR run_benchmark.py (e.g. Slurm job data dir).",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Torch device string (use 'cuda' or 'auto' on GPU nodes).",
    )
    p.add_argument(
        "--scale",
        choices=("smoke", "default", "full"),
        default="default",
        help="Dataset size / epochs preset (default = each run_benchmark.py defaults).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Alias for --scale smoke.",
    )
    args = p.parse_args()

    scale = "smoke" if args.smoke else args.scale

    scripts = {
        name: ROOT / "testModels" / _MODEL_TIER_DIR[name] / name / "run_benchmark.py"
        for name in ALL_MODELS
    }
    order = list(ALL_MODELS) if args.model == "all" else [args.model]
    output_base = Path(args.output_dir)

    for name in order:
        path = scripts[name]
        if not path.is_file():
            print(f"Skip missing {path}")
            continue
        tier = _MODEL_TIER_DIR.get(name, "unknown")
        print(f"\n========== benchmark: {name} [{tier}] (scale={scale}) ==========")
        sys.argv = _argv_for_benchmark(
            name,
            path,
            output_base,
            args.data_root,
            args.device,
            scale,
        )
        runpy.run_path(str(path), run_name="__main__")

    print("\n========== cross-model summary ==========")
    bench_out = output_base
    models_for_summary = list(order) if args.model != "all" else list(DEFAULT_MODELS)
    summarize(bench_out, models_for_summary)


if __name__ == "__main__":
    main()
