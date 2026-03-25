import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def ensure_exists(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required path: {path}\nHint: {hint}")


def validate_model_arg(model_arg: str) -> None:
    p = Path(model_arg)
    if (p.is_absolute() or model_arg.startswith(".")) and not p.exists():
        raise FileNotFoundError(
            f"Base model path does not exist: {p}\n"
            "Use a valid local model directory, or pass a Hub repo id with "
            "--base-model (e.g. Qwen/Qwen2.5-1.5B-Instruct)."
        )


def resolve_base_model(model_arg: str) -> str:
    if model_arg:
        validate_model_arg(model_arg)
        return model_arg

    env_model = os.environ.get("BASE_MODEL_PATH", "").strip()
    if env_model:
        validate_model_arg(env_model)
        print(f"[INFO] Using BASE_MODEL_PATH from env: {env_model}")
        return env_model

    local_candidates = [
        "/public/huggingface-models/Qwen/Qwen2.5-1.5B-Instruct",
        "/models/Qwen/Qwen2.5-1.5B-Instruct",
    ]
    for candidate in local_candidates:
        if Path(candidate).exists():
            print(f"[INFO] Using detected local base model: {candidate}")
            return candidate

    hub_model = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"[INFO] Local base model not found. Falling back to hub repo id: {hub_model}")
    return hub_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--base-model", default="")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--sft-config", default="configs/sft.yaml")
    parser.add_argument("--grpo-config", default="configs/grpo.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--skip-sft", action="store_true")
    parser.add_argument("--skip-grpo", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    py = sys.executable
    base_model = resolve_base_model(args.base_model)

    base_preds = root / "outputs" / "baseline" / "qwen2.5-1.5b-test_predictions.jsonl"
    sft_model = root / "outputs" / "sft" / "qwen2.5-1.5b-sft-lora"
    sft_preds = root / "outputs" / "eval" / "sft_test_predictions.jsonl"
    grpo_model = root / "outputs" / "grpo" / "qwen2.5-1.5b-sft-grpo-lora"
    grpo_preds = root / "outputs" / "eval" / "grpo_test_predictions.jsonl"

    if not args.skip_data:
        run(
            [py, "scripts/prepare_data.py", "--config", args.data_config],
            cwd=root,
        )

    if not args.skip_base:
        run(
            [
                py,
                "scripts/run_baseline.py",
                "--config",
                args.eval_config,
                "--model",
                base_model,
                "--split",
                "test",
                "--output",
                str(base_preds),
            ],
            cwd=root,
        )

    if not args.skip_sft:
        run(
            [py, "scripts/run_sft.py", "--config", args.sft_config, "--model", base_model],
            cwd=root,
        )
    else:
        ensure_exists(
            sft_model,
            "Run SFT first, or remove --skip-sft.",
        )

    # SFT-only test inference
    run(
        [
            py,
            "scripts/run_baseline.py",
            "--config",
            args.eval_config,
            "--model",
            str(sft_model),
            "--split",
            "test",
            "--output",
            str(sft_preds),
        ],
        cwd=root,
    )

    if not args.skip_grpo:
        run(
            [py, "scripts/run_grpo.py", "--config", args.grpo_config],
            cwd=root,
        )
    else:
        ensure_exists(
            grpo_model,
            "Run GRPO first, or remove --skip-grpo.",
        )

    # SFT+GRPO test inference
    run(
        [
            py,
            "scripts/run_baseline.py",
            "--config",
            args.eval_config,
            "--model",
            str(grpo_model),
            "--split",
            "test",
            "--output",
            str(grpo_preds),
        ],
        cwd=root,
    )

    run(
        [
            py,
            "scripts/evaluate.py",
            "--config",
            args.eval_config,
            "--base-preds",
            str(base_preds),
            "--sft-preds",
            str(sft_preds),
            "--grpo-preds",
            str(grpo_preds),
        ],
        cwd=root,
    )

    print("\nAll experiment steps completed.")
    print("Summary CSV: reports/metrics_summary.csv")
    print("Base preds :", base_preds)
    print("SFT preds  :", sft_preds)
    print("GRPO preds :", grpo_preds)


if __name__ == "__main__":
    main()
