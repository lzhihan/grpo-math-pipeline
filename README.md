# transparent-grpo

Minimal pipeline for math reasoning experiments:
`Base -> SFT -> GRPO -> Evaluation`.

## 1) Environment

- Python 3.10+
- GPU recommended
- Install deps:

```bash
pip install -r requirements.txt
```

## 2) Quick Start

Run full pipeline:

```bash
python scripts/run_all_experiments.py --project-root ./
```

Resume from checkpoints:

```bash
python scripts/run_all_experiments.py --project-root ./ --skip-data --skip-base
python scripts/run_all_experiments.py --project-root ./ --skip-data --skip-base --skip-sft
```

## 3) Local Model / Offline

Use local model path directly:

```bash
python scripts/run_all_experiments.py --project-root ./ --base-model /public/huggingface-models/Qwen/Qwen2.5-1.5B-Instruct
```

Or set env var once:

```bash
export BASE_MODEL_PATH=/public/huggingface-models/Qwen/Qwen2.5-1.5B-Instruct
python scripts/run_all_experiments.py --project-root ./
```

If local path is unavailable, script falls back to `Qwen/Qwen2.5-1.5B-Instruct` from Hub.

## 4) Data Flow

`scripts/prepare_data.py` generates:

- `data/raw/simple_math_source.jsonl`
- `data/processed/train.jsonl`
- `data/processed/val.jsonl`
- `data/processed/test.jsonl`
- `data/processed/sample_debug.jsonl`

## 5) Pipeline Steps

- `scripts/run_baseline.py`: evaluate base model on split.
- `scripts/run_sft.py`: supervised fine-tuning with LoRA.
- `scripts/run_grpo.py`: GRPO training from SFT model.
- `scripts/evaluate.py`: aggregate metrics to CSV.
- `scripts/run_all_experiments.py`: orchestration entrypoint.

## 6) Config Files

- `configs/data.yaml`: sample count, split ratio, data paths.
- `configs/sft.yaml`: SFT training hyperparameters and output path.
- `configs/grpo.yaml`: GRPO hyperparameters and output path.
- `configs/eval.yaml`: generation settings and report output path.

## 7) Outputs

- Baseline predictions: `outputs/baseline/*.jsonl`
- SFT model: `outputs/sft/*`
- SFT/GRPO eval predictions: `outputs/eval/*.jsonl`
- GRPO model: `outputs/grpo/*`
- Summary: `reports/metrics_summary.csv`

## 8) Metrics

- `accuracy`: exact answer accuracy (main metric).
- `boxed_rate`: ratio of outputs containing `\boxed{}`.
- `avg_output_tokens_rough`: rough output length.

Interpretation:

- SFT may increase `boxed_rate` but reduce `accuracy` if outputs over-shorten.
- GRPO should improve `accuracy` relative to SFT; compare all three: base/sft/grpo.

## 9) Reproducibility

- Keep seed fixed in config for stable comparisons.
- Run multiple seeds to confirm gains.
- Inspect case-level differences using prediction JSONL files.
