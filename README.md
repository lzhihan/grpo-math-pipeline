# GRPO-Math-Pipeline

**用 `Qwen2.5-1.5B-Instruct` 这类基础 instruct 模型，在一批简单可验证数学题上，先做一个小 SFT 保证格式稳定，再做小规模 GRPO，并通过 `Base / SFT-only / SFT+GRPO` 三组对照验证 GRPO 的增量效果。**


中文：这是一个最小可复现的数学推理实验流水线：`Base -> SFT -> GRPO -> Evaluation`。  
English: This is a minimal reproducible math-reasoning pipeline: `Base -> SFT -> GRPO -> Evaluation`.

## 1) 项目目标 | Project Goal

中文：
- 使用通用 Instruct 基座模型（默认 `Qwen/Qwen2.5-1.5B-Instruct`）。
- 在可程序判分的简单数学任务上训练与评测。
- 通过对照实验验证 GRPO 的增量效果。

English:
- Use a general instruct base model (default `Qwen/Qwen2.5-1.5B-Instruct`).
- Train/evaluate on simple math tasks with programmatic grading.
- Verify incremental gains from GRPO using controlled comparisons.

## 2) 实验设计 | Experimental Design

中文：建议保留三组结果：
- `Base`：基座模型直接推理。
- `SFT-only`：仅做 LoRA SFT，不做 RL。
- `SFT+GRPO`：先 SFT，再 GRPO。

English: Keep three result groups:
- `Base`: direct inference from the base model.
- `SFT-only`: LoRA SFT only, no RL.
- `SFT+GRPO`: SFT first, then GRPO.

中文：这三组是判断“改进来自哪里”的核心。  
English: These three groups are the core for attributing improvements.

## 3) 数据与提示词 | Data and Prompt

中文：建议数据规模（最小可行）：
- Train: 2,000~5,000
- Val: 300
- Test: 300

English: Recommended minimal scale:
- Train: 2,000~5,000
- Val: 300
- Test: 300

中文：任务应为短链条、可精确判分（算术/一元一次方程等）。  
English: Tasks should be short-chain and exactly checkable (arithmetic / linear equations, etc.).

统一提示词模板 | Unified prompt template:

```text
Solve the following math problem. Show your reasoning briefly and put the final answer in \boxed{}.

Problem: {problem}
```

## 4) 奖励函数 | Reward Setup

中文：建议先只用两项奖励：
- 正确性奖励：正确 `1.0`，错误 `0.0`
- 格式奖励：包含合法 `\boxed{}` 给 `0.05`

English: Start with two rewards only:
- Correctness reward: `1.0` if correct, `0.0` otherwise
- Format reward: `0.05` for valid `\boxed{}` format

## 5) 环境要求 | Environment

中文：
- Python 3.10+
- 建议 GPU

English:
- Python 3.10+
- GPU recommended

安装依赖 | Install dependencies:

```bash
pip install -r requirements.txt
```

## 6) 运行命令 | Run Commands

全流程运行 | Full pipeline:

```bash
python scripts/run_all_experiments.py --project-root ./
```

从 SFT 继续 | Resume from SFT:

```bash
python scripts/run_all_experiments.py --project-root ./ --skip-data --skip-base
```

从 GRPO 继续 | Resume from GRPO:

```bash
python scripts/run_all_experiments.py --project-root ./ --skip-data --skip-base --skip-sft
```

## 7) 本地模型与离线运行 | Local Model and Offline Mode

使用本地模型路径 | Use a local model path:

```bash
python scripts/run_all_experiments.py --project-root ./ --base-model /public/huggingface-models/Qwen/Qwen2.5-1.5B-Instruct
```

设置环境变量（推荐）| Set env var once (recommended):

```bash
export BASE_MODEL_PATH=/public/huggingface-models/Qwen/Qwen2.5-1.5B-Instruct
python scripts/run_all_experiments.py --project-root ./
```

中文：若本地路径不可用，脚本会回退到 Hub ID。  
English: If local path is unavailable, the script falls back to the Hub ID.

## 8) 目录产物 | Outputs

- Baseline 预测 | Baseline predictions: `outputs/baseline/*.jsonl`
- SFT 模型 | SFT model: `outputs/sft/*`
- GRPO 模型 | GRPO model: `outputs/grpo/*`
- 评测预测 | Eval predictions: `outputs/eval/*.jsonl`
- 汇总指标 | Summary metrics: `reports/metrics_summary.csv`

## 9) 指标解读 | Metric Interpretation

关键指标 | Key metrics:
- `accuracy`：主指标，答案正确率 | main metric, exact accuracy
- `boxed_rate`：格式遵循率 | format compliance rate
- `avg_output_tokens_rough`：平均输出长度近似 | rough output length proxy

如何看结果 | How to read results:
- `Base -> SFT`：常见现象是格式先提升。 | format often improves first.
- `SFT -> GRPO`：若 RL 有效，准确率应继续提升。 | accuracy should improve if RL is effective.
- 若 `boxed_rate` 很高但 `accuracy` 不升，通常是“只学格式”。 | high `boxed_rate` without accuracy gain often means format-only learning.

## 10) 成功标准 | Success Criteria

中文：
- 流程可端到端跑通。
- `SFT+GRPO` 在测试集准确率高于 `SFT-only`。
- 提升不仅体现在格式，还体现在正确率。

English:
- The pipeline runs end-to-end.
- `SFT+GRPO` outperforms `SFT-only` on test accuracy.
- Gains are in correctness, not only formatting.

## 11) 配置说明 | Config Map

- `configs/data.yaml`：样本量、切分比例、数据路径 | sample count, split ratio, data paths
- `configs/sft.yaml`：SFT 超参数、输出路径 | SFT hyperparameters, output path
- `configs/grpo.yaml`：GRPO 超参数、输出路径 | GRPO hyperparameters, output path
- `configs/eval.yaml`：生成参数、评测输出路径 | generation settings, evaluation output path


## 12) 每个目录做什么

`configs`
- 放实验配置，避免把超参数写死在脚本里。

`data/raw`
- 原始题库，别直接改。

`data/processed`
- 训练、验证、测试集。
- 这里一定要固定切分，保证可复现。

`data/samples`
- 小样本调试集，比如 32 条或 128 条。
- 用来先跑通脚本，能省很多时间。

`scripts`
- 只放“可直接执行”的入口脚本。

`src`
- 放可复用逻辑，比如 prompt、reward、答案提取、指标计算。

`outputs`
- 训练产物、checkpoint、生成结果。

`reports`
- 最终对比表、错误案例、实验结论。

