**教学型最小实验清单**

这版目标不是“把模型训强”，而是**尽可能干净地验证 GRPO 在起作用**。

**实验目标**
- 用一个基础开源 instruct 模型
- 在简单、可验证的数学题上做小规模 GRPO
- 用对照组回答：模型提升到底是不是 GRPO 带来的

**1. 选模型**
推荐：
- `Qwen/Qwen2.5-1.5B-Instruct`

不选 Math 版，原因很简单：
- 先减少“模型本来就会数学”的干扰
- 让 GRPO 的作用更容易观察

也不要选太弱的 base model，否则 reward 太稀疏，训练容易学不动。

**2. 任务难度**
只做非常简单、短链条、可程序验证的题。

题型建议：
- 两位数加减乘除
- 一元一次方程
- 简单分数运算
- 简单平均数、比例
- 2 到 4 步以内的代数变形

先不要上：
- 几何证明
- 竞赛数学
- 长链条数论/组合

**3. 数据规模**
最小可行版：

- 训练集：`2,000` 到 `5,000` 题
- 验证集：`300` 题
- 测试集：`300` 题

训练/验证/测试必须严格分开。

**4. 数据构造原则**
每道题都要能自动判分。

样本格式：
```json
{
  "problem": "Solve for x: 3x + 5 = 17",
  "answer": "4"
}
```

prompt 统一成：
```text
Solve the following math problem. Show your reasoning briefly and put the final answer in \boxed{}.

Problem: {problem}
```

这样后面 reward 和 eval 都会很稳。

**5. 三个对照组**
这是整个实验最重要的部分。

你至少保留这三组结果：

1. `Base`
- 原始模型直接测试

2. `SFT-only`
- 用训练集做少量监督微调
- 不做 GRPO

3. `SFT + GRPO`
- 先做相同的 SFT
- 再做 GRPO

如果你时间更紧，也可以做两组：
- `Base`
- `GRPO` 或 `SFT+GRPO`

但最推荐还是三组，因为这样你能分清：
- 是训练数据起作用
- 还是 RL 额外带来了增益

**6. 为什么先做小 SFT**
因为对基础 instruct 模型来说，直接 GRPO 可能有两个问题：
- 不按你要的格式输出
- 正确样本太少，reward 信号太稀

所以建议加一个很小的冷启动：
- `500` 到 `1,000` 条就够
- 只是让模型学会：
  - 按数学题格式回答
  - 用 `\boxed{}` 给最终答案

这一步不是为了提升很多数学能力，而是为了让 GRPO 更稳定。

**7. 奖励函数**
只保留两个奖励。

主奖励：
- 最终答案正确：`1.0`
- 错误：`0.0`

辅助奖励：
- 包含合法 `\boxed{}`：`0.05`

不要再加别的。你现在要的是**可解释性**，不是复杂优化。

**8. 评测指标**
每次评测都记这几项：

- `accuracy / exact match`
- `boxed format rate`
- 平均输出长度
- 典型错误样例数量
- 典型“格式对但答案错”的样例

你真正要看的是：
- `SFT-only` 相比 `Base` 提升多少
- `SFT+GRPO` 相比 `SFT-only` 再提升多少

**9. 训练参数建议**
单卡 A800 80GB，保守起步：

SFT：
- LoRA
- `lr = 1e-4` 左右
- `epochs = 1-2`
- `max_length = 512`

GRPO：
- `num_generations = 4`
- `per_device_train_batch_size = 1-2`
- `gradient_accumulation_steps = 8-16`
- `max_prompt_length = 256`
- `max_completion_length = 256`
- `lr = 1e-5` 到 `5e-5`
- 训练步数先控制在 `300-800`

先跑小，不要第一轮就加大 completion。

**10. 成功标准**
这次实验不需要“很强”，只要满足这些就算成功：

- pipeline 跑通
- `SFT+GRPO` 在测试集上优于 `SFT-only`
- 提升不是只体现在格式，而是准确率也上升
- 你能找到一批题，说明模型在 RL 后更稳定地给出正确答案

**11. 你最该观察的现象**
如果实验设计得好，你会看到几种典型情况：

- `Base -> SFT`
  - 格式遵守率明显提升
  - 简单题准确率小幅上涨

- `SFT -> GRPO`
  - 正确率继续涨
  - 同类题的稳定性变强
  - 模型更倾向输出可判分的最终答案

如果你看到的是：
- format rate 大涨
- accuracy 几乎不动

那说明模型主要学会了“配合奖励格式化”，而不是数学推理真的变强了。

**12. 最小执行顺序**
1. 准备简单数学数据集，并拆分 train/val/test
2. 写一个 `boxed` 答案提取和判分脚本
3. 跑 `Base` baseline
4. 做一个很小的 `SFT-only`
5. 评测 `SFT-only`
6. 在相同训练集上做 `GRPO`
7. 评测 `SFT+GRPO`
8. 做对照分析

**13. 最终输出建议**
实验结束后，你整理一张很小的表就够了：

- Base accuracy
- SFT-only accuracy
- SFT+GRPO accuracy
- boxed rate
- average length

再补 10 到 20 个案例：
- GRPO 后变好的题
- GRPO 后仍然错的题
- 看起来像 reward hacking 的题

**14. 一句话版方案**
**用 `Qwen2.5-1.5B-Instruct` 这类基础 instruct 模型，在一批简单可验证数学题上，先做一个小 SFT 保证格式稳定，再做小规模 GRPO，并通过 `Base / SFT-only / SFT+GRPO` 三组对照验证 GRPO 的增量效果。**

如果你愿意，我下一条可以继续直接给你两样最实用的东西之一：

1. **这个实验的目录结构 + 文件清单**
2. **一份可直接改的 TRL/GRPO 训练脚手架配置**

---

**推荐目录结构**

```text
D:\Programs\PythonProjects\transparent-grpo\
├─ README.md
├─ requirements.txt
├─ configs\
│  ├─ data.yaml
│  ├─ sft.yaml
│  ├─ grpo.yaml
│  └─ eval.yaml
├─ data\
│  ├─ raw\
│  │  └─ simple_math_source.jsonl
│  ├─ processed\
│  │  ├─ train.jsonl
│  │  ├─ val.jsonl
│  │  └─ test.jsonl
│  └─ samples\
│     └─ debug_32.jsonl
├─ scripts\
│  ├─ prepare_data.py
│  ├─ run_baseline.py
│  ├─ run_sft.py
│  ├─ run_grpo.py
│  ├─ evaluate.py
│  └─ inspect_cases.py
├─ src\
│  ├─ prompts.py
│  ├─ dataset.py
│  ├─ answer_extractor.py
│  ├─ rewards.py
│  ├─ metrics.py
│  └─ utils.py
├─ outputs\
│  ├─ baseline\
│  ├─ sft\
│  ├─ grpo\
│  └─ eval\
├─ reports\
│  ├─ metrics_summary.csv
│  ├─ error_cases.md
│  └─ experiment_notes.md
└─ notebooks\
   └─ analysis.ipynb
```

**每个目录做什么**

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

`notebooks`
- 可选，用来做分析图表；不是必须。

**最小文件清单**

你第一版只需要真的写好这些文件：

**1. [requirements.txt](D:/Programs/PythonProjects/transparent-grpo/requirements.txt)**
- 依赖清单。
- 最少包括：`torch`, `transformers`, `trl`, `peft`, `accelerate`, `datasets`, `pandas`

**2. [configs/data.yaml](D:/Programs/PythonProjects/transparent-grpo/configs/data.yaml)**
- 数据路径
- train/val/test 切分比例
- 随机种子
- 是否要求 `\boxed{}`

**3. [configs/sft.yaml](D:/Programs/PythonProjects/transparent-grpo/configs/sft.yaml)**
- 模型名
- LoRA 参数
- batch size
- lr
- epochs
- max length

**4. [configs/grpo.yaml](D:/Programs/PythonProjects/transparent-grpo/configs/grpo.yaml)**
- 模型路径
- `num_generations`
- `max_prompt_length`
- `max_completion_length`
- lr
- batch size
- eval/save/logging steps

**5. [configs/eval.yaml](D:/Programs/PythonProjects/transparent-grpo/configs/eval.yaml)**
- 要评测的模型路径
- 数据路径
- 解码参数
- 输出保存路径

**6. [scripts/prepare_data.py](D:/Programs/PythonProjects/transparent-grpo/scripts/prepare_data.py)**
负责：
- 读取原始数学题
- 清洗
- 统一字段名
- 切分 train/val/test
- 输出 jsonl

输出格式建议统一成：
```json
{"problem":"Solve for x: 3x+5=17","answer":"4"}
```

**7. [src/prompts.py](D:/Programs/PythonProjects/transparent-grpo/src/prompts.py)**
负责：
- 定义统一 prompt 模板

例如：
```text
Solve the following math problem. Show your reasoning briefly and put the final answer in \boxed{}.

Problem: {problem}
```

**8. [src/answer_extractor.py](D:/Programs/PythonProjects/transparent-grpo/src/answer_extractor.py)**
负责：
- 从模型输出中提取 `\boxed{...}`
- 如果没 boxed，做一个后备提取规则

这是最关键的小模块之一。

**9. [src/rewards.py](D:/Programs/PythonProjects/transparent-grpo/src/rewards.py)**
负责：
- 正确性奖励
- 格式奖励

第一版只要两个函数：
- `correctness_reward`
- `format_reward`

**10. [src/metrics.py](D:/Programs/PythonProjects/transparent-grpo/src/metrics.py)**
负责：
- accuracy
- boxed rate
- avg completion length

**11. [scripts/run_baseline.py](D:/Programs/PythonProjects/transparent-grpo/scripts/run_baseline.py)**
负责：
- 用原始模型在 val/test 上生成答案
- 保存结果
- 调用 metrics 统计 baseline

**12. [scripts/run_sft.py](D:/Programs/PythonProjects/transparent-grpo/scripts/run_sft.py)**
负责：
- 跑一个很小的 SFT
- 保存 adapter 或 checkpoint

**13. [scripts/run_grpo.py](D:/Programs/PythonProjects/transparent-grpo/scripts/run_grpo.py)**
负责：
- 加载 SFT 后模型
- 接 TRL 的 GRPOTrainer
- 调用 reward 函数训练

**14. [scripts/evaluate.py](D:/Programs/PythonProjects/transparent-grpo/scripts/evaluate.py)**
负责：
- 对 `base / sft / grpo` 三个模型统一评测
- 输出表格到 `reports/metrics_summary.csv`

**15. [scripts/inspect_cases.py](D:/Programs/PythonProjects/transparent-grpo/scripts/inspect_cases.py)**
负责：
- 抽取典型样例
- 找出：
  - baseline 错、grpo 对
  - baseline 对、grpo 错
  - 格式对但答案错

这个脚本很有学习价值。

**第一版可以暂时不写的**
这些不是第一天必须有：
- `notebooks/analysis.ipynb`
- 复杂日志可视化
- 多 reward 混合
- 推理过程评分器
- vLLM 支持
- Web UI

**推荐的实验产物命名**
`outputs` 下建议这样放：

```text
outputs\
├─ baseline\
│  └─ qwen2.5-1.5b-test_predictions.jsonl
├─ sft\
│  └─ qwen2.5-1.5b-sft-lora\
├─ grpo\
│  └─ qwen2.5-1.5b-sft-grpo-lora\
└─ eval\
   ├─ base_metrics.json
   ├─ sft_metrics.json
   └─ grpo_metrics.json
```

**你真正最先要完成的顺序**
别一起开工，按这个顺序最稳：

1. `prepare_data.py`
2. `prompts.py`
3. `answer_extractor.py`
4. `run_baseline.py`
5. `metrics.py`
6. `run_sft.py`
7. `rewards.py`
8. `run_grpo.py`
9. `evaluate.py`
10. `inspect_cases.py`

**最小里程碑**
当你完成下面 3 件事，就已经走通一半了：

- 能从题目生成统一 prompt
- 能从输出稳定提取 `\boxed{答案}`
- 能跑出 baseline accuracy

接下来 SFT 和 GRPO 只是往这个闭环里接模块。

如果你愿意，我下一条可以继续直接给你 **这些文件里最核心的 5 个数据结构/函数接口设计**，这样你写代码时不会乱。

---

下面给你一套**最小且够用的接口设计**。目标是先把工程骨架钉住，让 `baseline / SFT / GRPO / eval` 共用同一套数据与判分逻辑。

**1. 统一样本结构**
文件：[train.jsonl](D:/Programs/PythonProjects/transparent-grpo/data/processed/train.jsonl)

每条样本建议固定成这样：

```json
{
  "id": "train-000001",
  "problem": "Solve for x: 3x + 5 = 17",
  "answer": "4",
  "metadata": {
    "source": "synthetic_linear_eq",
    "difficulty": "easy",
    "split": "train"
  }
}
```

字段约定：
- `id`: 唯一标识，后面做 case analysis 很有用
- `problem`: 题目正文
- `answer`: 标准最终答案，尽量归一化成字符串
- `metadata`: 可选，但建议保留

**2. prompt 接口**
文件：[prompts.py](D:/Programs/PythonProjects/transparent-grpo/src/prompts.py)

```python
SYSTEM_PROMPT = (
    "You are a helpful math tutor. "
    "Solve the problem carefully. "
    "Put the final answer in \\boxed{}."
)

def build_prompt(problem: str) -> str:
    return (
        "Solve the following math problem. "
        "Show your reasoning briefly and put the final answer in \\boxed{}.\n\n"
        f"Problem: {problem}"
    )
```

如果你后面想走 chat template，也尽量把接口保持简单：

```python
def build_messages(problem: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem: {problem}"},
    ]
```

建议：
- 第一版只保留一种 prompt 格式
- `baseline / SFT / GRPO / eval` 全部复用，避免变量太多

**3. 数据读取接口**
文件：[dataset.py](D:/Programs/PythonProjects/transparent-grpo/src/dataset.py)

```python
import json
from pathlib import Path
from typing import Iterable

def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def to_sft_records(rows: list[dict]) -> list[dict]:
    records = []
    for row in rows:
        prompt = build_prompt(row["problem"])
        completion = f"Final answer: \\boxed{{{row['answer']}}}"
        records.append({
            "id": row["id"],
            "prompt": prompt,
            "completion": completion,
            "answer": row["answer"],
        })
    return records
```

这里最重要的是：
- 数据集原始结构和训练结构分开
- 不要把各种临时字段混在一起

**4. 答案提取接口**
文件：[answer_extractor.py](D:/Programs/PythonProjects/transparent-grpo/src/answer_extractor.py)

```python
import re

BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")

def extract_boxed_answer(text: str) -> str | None:
    matches = BOXED_PATTERN.findall(text)
    if not matches:
        return None
    return normalize_answer(matches[-1])

def extract_answer(text: str) -> str | None:
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed

    # fallback: last numeric-looking token
    fallback = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", text)
    if not fallback:
        return None
    return normalize_answer(fallback[-1])

def normalize_answer(answer: str) -> str:
    answer = answer.strip()
    answer = answer.replace(" ", "")
    if answer.endswith(".0"):
        answer = answer[:-2]
    return answer
```

第一版不要追求完美数学 parser。  
只要你题目控制得简单，这个提取器已经够做教学实验。

**5. 指标接口**
文件：[metrics.py](D:/Programs/PythonProjects/transparent-grpo/src/metrics.py)

```python
from statistics import mean

def is_correct(predicted: str | None, gold: str) -> bool:
    if predicted is None:
        return False
    return predicted == gold

def compute_metrics(rows: list[dict]) -> dict:
    correct = []
    boxed = []
    lengths = []

    for row in rows:
        predicted = row.get("predicted_answer")
        gold = row["answer"]
        output_text = row.get("model_output", "")

        correct.append(is_correct(predicted, gold))
        boxed.append("\\boxed{" in output_text)
        lengths.append(len(output_text.split()))

    return {
        "num_samples": len(rows),
        "accuracy": sum(correct) / len(correct) if rows else 0.0,
        "boxed_rate": sum(boxed) / len(boxed) if rows else 0.0,
        "avg_output_tokens_rough": mean(lengths) if lengths else 0.0,
    }
```

这里故意保持简单：
- `accuracy`
- `boxed_rate`
- 粗略输出长度

这三项已经够你判断很多现象。

**6. reward 接口**
文件：[rewards.py](D:/Programs/PythonProjects/transparent-grpo/src/rewards.py)

```python
from src.answer_extractor import extract_answer, normalize_answer

def correctness_reward(completions: list[str], answers: list[str], **kwargs) -> list[float]:
    rewards = []
    for completion, gold in zip(completions, answers):
        pred = extract_answer(completion)
        rewards.append(1.0 if pred == normalize_answer(gold) else 0.0)
    return rewards

def format_reward(completions: list[str], **kwargs) -> list[float]:
    rewards = []
    for completion in completions:
        rewards.append(0.05 if "\\boxed{" in completion else 0.0)
    return rewards
```

你后面接 `GRPOTrainer` 时，通常就是把它们作为 reward funcs 传进去。

**7. baseline/eval 结果结构**
文件：[qwen2.5-1.5b-test_predictions.jsonl](D:/Programs/PythonProjects/transparent-grpo/outputs/baseline/qwen2.5-1.5b-test_predictions.jsonl)

建议每条预测保存成这样：

```json
{
  "id": "test-000001",
  "problem": "Solve for x: 3x + 5 = 17",
  "answer": "4",
  "prompt": "Solve the following math problem...",
  "model_output": "We solve 3x+5=17 ... Final answer: \\boxed{4}",
  "predicted_answer": "4",
  "is_correct": true,
  "model_name": "Qwen/Qwen2.5-1.5B-Instruct"
}
```

这个结构很好用，因为：
- `evaluate.py` 直接算指标
- `inspect_cases.py` 直接筛错题/进步题
- 后面换模型也能复用

**8. `run_baseline.py` 的最小职责**
文件：[run_baseline.py](D:/Programs/PythonProjects/transparent-grpo/scripts/run_baseline.py)

它只做 4 件事：
- 读取 test/val 数据
- 构造 prompt
- 调模型生成
- 保存预测结果并计算 metrics

它不应该负责：
- 数据清洗
- 复杂分析
- reward 逻辑

**9. `run_grpo.py` 的最小输入输出**
输入：
- 一个训练集路径
- 一个验证集路径
- 一个模型名或 SFT checkpoint
- 一组 reward 函数

输出：
- adapter/checkpoint 到 [outputs/grpo](D:/Programs/PythonProjects/transparent-grpo/outputs/grpo)
- 训练日志
- 可选的中间评测结果

你可以把脚本设计成这种参数接口：

```python
def train_grpo(
    model_name_or_path: str,
    train_path: str,
    val_path: str,
    output_dir: str,
) -> None:
    ...
```

**10. `inspect_cases.py` 最值得保留的函数**
文件：[inspect_cases.py](D:/Programs/PythonProjects/transparent-grpo/scripts/inspect_cases.py)

至少写这两个筛选逻辑：

```python
def improved_cases(before_rows: list[dict], after_rows: list[dict]) -> list[dict]:
    ...

def regressed_cases(before_rows: list[dict], after_rows: list[dict]) -> list[dict]:
    ...
```

这会直接帮助你回答：
- GRPO 具体改好了哪些题
- 又弄坏了哪些题

这比只看总分更有学习价值。

**最关键的工程原则**
整套实验里，最重要的是这三件事保持统一：
- 同一套 prompt
- 同一套答案提取器
- 同一套评测逻辑

否则你很容易把 prompt 改动、提取器改动、reward 改动误当成 GRPO 效果。

**建议你下一步先写的 5 个文件**
优先级如下：
1. [prompts.py](D:/Programs/PythonProjects/transparent-grpo/src/prompts.py)
2. [answer_extractor.py](D:/Programs/PythonProjects/transparent-grpo/src/answer_extractor.py)
3. [metrics.py](D:/Programs/PythonProjects/transparent-grpo/src/metrics.py)
4. [prepare_data.py](D:/Programs/PythonProjects/transparent-grpo/scripts/prepare_data.py)
5. [run_baseline.py](D:/Programs/PythonProjects/transparent-grpo/scripts/run_baseline.py)

把这 5 个做好，你的实验骨架就已经成立了。

如果你愿意，我下一条可以继续给你：
- **一份最小数据集构造方案**，适合你这个教学实验  
或者
- **`prepare_data.py` 的具体设计**，包括简单数学题怎么自动生成。

---

可以开始三组对照实验了。当前项目骨架已经支持这三组流程，但你还需要先在本机安装依赖并实际执行训练/评测命令；目前仓库里还没有真实实验结果文件。

**执行步骤**

1. 安装依赖  
`pip install -r D:\Programs\PythonProjects\transparent-grpo\requirements.txt`

2. 生成训练/验证/测试数据（会覆盖 `data/processed/*.jsonl` 占位文件）  
`python D:\Programs\PythonProjects\transparent-grpo\scripts\prepare_data.py --config D:\Programs\PythonProjects\transparent-grpo\configs\data.yaml`

3. 组1：Base（原始模型直接测试）  
`python D:\Programs\PythonProjects\transparent-grpo\scripts\run_baseline.py --config D:\Programs\PythonProjects\transparent-grpo\configs\eval.yaml --model Qwen/Qwen2.5-1.5B-Instruct --split test --output D:\Programs\PythonProjects\transparent-grpo\outputs\baseline\qwen2.5-1.5b-test_predictions.jsonl`

4. 组2：SFT-only  
先做 SFT：  
`python D:\Programs\PythonProjects\transparent-grpo\scripts\run_sft.py --config D:\Programs\PythonProjects\transparent-grpo\configs\sft.yaml`  
再用 SFT 模型做同一测试集推理：  
`python D:\Programs\PythonProjects\transparent-grpo\scripts\run_baseline.py --config D:\Programs\PythonProjects\transparent-grpo\configs\eval.yaml --model D:\Programs\PythonProjects\transparent-grpo\outputs\sft\qwen2.5-1.5b-sft-lora --split test --output D:\Programs\PythonProjects\transparent-grpo\outputs\eval\sft_test_predictions.jsonl`

5. 组3：SFT + GRPO  
先在同一个 SFT 模型基础上做 GRPO：  
`python D:\Programs\PythonProjects\transparent-grpo\scripts\run_grpo.py --config D:\Programs\PythonProjects\transparent-grpo\configs\grpo.yaml`  
再用 GRPO 模型做同一测试集推理：  
`python D:\Programs\PythonProjects\transparent-grpo\scripts\run_baseline.py --config D:\Programs\PythonProjects\transparent-grpo\configs\eval.yaml --model D:\Programs\PythonProjects\transparent-grpo\outputs\grpo\qwen2.5-1.5b-sft-grpo-lora --split test --output D:\Programs\PythonProjects\transparent-grpo\outputs\eval\grpo_test_predictions.jsonl`

6. 汇总三组指标  
`python D:\Programs\PythonProjects\transparent-grpo\scripts\evaluate.py --config D:\Programs\PythonProjects\transparent-grpo\configs\eval.yaml --base-preds D:\Programs\PythonProjects\transparent-grpo\outputs\baseline\qwen2.5-1.5b-test_predictions.jsonl --sft-preds D:\Programs\PythonProjects\transparent-grpo\outputs\eval\sft_test_predictions.jsonl --grpo-preds D:\Programs\PythonProjects\transparent-grpo\outputs\eval\grpo_test_predictions.jsonl`

**结果保存位置**

- Base 预测结果：[qwen2.5-1.5b-test_predictions.jsonl](D:/Programs/PythonProjects/transparent-grpo/outputs/baseline/qwen2.5-1.5b-test_predictions.jsonl)
- SFT 模型目录：[qwen2.5-1.5b-sft-lora](D:/Programs/PythonProjects/transparent-grpo/outputs/sft/qwen2.5-1.5b-sft-lora)
- SFT 测试预测：[sft_test_predictions.jsonl](D:/Programs/PythonProjects/transparent-grpo/outputs/eval/sft_test_predictions.jsonl)
- GRPO 模型目录：[qwen2.5-1.5b-sft-grpo-lora](D:/Programs/PythonProjects/transparent-grpo/outputs/grpo/qwen2.5-1.5b-sft-grpo-lora)
- GRPO 测试预测：[grpo_test_predictions.jsonl](D:/Programs/PythonProjects/transparent-grpo/outputs/eval/grpo_test_predictions.jsonl)
- 三组汇总表：[metrics_summary.csv](D:/Programs/PythonProjects/transparent-grpo/reports/metrics_summary.csv)

**如何对比**

先看 [metrics_summary.csv](D:/Programs/PythonProjects/transparent-grpo/reports/metrics_summary.csv) 里的三列核心指标：`accuracy`、`boxed_rate`、`avg_output_tokens_rough`。  
再做样例级对比（看哪些题变好/变坏）：  
`python D:\Programs\PythonProjects\transparent-grpo\scripts\inspect_cases.py --before D:\Programs\PythonProjects\transparent-grpo\outputs\baseline\qwen2.5-1.5b-test_predictions.jsonl --after D:\Programs\PythonProjects\transparent-grpo\outputs\eval\grpo_test_predictions.jsonl --limit 30`

如果你愿意，我下一步可以直接帮你加一个一键脚本（`scripts/run_all_experiments.py`），一次跑完三组并自动生成对比表。

---

一键脚本已经加好了：[run_all_experiments.py](D:/Programs/PythonProjects/transparent-grpo/scripts/run_all_experiments.py)。

直接运行：
```powershell
python D:\Programs\PythonProjects\transparent-grpo\scripts\run_all_experiments.py --project-root D:\Programs\PythonProjects\transparent-grpo
```

它会按顺序执行：`prepare_data -> Base -> SFT -> SFT-only测试 -> GRPO -> SFT+GRPO测试 -> evaluate`，并把最终对比表写到 [metrics_summary.csv](D:/Programs/PythonProjects/transparent-grpo/reports/metrics_summary.csv)。

可选跳过参数也支持：`--skip-data --skip-base --skip-sft --skip-grpo`。我已经加了模型存在性检查，跳过训练时如果对应模型不存在，会给出明确提示。

---

在 Linux 上你现在可以这样跑：

```bash
cd ~/transparent-grpo
pip install -r requirements.txt
python scripts/run_all_experiments.py --project-root ./
```

---

已完成，默认模型路径现在更稳了，不会再因为无效本地路径直接报 `HFValidationError`。

改动如下：

- [run_all_experiments.py](D:/Programs/PythonProjects/transparent-grpo/scripts/run_all_experiments.py)
- [README.md](D:/Programs/PythonProjects/transparent-grpo/README.md)

现在 `run_all_experiments.py` 的 `--base-model` 逻辑是：

1. 你显式传了 `--base-model`：就用它，并校验本地路径是否存在  
2. 没传：先看环境变量 `BASE_MODEL_PATH`  
3. 再没配：自动探测常见本地目录  
4. 都没有：自动回退到 `Qwen/Qwen2.5-1.5B-Instruct`（Hub repo id）

你在 Linux 上直接运行即可：

```bash
python scripts/run_all_experiments.py --project-root ./
```

如果你有固定本地模型目录，建议这样设置一次：

```bash
export BASE_MODEL_PATH=/你的真实模型目录
python scripts/run_all_experiments.py --project-root ./
```

如果你愿意，我下一步可以帮你把 `run_sft.py` 也加同样的“自动模型路径解析”，让单独跑 SFT 时也完全一致。

---

```bash
cd ~/transparent-grpo
python scripts/run_all_experiments.py --project-root ./ --base-model /你的本地模型目录
```

例如你的目录就是这个时：

```bash
python scripts/run_all_experiments.py --project-root ./ --base-model /public/huggingface-models/Qwen/Qwen2.5-1.5B-Instruct
```

也可以先设环境变量（之后可不再传 `--base-model`）：

```bash
export BASE_MODEL_PATH=/public/huggingface-models/Qwen/Qwen2.5-1.5B-Instruct
python scripts/run_all_experiments.py --project-root ./
```
