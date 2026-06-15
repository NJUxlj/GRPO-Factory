# M05: RewardManager 集成（4 种评分函数 + LLM-as-Judge）

> **阶段编号**: M05  
> **对应原里程碑**: M3  
> **创建时间**: 2026-06-10  
> **预计工期**: 5-6天  
> **前置阶段**: 无（可与 M01-M04 并行开发）

---

## 1. 阶段定位

本阶段实现 **RewardManager** 模块，提供 4 种独立的评分函数，覆盖常见 RLHF 场景：

1. **math_score**：数学答案评分（支持 boxed/hash/last_number 三种抽取模式）
2. **multiple_choice_score**：多选题评分（A/B/C/D 选项抽取）
3. **string_match_score**：字符串匹配（严格/规范化两种模式）
4. **llm_judge_score**：LLM-as-Judge（异步并发调用外部大模型）

RewardManager 与算法（GRPO/DAPO/GSPO/DCPO）解耦，可任意组合，为所有算法提供奖励信号。

---

## 2. 阶段目标

### 2.1 业务目标

- 支持数学推理、多选题、通用 QA、开放式生成等多种任务的 RLHF 训练
- 提供 LLM-as-Judge 能力，用于复杂奖励场景（工具调用、Agent 行为评估）

### 2.2 技术目标

- 新建 `reward/` 子目录结构（manager/registry/math/multiple_choice/string_match/llm_judge）
- 实现 `registry.py` 的 `SCORE_REGISTRY` + `get_score_fn`
- 实现 4 种评分函数（math/multiple_choice/string_match/llm_judge）
- 实现 `RewardManager` 主类（对齐 verl NaiveRewardManager 接口）
- 在 `finetuning_args.py` 中添加 `grpo_reward_*` 参数
- Trainer 中将 `reward_fn` 替换为 `reward_manager`
- Workflow 中加入 `create_reward_manager` 工厂方法
- 单元测试：4 种评分函数在 toy 样本上的数值正确性
- 集成测试：DCPO + math 在小数据集上 loss 下降

---

## 3. 核心任务

### 任务 3.1: 创建 `reward/` 目录结构

**任务描述**：建立 RewardManager 模块的目录骨架。

**文件清单**：
```
src/llamafactory/train/grpo/reward/
├── __init__.py
├── manager.py       # RewardManager 主类
├── registry.py      # SCORE_REGISTRY 评分函数注册表
├── math.py          # 数学答案评分
├── multiple_choice.py # 多选题评分
├── string_match.py  # 字符串匹配
└── llm_judge.py     # LLM-as-Judge
```

---

### 任务 3.2: 实现评分函数注册表 (`registry.py`)

**任务描述**：实现 `SCORE_REGISTRY` 字典和 `get_score_fn` 路由函数。

**技术细节**：

```python
from typing import Callable, Dict
from .math import math_score
from .multiple_choice import multiple_choice_score
from .string_match import string_match_score
from .llm_judge import llm_judge_score

SCORE_REGISTRY: Dict[str, Callable] = {
    "math": math_score,
    "multiple_choice": multiple_choice_score,
    "string_match": string_match_score,
    "llm_judge": llm_judge_score,
}


def get_score_fn(reward_type: str) -> Callable:
    if reward_type not in SCORE_REGISTRY:
        raise ValueError(
            f"Unknown reward_type={reward_type}. "
            f"Available: {list(SCORE_REGISTRY.keys())}"
        )
    return SCORE_REGISTRY[reward_type]
```

---

### 任务 3.3: 实现数学答案评分 (`math.py`)

**任务描述**：支持 3 种答案抽取模式（boxed/hash/last_number）。

**技术细节**：

```python
import re
from typing import Optional


def _extract_boxed_answer(text: str) -> Optional[str]:
    """匹配 \\boxed{...}"""
    m = re.search(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return m.group(1).strip() if m else None


def _extract_hash_answer(text: str) -> Optional[str]:
    """匹配 GSM8K 风格的 #### 数字"""
    m = re.search(r"####\s*(-?\d[\d,\.]*)", text)
    if not m:
        return None
    return m.group(1).replace(",", "").rstrip(".")


def _extract_last_number(text: str) -> Optional[str]:
    """抓取最后一个数字（兜底策略）"""
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


_EXTRACTORS = {
    "boxed": _extract_boxed_answer,
    "hash": _extract_hash_answer,
    "last_number": _extract_last_number,
}


def _normalize_answer(ans: str) -> str:
    """规范化: 去空格/逗号/末尾点号, 转 str"""
    return ans.replace(",", "").replace(" ", "").rstrip(".").lower()


def math_score(
    response: str,
    ground_truth: str,
    extract_mode: str = "boxed",
) -> float:
    """从 response 中抽取数学答案, 与 ground_truth 规范化比对
    返回: 1.0 (匹配) 或 0.0 (不匹配/无法抽取)
    """
    extractor = _EXTRACTORS.get(extract_mode, _extract_boxed_answer)
    pred = extractor(response)
    if pred is None:
        return 0.0
    return 1.0 if _normalize_answer(pred) == _normalize_answer(ground_truth) else 0.0
```

---

### 任务 3.4: 实现多选题评分 (`multiple_choice.py`)

**任务描述**：抽取 A/B/C/D 选项，与 ground_truth 比对。

**技术细节**：

```python
import re
from typing import Optional


def _extract_choice(response: str, pattern: str) -> Optional[str]:
    """从 response 中抽取 A/B/C/D 选项"""
    m = re.search(pattern, response)
    if not m:
        return None
    for g in m.groups():
        if g is not None:
            return g.upper()
    return None


def multiple_choice_score(
    response: str,
    ground_truth: str,
    pattern: str = r"(?i)\\boxed\{\s*([A-D])\s*\}|answer\s*[:：]?\s*([A-D])",
) -> float:
    """抽取 A/B/C/D 选项, 与 ground_truth 比对
    返回: 1.0 / 0.0
    """
    pred = _extract_choice(response, pattern)
    if pred is None:
        return 0.0
    return 1.0 if pred == ground_truth.strip().upper() else 0.0
```

---

### 任务 3.5: 实现字符串匹配 (`string_match.py`)

**任务描述**：支持严格相等和规范化匹配两种模式。

**技术细节**：

```python
import re
import string


_WHITESPACE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(text: str, strict: bool = False) -> str:
    """规范化:
    strict=True:  仅去多余空白
    strict=False: 去空白 + 去标点 + 小写
    """
    text = text.strip()
    if strict:
        return _WHITESPACE.sub(" ", text)
    text = _PUNCT_TABLE.sub("", text)
    return _WHITESPACE.sub("", text).lower()


def string_match_score(
    response: str,
    ground_truth: str,
    strict: bool = False,
) -> float:
    """字符串完全匹配 / 规范化匹配
    返回: 1.0 / 0.0
    """
    return 1.0 if _normalize(response, strict) == _normalize(ground_truth, strict) else 0.0
```

---

### 任务 3.6: 实现 LLM-as-Judge (`llm_judge.py`)

**任务描述**：异步并发调用外部大模型做语义一致性判断。

**技术细节**：

```python
import asyncio
import aiohttp
from typing import List, Optional


DEFAULT_JUDGE_PROMPT = """You are a strict answer-checking judge.

You will be given TWO answers:
- **Ground Truth Answer**: the correct/reference answer
- **Model Prediction**: the answer produced by a language model

Your task: determine whether the **Model Prediction** is **semantically equivalent** to the **Ground Truth Answer**.

Rules:
1. Ignore minor formatting differences (whitespace, punctuation, casing).
2. For numbers, treat mathematically equivalent values as equal (e.g. 1/2 == 0.5).
3. For multi-choice or short factual answers, the prediction must match the ground truth.
4. If the prediction is partially correct but missing key information, return "no".
5. If the prediction is empty, irrelevant, or does not address the question, return "no".

Output ONLY one token, either:
- "yes"  → semantically equivalent
- "no"   → not equivalent

Do not output any explanation.

---

Ground Truth Answer:
{ground_truth}

Model Prediction:
{prediction}

Your verdict (yes/no):"""


class LLMJudgeClient:
    """异步 LLM 评判客户端"""

    def __init__(
        self,
        url: str,
        model: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        timeout: int = 30,
        concurrency: int = 16,
        fallback_score: float = 0.0,
        prompt_template: str = DEFAULT_JUDGE_PROMPT,
    ):
        self.url = url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(concurrency)
        self.fallback_score = fallback_score
        self.prompt_template = prompt_template

    async def _judge_one(
        self, session: aiohttp.ClientSession, prediction: str, ground_truth: str
    ) -> float:
        prompt = self.prompt_template.format(
            ground_truth=ground_truth.strip(),
            prediction=prediction.strip(),
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        try:
            async with self.semaphore:
                async with session.post(
                    self.url, json=payload, timeout=self.timeout
                ) as resp:
                    data = await resp.json()
            content = data["choices"][0]["message"]["content"].strip().lower()
            return 1.0 if content.startswith("yes") else 0.0
        except Exception:
            return self.fallback_score

    async def judge_batch(
        self, predictions: List[str], ground_truths: List[str]
    ) -> List[float]:
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._judge_one(session, p, g)
                for p, g in zip(predictions, ground_truths)
            ]
            return await asyncio.gather(*tasks)


def llm_judge_score(
    response: str,
    ground_truth: str,
    judge_client: Optional[LLMJudgeClient] = None,
) -> float:
    """同步包装: 单条评分 (用于单元测试 / debug)"""
    if judge_client is None:
        raise ValueError(
            "llm_judge_score requires judge_client."
        )
    return judge_client.judge_batch([response], [ground_truth])[0]
```

---

### 任务 3.7: 实现 RewardManager 主类 (`manager.py`)

**任务描述**：统一奖励管理器，对齐 verl NaiveRewardManager 接口。

**技术细节**：

```python
import torch
import asyncio
from dataclasses import dataclass
from typing import List, Optional

from .registry import get_score_fn
from .llm_judge import LLMJudgeClient


@dataclass
class RewardInput:
    """单条评分输入"""
    response: str
    ground_truth: str
    prompt: Optional[str] = None


class RewardManager:
    """统一奖励管理器"""

    def __init__(self, finetuning_args):
        self.reward_type = finetuning_args.grpo_reward_type
        self.score_mode = finetuning_args.grpo_reward_score_mode
        self.args = finetuning_args

        if self.reward_type == "llm_judge":
            self.judge_client = LLMJudgeClient(
                url=finetuning_args.grpo_llm_judge_url,
                model=finetuning_args.grpo_llm_judge_model,
                max_tokens=finetuning_args.grpo_llm_judge_max_tokens,
                temperature=finetuning_args.grpo_llm_judge_temperature,
                timeout=finetuning_args.grpo_llm_judge_timeout,
                concurrency=finetuning_args.grpo_llm_judge_concurrency,
                fallback_score=finetuning_args.grpo_llm_judge_fallback_score,
            )
            from .llm_judge import llm_judge_score
            self.score_fn = llm_judge_score
        else:
            self.score_fn = get_score_fn(self.reward_type)

    def _score_one(self, response: str, ground_truth: str) -> float:
        """单条评分 (规则类函数)"""
        if self.reward_type == "math":
            return self.score_fn(
                response, ground_truth,
                extract_mode=self.args.grpo_reward_math_extract_mode,
            )
        elif self.reward_type == "multiple_choice":
            return self.score_fn(
                response, ground_truth,
                pattern=self.args.grpo_reward_mc_pattern,
            )
        elif self.reward_type == "string_match":
            return self.score_fn(
                response, ground_truth,
                strict=self.args.grpo_reward_strict_match,
            )
        elif self.reward_type == "llm_judge":
            raise NotImplementedError(
                "_score_one 不支持 llm_judge"
            )
        else:
            return self.score_fn(response, ground_truth)

    def __call__(self, inputs: List[RewardInput]) -> torch.Tensor:
        """批量评分入口"""
        if self.reward_type == "llm_judge":
            preds = [x.response for x in inputs]
            gts = [x.ground_truth for x in inputs]
            scores = asyncio.run(
                self.judge_client.judge_batch(preds, gts)
            )
        else:
            scores = [
                self._score_one(x.response, x.ground_truth)
                for x in inputs
            ]
        return torch.tensor(scores, dtype=torch.float32)
```

---

### 任务 3.8: 扩展 `finetuning_args.py` 参数

**任务描述**：在 `finetuning_args.py` 中添加 RewardManager 相关参数。

**新增参数**：

```python
@dataclass
class FinetuningArguments:
    # ... (M01-M04 已有参数)
    
    # === RewardManager 参数 ===
    grpo_reward_type: Literal[
        "math", "multiple_choice", "string_match", "llm_judge"
    ] = "math"
    grpo_reward_score_mode: Literal["binary"] = "binary"
    
    # math
    grpo_reward_math_extract_mode: Literal["boxed", "hash", "last_number"] = "boxed"
    
    # multiple_choice
    grpo_reward_mc_pattern: str = r"(?i)\\boxed\{\s*([A-D])\s*\}|answer\s*[:：]?\s*([A-D])"
    
    # string_match
    grpo_reward_strict_match: bool = False
    
    # llm_judge
    grpo_llm_judge_url: str = ""
    grpo_llm_judge_model: str = ""
    grpo_llm_judge_max_tokens: int = 256
    grpo_llm_judge_temperature: float = 0.0
    grpo_llm_judge_timeout: int = 30
    grpo_llm_judge_concurrency: int = 16
    grpo_llm_judge_fallback_score: float = 0.0
```

---

### 任务 3.9: Trainer 集成 RewardManager (`trainer.py`)

**任务描述**：将 `reward_fn` 替换为 `reward_manager`。

**修改内容**：

```python
from .reward.manager import RewardManager, RewardInput

class CustomGRPOTrainer(Trainer):
    def __init__(self, ref_model, reward_manager, finetuning_args, **kwargs):
        super().__init__(**kwargs)
        self.ref_model = ref_model
        self.grpo_args = finetuning_args
        self.reward_manager: RewardManager = reward_manager

    def _compute_rewards(self, prompts, responses, ground_truths):
        """通过 RewardManager 批量评分"""
        inputs = [
            RewardInput(response=r, ground_truth=g, prompt=p)
            for r, g, p in zip(responses, ground_truths, prompts)
        ]
        return self.reward_manager(inputs)

    def training_step(self, model, inputs):
        prompts = inputs["input_ids"]
        ground_truths = inputs["ground_truth"]
        
        # 1. Rollout
        responses, log_probs, mask = self._rollout(model, prompts)
        
        # 2. RewardManager 评分
        response_strs = self._decode_responses(responses)
        rewards = self._compute_rewards(prompts, response_strs, ground_truths)
        
        # ... (后续步骤同 M01-M04)
```

---

### 任务 3.10: Workflow 集成 RewardManager (`workflow.py`)

**任务描述**：加入 `create_reward_manager` 工厂方法。

**修改内容**：

```python
from .reward.manager import RewardManager


def create_reward_manager(finetuning_args) -> RewardManager:
    """工厂方法: 根据 grpo_reward_type 构造 RewardManager"""
    return RewardManager(finetuning_args)


def run_grpo(model_args, data_args, training_args, finetuning_args, generating_args):
    """GRPO/DAPO/GSPO/DCPO 统一训练入口"""
    # ... (前面步骤)
    
    reward_manager = create_reward_manager(finetuning_args)

    trainer = CustomGRPOTrainer(
        model=model, ref_model=ref_model,
        reward_manager=reward_manager,
        finetuning_args=finetuning_args, args=training_args,
        train_dataset=dataset, tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model()
```

---

### 任务 3.11: 单元测试 - 4 种评分函数

**任务描述**：编写 `tests/test_reward_*.py`，验证评分函数数值正确性。

**测试示例**：

```python
# tests/test_reward_math.py
from llamafactory.train.grpo.reward.math import math_score


def test_math_score_boxed():
    response = "The answer is \\boxed{42}."
    ground_truth = "42"
    assert math_score(response, ground_truth, extract_mode="boxed") == 1.0


def test_math_score_hash():
    response = "Calculate: 3 + 4 = #### 7"
    ground_truth = "7"
    assert math_score(response, ground_truth, extract_mode="hash") == 1.0


# tests/test_reward_string_match.py
from llamafactory.train.grpo.reward.string_match import string_match_score


def test_string_match_strict():
    assert string_match_score("hello world", "hello world", strict=True) == 1.0
    assert string_match_score("hello  world", "hello world", strict=True) == 0.0


def test_string_match_relaxed():
    assert string_match_score("Hello, World!", "hello world", strict=False) == 1.0
```

---

## 4. 交付物清单

| 编号 | 交付物 | 路径 | 类型 |
|------|--------|------|------|
| D-M05-01 | Reward 目录结构 | `src/llamafactory/train/grpo/reward/` | 目录 |
| D-M05-02 | 注册表 | `src/llamafactory/train/grpo/reward/registry.py` | 代码 |
| D-M05-03 | 数学评分 | `src/llamafactory/train/grpo/reward/math.py` | 代码 |
| D-M05-04 | 多选题评分 | `src/llamafactory/train/grpo/reward/multiple_choice.py` | 代码 |
| D-M05-05 | 字符串匹配 | `src/llamafactory/train/grpo/reward/string_match.py` | 代码 |
| D-M05-06 | LLM-as-Judge | `src/llamafactory/train/grpo/reward/llm_judge.py` | 代码 |
| D-M05-07 | RewardManager 主类 | `src/llamafactory/train/grpo/reward/manager.py` | 代码 |
| D-M05-08 | Reward 参数定义 | `src/llamafactory/hparams/finetuning_args.py` (扩展) | 代码修改 |
| D-M05-09 | Trainer 集成 | `src/llamafactory/train/grpo/trainer.py` (扩展) | 代码修改 |
| D-M05-10 | Workflow 集成 | `src/llamafactory/train/grpo/workflow.py` (扩展) | 代码修改 |
| D-M05-11 | 评分函数单元测试 | `tests/test_reward_*.py` | 测试代码 |

---

## 5. 验收标准

### 5.1 功能验收

- ✅ 4 种 reward_type 均可独立运行（`grpo_reward_type=math/multiple_choice/string_match/llm_judge`）
- ✅ 切换 reward_type 不影响算法逻辑（GRPO/DAPO/GSPO/DCPO 均可正常训练）
- ✅ LLM-as-Judge 在网络异常时使用 fallback 不阻塞训练

### 5.2 代码质量验收

- ✅ 所有评分函数在 toy 样本上的数值正确性通过单元测试
- ✅ `RewardInput` 包含完整的 docstring 和类型注解

### 5.3 性能验收

- ✅ LLM-as-Judge 异步并发：16 并发下，batch_size=64 的评分耗时 < 10s
- ✅ 规则类评分（math/multiple_choice/string_match）单条耗时 < 1ms

---

## 6. 依赖关系

### 上游依赖

- **无**：本阶段可与 M01-M04 并行开发

### 下游依赖

- **M01-M04 (算法)**: 所有算法的训练都依赖 RewardManager 提供奖励信号
- **M06 (DCPO 进阶)**: DCPO 进阶特性需与 RewardManager 深度集成

### 并行依赖

- **M01-M04**: 需约定接口契约：
  - `reward_manager(inputs: List[RewardInput]) -> torch.Tensor[batch]`

---

## 7. 详细技术规范

### 7.1 评分函数接口约定

所有评分函数统一签名：
```python
def score_fn(response: str, ground_truth: str, **kwargs) -> float:
    """
    返回: float ∈ [0.0, 1.0]
    """
    pass
```

### 7.2 LLM-as-Judge 异步并发

使用 `asyncio.Semaphore(concurrency)` 控制并发数，默认 16：
```python
async with self.semaphore:
    async with session.post(...) as resp:
        ...
```

### 7.3 Fallback 机制

评分失败时返回 `fallback_score`（默认 0.0），不阻断训练：
```python
except Exception:
    return self.fallback_score
```

---

## 8. 风险与应对

### 风险 8.1: LLM-as-Judge 网络延迟

**风险描述**：外部大模型响应慢可能导致训练阻塞。

**应对策略**：
- 默认 `timeout=30s`，超时返回 fallback_score
- 使用 `concurrency=16` 异步并发，避免串行等待
- 建议用户使用本地部署的 Judge 模型（如 vLLM）

### 风险 8.2: 答案抽取失败

**风险描述**：数学答案抽取可能因格式不规范而失败。

**应对策略**：
- 提供 3 种抽取模式（boxed/hash/last_number），用户可根据数据集选择
- 抽取失败时返回 0.0，不阻断训练

### 风险 8.3: RewardManager 与 Trainer 耦合

**风险描述**：RewardManager 可能在 Trainer 中难以替换为 mock。

**应对策略**：
- 采用外部注入模式（workflow 层构造 RewardManager 后注入 Trainer）
- 单元测试中可直接构造 `RewardManager(finetuning_args)` 而无需 trainer 上下文

---

## 9. 阶段完成 Checklist

- [ ] `reward/` 目录结构创建完成
- [ ] `registry.py` 实现 `SCORE_REGISTRY` + `get_score_fn`
- [ ] `math.py` 实现 `math_score`（3 种抽取模式）
- [ ] `multiple_choice.py` 实现 `multiple_choice_score`
- [ ] `string_match.py` 实现 `string_match_score`（严格/宽松模式）
- [ ] `llm_judge.py` 实现 `LLMJudgeClient` + `llm_judge_score`
- [ ] `manager.py` 实现 `RewardManager` 主类
- [ ] `finetuning_args.py` 新增 `grpo_reward_*` 参数（至少 12 个字段）
- [ ] `trainer.py` 将 `reward_fn` 替换为 `reward_manager`
- [ ] `workflow.py` 加入 `create_reward_manager` 工厂方法
- [ ] 4 种评分函数单元测试通过
- [ ] DCPO + math 集成测试通过（小数据集 loss 下降）
- [ ] 在 `/docs/开发进度/` 创建 `M05_完成.md`，记录变更文件与验证结果

---

> **下一步**: 完成 M05 后，进入 **M06: DCPO 进阶特性**（可选，与 DAPO Dynamic Sampling 组合、Megatron 分布式训练支持等）。
