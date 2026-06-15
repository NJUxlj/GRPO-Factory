# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LLM-as-Judge: asynchronous concurrent evaluation using external LLM."""

import asyncio
from typing import List, Optional

from ....extras import logging


logger = logging.get_logger(__name__)


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
    """Asynchronous LLM judge client for scoring responses.

    Makes concurrent HTTP requests to an external LLM API to evaluate
    whether model predictions are semantically equivalent to ground truth.
    """

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
        """Initialize the LLM judge client.

        Args:
            url: API endpoint URL (OpenAI-compatible).
            model: Model name to use for judging.
            max_tokens: Maximum tokens in judge response.
            temperature: Sampling temperature (0.0 for deterministic).
            timeout: Request timeout in seconds.
            concurrency: Maximum concurrent API requests.
            fallback_score: Score to use when API call fails.
            prompt_template: Prompt template for the judge.
        """
        self.url = url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(concurrency)
        self.fallback_score = fallback_score
        self.prompt_template = prompt_template

    async def _judge_one(
        self, session, prediction: str, ground_truth: str
    ) -> float:
        """Judge a single prediction against ground truth.

        Args:
            session: aiohttp ClientSession.
            prediction: Model prediction text.
            ground_truth: Ground truth text.

        Returns:
            1.0 if judged correct, 0.0 if incorrect or on error.
        """
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
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
                .lower()
            )
            return 1.0 if content.startswith("yes") else 0.0
        except Exception as e:
            logger.warning(f"LLM judge request failed: {e}")
            return self.fallback_score

    async def judge_batch(
        self, predictions: List[str], ground_truths: List[str]
    ) -> List[float]:
        """Judge a batch of predictions asynchronously.

        Args:
            predictions: List of model prediction strings.
            ground_truths: List of ground truth strings.

        Returns:
            List of scores (0.0 or 1.0).
        """
        try:
            import aiohttp
        except ImportError:
            raise ImportError(
                "aiohttp is required for LLM-as-Judge. "
                "Install it with: pip install aiohttp"
            )

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
    """Synchronous wrapper: score a single response using LLM judge.

    This is primarily used for unit testing and debugging. For batch scoring,
    use LLMJudgeClient.judge_batch directly.

    Args:
        response: Model prediction text.
        ground_truth: Ground truth text.
        judge_client: Pre-configured LLMJudgeClient instance.

    Returns:
        1.0 if judged correct, 0.0 otherwise.

    Raises:
        ValueError: If judge_client is None.
    """
    if judge_client is None:
        raise ValueError("llm_judge_score requires a judge_client instance.")
    return judge_client.judge_batch([response], [ground_truth])[0]
