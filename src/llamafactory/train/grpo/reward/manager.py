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

"""RewardManager: unified reward computation for all GRPO-family algorithms."""

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import torch

from .registry import get_score_fn


if TYPE_CHECKING:
    from ....hparams import FinetuningArguments


@dataclass
class RewardInput:
    """Single reward computation input.

    Attributes:
        response: Decoded response string from the model.
        ground_truth: Ground truth answer/reference string.
        prompt: Optional decoded prompt string (for context-aware scoring).
    """

    response: str
    ground_truth: str
    prompt: Optional[str] = None


class RewardManager:
    """Unified reward manager that dispatches to the configured score function.

    Supports 4 scoring modes:
    - "math": Math answer scoring (boxed/hash/last_number extraction).
    - "multiple_choice": A/B/C/D multiple choice scoring.
    - "string_match": Exact or normalized string matching.
    - "llm_judge": LLM-as-Judge using external LLM API.

    The RewardManager is decoupled from the training algorithm (GRPO/DAPO/
    GSPO/DCPO) and can be combined with any of them.
    """

    def __init__(self, finetuning_args: "FinetuningArguments"):
        """Initialize the RewardManager from finetuning arguments.

        Args:
            finetuning_args: Finetuning arguments containing reward configuration.
        """
        self.reward_type = finetuning_args.grpo_reward_type
        self.score_mode = getattr(finetuning_args, "grpo_reward_score_mode", "binary")
        self.args = finetuning_args

        # Setup LLM judge if configured
        if self.reward_type == "llm_judge":
            from .llm_judge import LLMJudgeClient, llm_judge_score

            self.judge_client = LLMJudgeClient(
                url=finetuning_args.grpo_llm_judge_url,
                model=finetuning_args.grpo_llm_judge_model,
                api_key=finetuning_args.grpo_llm_judge_api_key,
                max_tokens=finetuning_args.grpo_llm_judge_max_tokens,
                temperature=finetuning_args.grpo_llm_judge_temperature,
                timeout=finetuning_args.grpo_llm_judge_timeout,
                concurrency=finetuning_args.grpo_llm_judge_concurrency,
                fallback_score=finetuning_args.grpo_llm_judge_fallback_score,
            )
            self.score_fn = llm_judge_score
        else:
            self.score_fn = get_score_fn(self.reward_type)

        # Rule-based reward support
        self.use_rule_based = getattr(
            finetuning_args, "grpo_use_rule_based_reward", False
        )
        self.rule_based_weight = getattr(
            finetuning_args, "grpo_rule_based_weight", 0.3
        )

    def _score_one(self, response: str, ground_truth: str) -> float:
        """Score a single response (rule-based scoring functions).

        Args:
            response: Model's response text.
            ground_truth: Ground truth answer.

        Returns:
            Score in [0.0, 1.0].

        Raises:
            NotImplementedError: If called with llm_judge reward type.
        """
        if self.reward_type == "math":
            return self.score_fn(
                response,
                ground_truth,
                extract_mode=self.args.grpo_reward_math_extract_mode,
            )
        elif self.reward_type == "multiple_choice":
            return self.score_fn(
                response,
                ground_truth,
                pattern=self.args.grpo_reward_mc_pattern,
            )
        elif self.reward_type == "string_match":
            return self.score_fn(
                response,
                ground_truth,
                strict=self.args.grpo_reward_strict_match,
            )
        elif self.reward_type == "llm_judge":
            raise NotImplementedError(
                "_score_one does not support llm_judge. Use __call__ for batch scoring."
            )
        else:
            return self.score_fn(response, ground_truth)

    def _score_rule_based(self, item: RewardInput) -> float:
        """Lightweight rule-based reward for format and non-empty completion.

        This optional score is intentionally generic so it can be combined with
        any main reward type without depending on task-specific code.
        """
        response = item.response.strip()
        if not response:
            return 0.0

        score = 0.5
        if self.reward_type == "math":
            if "\\boxed{" in response or "####" in response:
                score += 0.5
        elif self.reward_type == "multiple_choice":
            if any(choice in response.upper() for choice in ("A", "B", "C", "D")):
                score += 0.5
        else:
            score += 0.5

        return min(score, 1.0)

    def __call__(self, inputs: List[RewardInput]) -> torch.Tensor:
        """Batch scoring entry point.

        Args:
            inputs: List of RewardInput objects.

        Returns:
            rewards: [batch] tensor of scalar rewards.
        """
        if self.reward_type == "llm_judge":
            preds = [x.response for x in inputs]
            gts = [x.ground_truth for x in inputs]
            scores = asyncio.run(self.judge_client.judge_batch(preds, gts))
        else:
            scores = [
                self._score_one(x.response, x.ground_truth) for x in inputs
            ]

        if self.use_rule_based:
            rule_scores = [self._score_rule_based(x) for x in inputs]
            main_weight = 1.0 - self.rule_based_weight
            scores = [
                main_weight * main_score + self.rule_based_weight * rule_score
                for main_score, rule_score in zip(scores, rule_scores)
            ]

        return torch.tensor(scores, dtype=torch.float32)
