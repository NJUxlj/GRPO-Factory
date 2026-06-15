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

"""Score function registry for RewardManager."""

from typing import Callable, Dict

from .llm_judge import llm_judge_score
from .math import math_score
from .multiple_choice import multiple_choice_score
from .string_match import string_match_score


SCORE_REGISTRY: Dict[str, Callable] = {
    "math": math_score,
    "multiple_choice": multiple_choice_score,
    "string_match": string_match_score,
    "llm_judge": llm_judge_score,
}


def get_score_fn(reward_type: str) -> Callable:
    """Get a score function by reward type name.

    Args:
        reward_type: One of "math", "multiple_choice", "string_match", "llm_judge".

    Returns:
        The corresponding score function.

    Raises:
        ValueError: If reward_type is not registered.
    """
    if reward_type not in SCORE_REGISTRY:
        raise ValueError(
            f"Unknown reward_type={reward_type}. "
            f"Available: {list(SCORE_REGISTRY.keys())}"
        )
    return SCORE_REGISTRY[reward_type]
