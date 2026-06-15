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

"""Multiple-choice scoring: extract A/B/C/D options and compare."""

import re
from typing import Optional


def _extract_choice(response: str, pattern: str) -> Optional[str]:
    """Extract a choice (A/B/C/D) from response text using a regex pattern.

    Args:
        response: Model's response text.
        pattern: Regex pattern with capture groups for choice letters.

    Returns:
        Uppercase choice letter, or None if no match found.
    """
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
    """Score a multiple-choice response by extracting and comparing the choice.

    Supports two common formats:
    - \\boxed{A} (LaTeX)
    - answer: A / answer：A

    Args:
        response: Model's response text.
        ground_truth: Expected choice letter (A/B/C/D).
        pattern: Custom regex pattern for extracting the choice.

    Returns:
        1.0 if extracted choice matches ground truth, 0.0 otherwise.
    """
    pred = _extract_choice(response, pattern)
    if pred is None:
        return 0.0
    return 1.0 if pred == ground_truth.strip().upper() else 0.0
