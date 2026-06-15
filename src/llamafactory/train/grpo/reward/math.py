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

"""Math answer scoring: supports boxed, hash, and last_number extraction modes."""

import re
from typing import Optional


def _extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} pattern.

    Supports nested braces by matching balanced pairs.

    Args:
        text: Response text to search.

    Returns:
        Extracted answer string, or None if no match.
    """
    m = re.search(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    return m.group(1).strip() if m else None


def _extract_hash_answer(text: str) -> Optional[str]:
    """Extract answer from GSM8K-style #### pattern.

    Matches numeric values after the #### marker.

    Args:
        text: Response text to search.

    Returns:
        Extracted numeric string, or None if no match.
    """
    m = re.search(r"####\s*(-?\d[\d,\.]*)", text)
    if not m:
        # Also try looking for the last #### line
        lines = text.split("\n")
        for line in reversed(lines):
            m2 = re.search(r"####\s*(-?\d[\d,\.]*)", line.strip())
            if m2:
                return m2.group(1).replace(",", "").rstrip(".")
        return None
    return m.group(1).replace(",", "").rstrip(".")


def _extract_last_number(text: str) -> Optional[str]:
    """Extract the last numeric value in the text (fallback strategy).

    Args:
        text: Response text to search.

    Returns:
        Last numeric string found, or None if no numbers found.
    """
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


_EXTRACTORS = {
    "boxed": _extract_boxed_answer,
    "hash": _extract_hash_answer,
    "last_number": _extract_last_number,
}


def _normalize_answer(ans: str) -> str:
    """Normalize an answer string for comparison.

    Removes commas, spaces, trailing dots, and lowercases.

    Args:
        ans: Raw answer string.

    Returns:
        Normalized answer string.
    """
    if ans is None:
        return ""
    return ans.replace(",", "").replace(" ", "").rstrip(".").lower()


def math_score(
    response: str,
    ground_truth: str,
    extract_mode: str = "boxed",
) -> float:
    """Score a math response by extracting and comparing answers.

    Supports three extraction modes:
    - "boxed": Extract from \\boxed{...} (LaTeX format).
    - "hash": Extract from #### (GSM8K format).
    - "last_number": Extract the last number in the text (fallback).

    Args:
        response: Model's response text.
        ground_truth: Ground truth answer string.
        extract_mode: Answer extraction mode. Default "boxed".

    Returns:
        1.0 if the extracted answer matches ground truth, 0.0 otherwise.
    """
    extractor = _EXTRACTORS.get(extract_mode, _extract_boxed_answer)
    pred = extractor(response)
    if pred is None:
        return 0.0
    return 1.0 if _normalize_answer(pred) == _normalize_answer(ground_truth) else 0.0
