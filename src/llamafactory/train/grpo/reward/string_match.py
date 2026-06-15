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

"""String match scoring: supports strict and relaxed (normalized) matching."""

import re
import string


_WHITESPACE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(text: str, strict: bool = False) -> str:
    """Normalize text for comparison.

    In strict mode, only extra whitespace is collapsed.
    In relaxed mode, whitespace, punctuation, and case are all normalized.

    Args:
        text: Input text string.
        strict: If True, preserve punctuation and case. If False, strip both.

    Returns:
        Normalized text string.
    """
    text = text.strip()
    if strict:
        return _WHITESPACE.sub(" ", text)
    text = text.translate(_PUNCT_TABLE)
    return _WHITESPACE.sub("", text).lower()


def string_match_score(
    response: str,
    ground_truth: str,
    strict: bool = False,
) -> float:
    """Score by string matching.

    Args:
        response: Model's response text.
        ground_truth: Ground truth text.
        strict: If True, only collapse whitespace. If False, also strip
            punctuation and lower case.

    Returns:
        1.0 if strings match, 0.0 otherwise.
    """
    return 1.0 if _normalize(response, strict) == _normalize(ground_truth, strict) else 0.0
