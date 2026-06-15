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

"""DAPO Overlong Reward Shaping: penalizing responses that approach the length limit."""

import torch


def apply_overlong_penalty(
    rewards: torch.Tensor,
    response_lengths: torch.Tensor,
    max_response_length: int,
    buffer_len: int = 256,
    penalty_factor: float = 1.0,
) -> torch.Tensor:
    """DAPO Overlong Reward Shaping: progressive penalty for responses near length limit.

    When a response length exceeds `max_response_length - buffer_len`, a linear
    penalty is applied. This discourages the model from generating overly long
    responses that waste computation without improving quality.

    Args:
        rewards: [batch] tensor of original scalar rewards.
        response_lengths: [batch] tensor of response lengths.
        max_response_length: Maximum allowed response length.
        buffer_len: Buffer zone before max length. Penalty starts at
            length = max_response_length - buffer_len. Default 256.
        penalty_factor: Strength of the penalty. Default 1.0.

    Returns:
        adjusted_rewards: [batch] tensor of penalty-adjusted rewards.

    Formula:
        over = max(0, length - (max_len - buffer_len))
        penalty = factor * (over / buffer_len)
        r_final = r_original - penalty
    """
    threshold = max_response_length - buffer_len
    over = (response_lengths - threshold).clamp(min=0).float()
    penalty = penalty_factor * (over / float(buffer_len))
    return rewards - penalty
