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

"""Advantage estimation functions for GRPO-family algorithms."""

import torch


def compute_group_relative_advantage(
    rewards: torch.Tensor,
    group_size: int,
    norm_by_std: bool = True,
) -> torch.Tensor:
    """Compute group-relative advantage by normalizing rewards within each group.

    For each group of `group_size` responses to the same prompt, the rewards
    are normalized to zero mean and (optionally) unit standard deviation.
    The resulting advantages measure how much better each response is relative
    to other responses to the same prompt.

    Args:
        rewards: [batch] tensor of scalar rewards.
        group_size: Number of responses per prompt (group).
        norm_by_std: If True, divide by group standard deviation (GRPO style).
            If False, only subtract group mean.

    Returns:
        advantages: [batch] tensor of group-relative advantages.

    Formula:
        A_i = (r_i - mean_group) / std_group   (if norm_by_std=True)
        A_i = r_i - mean_group                  (if norm_by_std=False)
    """
    num_groups = rewards.shape[0] // group_size
    if num_groups == 0:
        raise ValueError(
            f"rewards size {rewards.shape[0]} is smaller than group_size {group_size}"
        )
    rewards_grouped = rewards.view(num_groups, group_size)

    mean = rewards_grouped.mean(dim=-1, keepdim=True)
    if norm_by_std:
        std = rewards_grouped.std(dim=-1, keepdim=True).clamp(min=1e-8)
        advantages = (rewards_grouped - mean) / std
    else:
        advantages = rewards_grouped - mean

    return advantages.view(-1)


def compute_smoothed_advantage(
    rewards: torch.Tensor,
    group_size: int,
    threshold: float = 3.0,
) -> torch.Tensor:
    """DCPO SAS (Smooth Advantage Standardization) using tanh smoothing.

    Instead of hard-clipping advantages to [-threshold, threshold], this uses
    tanh to smoothly squash them. The tanh function is first-order continuous,
    which leads to more stable optimization.

    Args:
        rewards: [batch] tensor of scalar rewards.
        group_size: Number of responses per prompt (group).
        threshold: SAS clipping threshold k. Default 3.0 (covers ~99.7% of
            normal distribution).

    Returns:
        smoothed_advantages: [batch] tensor of smoothed advantages.

    Formula:
        A_smooth = tanh(A / k) * k
        where A is first computed via group-relative normalization.
    """
    # First compute standard group-relative advantages
    advantages = compute_group_relative_advantage(
        rewards, group_size, norm_by_std=True,
    )

    # SAS smoothing: tanh(adv/k) * k
    smoothed = torch.tanh(advantages / threshold) * threshold
    return smoothed
