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

"""DAPO Dynamic Sampling: filtering trivial groups for training efficiency."""

import torch

from ...extras import logging


logger = logging.get_logger(__name__)


def filter_trivial_groups(
    rewards: torch.Tensor,
    group_size: int,
    metric: str = "acc",
) -> torch.Tensor:
    """DAPO Dynamic Sampling: filter groups where all responses are correct or all wrong.

    Groups that are all-0 (all wrong) or all-1 (all correct) provide no useful
    gradient signal, so they are filtered out to save computation and improve
    training efficiency.

    Args:
        rewards: [batch] tensor of scalar rewards.
        group_size: Number of responses per prompt (group).
        metric: Filter metric type. "acc" treats rewards > 0 as correct and uses
            binary classification logic. "score" or "seq_reward" uses standard
            deviation threshold.

    Returns:
        valid_mask: [batch] boolean tensor, True for samples in valid groups,
            False for samples to be filtered.

    Formula:
        valid(group) = (sum(r > 0) > 0) AND (sum(r > 0) < |group|)
    """
    num_groups = rewards.shape[0] // group_size
    if num_groups == 0:
        return torch.ones_like(rewards, dtype=torch.bool)

    grouped = rewards.view(num_groups, group_size)

    if metric == "acc":
        binary = (grouped > 0).float()
        group_sum = binary.sum(dim=-1)
        valid = (group_sum > 0) & (group_sum < group_size)
    else:  # metric == "score" or "seq_reward"
        valid = grouped.std(dim=-1) > 1e-8

    # Expand valid mask back to original shape
    valid_mask = valid.unsqueeze(-1).expand(-1, group_size).reshape(-1)

    num_filtered = (~valid_mask).sum().item()
    if num_filtered > 0:
        logger.debug(
            f"Dynamic Sampling: filtered {num_filtered}/{rewards.shape[0]} "
            f"responses ({num_filtered / rewards.shape[0] * 100:.1f}%)"
        )

    return valid_mask
