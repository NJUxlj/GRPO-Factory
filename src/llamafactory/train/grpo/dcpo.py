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

"""DCPO-specific utilities: DAC scheduling and hybrid loss functions."""

from typing import Tuple

import torch


def get_dac_clip_ratios(
    schedule: str,
    global_step: int,
    max_steps: int,
    clip_ratio_low: float,
    clip_ratio_high: float,
) -> Tuple[float, float]:
    """DCPO DAC threshold scheduler: supports constant and linear_decay modes.

    In linear_decay mode, ε_low starts at ε_high and linearly converges to
    ε_low over the course of training. This provides a "warm-up" effect where
    the clip is initially looser and gradually tightens.

    Args:
        schedule: "constant" or "linear_decay".
        global_step: Current training step.
        max_steps: Maximum training steps.
        clip_ratio_low: Final ε_low value.
        clip_ratio_high: ε_high value (unchanged during scheduling).

    Returns:
        (current_clip_low, current_clip_high)
    """
    if schedule == "constant":
        return clip_ratio_low, clip_ratio_high
    elif schedule == "linear_decay":
        # Linearly decay ε_low from ε_high to ε_low over max_steps
        progress = min(1.0, global_step / max(1, max_steps))
        cur_low = clip_ratio_high - (clip_ratio_high - clip_ratio_low) * progress
        return cur_low, clip_ratio_high
    else:
        return clip_ratio_low, clip_ratio_high


def compute_otm_loss(token_loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """OTM (Only Token Mean) Loss aggregation.

    Per-response token-mean → batch-mean. Mathematically equivalent to
    the "seq-mean-token-mean" aggregation mode, but preserved as a
    standalone function for paper alignment.

    Args:
        token_loss: [batch, seq_len] per-token loss values.
        mask: [batch, seq_len] mask (1 for valid tokens, 0 for padding).

    Returns:
        Scalar loss tensor.

    Formula:
        L = (1/G) * Σ_g (1/T_g) * Σ_j L_g,j * m_g,j
    """
    lengths = mask.sum(dim=-1).clamp(min=1)
    return ((token_loss * mask).sum(dim=-1) / lengths).mean()


def compute_hybrid_dcpo_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_ratio_low: float = 0.16,
    clip_ratio_high: float = 0.20,
    dual_clip_ratio: float = 10.0,
    clip_ratio_c: float = 3.0,
    loss_agg_mode: str = "otm",
    hybrid_mode: str = "token-first",
) -> torch.Tensor:
    """DCPO hybrid mode: combines token-level DAC with sequence-level clip_c.

    Two modes:
    - "token-first": Apply DAC + Dual Clip at token level, then apply
      sequence-level clip_c after aggregation.
    - "seq-first": Apply sequence-level ratio and clip_c (GSPO-style), then
      mean over batch.

    Args:
        log_probs: [batch, seq_len] log-probabilities from the policy model.
        ref_log_probs: [batch, seq_len] log-probabilities from the reference model.
        advantages: [batch] (SAS-smoothed) advantages per response.
        mask: [batch, seq_len] mask for response tokens (1 = valid).
        clip_ratio_low: DAC lower clip bound. Default 0.16.
        clip_ratio_high: DAC upper clip bound. Default 0.20.
        dual_clip_ratio: Dual clip upper bound. Default 10.0.
        clip_ratio_c: Sequence-level clip bound. Default 3.0.
        loss_agg_mode: Loss aggregation mode. Default "otm".
        hybrid_mode: "token-first" or "seq-first".

    Returns:
        Scalar loss tensor.
    """
    from .loss import _aggregate_loss

    ratio = torch.exp(log_probs - ref_log_probs)
    adv = advantages.unsqueeze(-1)

    if hybrid_mode == "token-first":
        # Token-level DAC + Dual Clip
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high) * adv
        token_loss = -torch.min(surr1, surr2)

        # Dual Clip
        if dual_clip_ratio is not None and dual_clip_ratio > 0:
            neg_adv_mask = (adv < 0).float()
            dual_loss = -dual_clip_ratio * adv
            token_loss = (
                torch.max(token_loss, dual_loss) * neg_adv_mask
                + token_loss * (1 - neg_adv_mask)
            )

        # Aggregate then apply sequence-level clip_c
        seq_loss = _aggregate_loss(token_loss, mask, loss_agg_mode)
        return torch.clamp(seq_loss, -clip_ratio_c, clip_ratio_c)

    elif hybrid_mode == "seq-first":
        # Sequence-level ratio + clip_c (GSPO-style)
        seq_log_ratio = ((log_probs - ref_log_probs) * mask).sum(dim=-1)
        seq_ratio = torch.exp(seq_log_ratio)

        surr1 = seq_ratio * advantages
        surr2 = torch.clamp(seq_ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high) * advantages
        surr2 = torch.clamp(surr2, -clip_ratio_c, clip_ratio_c)

        return -torch.min(surr1, surr2).mean()

    else:
        raise ValueError(
            f"Unknown hybrid_mode: {hybrid_mode}. Expected 'token-first' or 'seq-first'."
        )
