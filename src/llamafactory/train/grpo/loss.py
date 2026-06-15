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

"""Loss functions for GRPO-family algorithms (GRPO, DAPO, GSPO, DCPO)."""

import torch


def _aggregate_loss(
    token_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
) -> torch.Tensor:
    """Aggregate token-level loss into a scalar according to the specified mode.

    Args:
        token_loss: [batch, seq_len] per-token loss values.
        mask: [batch, seq_len] mask (1 for valid tokens, 0 for padding).
        loss_agg_mode: Aggregation mode. One of:
            - "token-mean": Global mean over all valid tokens.
            - "seq-mean-token-sum": Sum tokens within each sequence, then
              mean over sequences.
            - "seq-mean-token-mean": Mean tokens within each sequence, then
              mean over sequences.
            - "otm": Only Token Mean — per-response token-mean, then
              batch-mean. Mathematically equivalent to "seq-mean-token-mean".

    Returns:
        Scalar loss tensor.
    """
    if loss_agg_mode == "token-mean":
        # Global mean over all valid tokens
        return (token_loss * mask).sum() / mask.sum().clamp(min=1)
    elif loss_agg_mode == "seq-mean-token-sum":
        # Sum tokens in each seq, then batch-mean
        return (token_loss * mask).sum(dim=-1).mean()
    elif loss_agg_mode in ("seq-mean-token-mean", "otm"):
        # Mean tokens in each seq, then batch-mean
        lengths = mask.sum(dim=-1).clamp(min=1)
        return ((token_loss * mask).sum(dim=-1) / lengths).mean()
    else:
        raise ValueError(
            f"Unknown loss_agg_mode: {loss_agg_mode}. "
            f"Expected one of: token-mean, seq-mean-token-sum, seq-mean-token-mean, otm."
        )


def compute_grpo_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_ratio: float = 0.2,
    loss_agg_mode: str = "seq-mean-token-mean",
) -> torch.Tensor:
    """Standard GRPO token-level clipped policy loss.

    Uses symmetric clip (same epsilon for lower and upper bounds).

    Args:
        log_probs: [batch, seq_len] log-probabilities from the policy model.
        ref_log_probs: [batch, seq_len] log-probabilities from the reference model.
        advantages: [batch] group-relative advantages per response.
        mask: [batch, seq_len] mask for response tokens (1 = valid).
        clip_ratio: Symmetric clip epsilon. Default 0.2.
        loss_agg_mode: Loss aggregation mode. Default "seq-mean-token-mean".

    Returns:
        Scalar loss tensor.

    Formula:
        r = exp(log_pi - log_ref)
        L = -min(r * A, clip(r, 1-ε, 1+ε) * A)
    """
    ratio = torch.exp(log_probs - ref_log_probs)  # [batch, seq_len]
    adv = advantages.unsqueeze(-1)  # [batch, 1]

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    token_loss = -torch.min(surr1, surr2)

    return _aggregate_loss(token_loss, mask, loss_agg_mode)


def compute_dapo_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.28,
) -> torch.Tensor:
    """DAPO: Asymmetric clip + token-mean aggregation.

    Key difference from GRPO: uses different clip thresholds for lower and
    upper bounds, allowing larger positive updates (ε_high > ε_low).
    Fixed to token-mean aggregation.

    Args:
        log_probs: [batch, seq_len] log-probabilities from the policy model.
        ref_log_probs: [batch, seq_len] log-probabilities from the reference model.
        advantages: [batch] group-relative advantages per response.
        mask: [batch, seq_len] mask for response tokens (1 = valid).
        clip_ratio_low: Lower clip bound ε_low. Default 0.2.
        clip_ratio_high: Upper clip bound ε_high. Default 0.28.

    Returns:
        Scalar loss tensor.

    Formula:
        r = exp(log_pi - log_ref)
        L = -min(r * A, clip(r, 1-ε_low, 1+ε_high) * A)
    """
    ratio = torch.exp(log_probs - ref_log_probs)
    adv = advantages.unsqueeze(-1)

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high) * adv
    token_loss = -torch.min(surr1, surr2)

    return _aggregate_loss(token_loss, mask, "token-mean")


def compute_gspo_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.28,
    clip_ratio_c: float = 3.0,
) -> torch.Tensor:
    """GSPO: Sequence-level importance ratio + clip.

    Key difference from GRPO/DAPO: computes importance ratio at the sequence
    level (sum of token-level log-ratios, then exp), rather than token-level.
    Applies a secondary clip ratio_c to limit extreme loss values.

    Args:
        log_probs: [batch, seq_len] log-probabilities from the policy model.
        ref_log_probs: [batch, seq_len] log-probabilities from the reference model.
        advantages: [batch] group-relative advantages per response.
        mask: [batch, seq_len] mask for response tokens (1 = valid).
        clip_ratio_low: Lower clip bound. Default 0.2.
        clip_ratio_high: Upper clip bound. Default 0.28.
        clip_ratio_c: Secondary clip bound for loss. Default 3.0.

    Returns:
        Scalar loss (batch mean).

    Formula:
        seq_log_ratio = Σ (log_pi - log_ref) * mask
        seq_ratio = exp(seq_log_ratio)
        surr1 = seq_ratio * A
        surr2 = clip(seq_ratio, 1-ε_low, 1+ε_high) * A
        surr2 = clip(surr2, -c, c)  # GSPO secondary clip
        L = -min(surr1, surr2).mean()
    """
    # Sequence-level log ratio: sum token-level log-ratios
    seq_log_ratio = ((log_probs - ref_log_probs) * mask).sum(dim=-1)
    seq_ratio = torch.exp(seq_log_ratio)  # [batch]

    # Sequence-level PPO clip
    surr1 = seq_ratio * advantages
    surr2 = torch.clamp(seq_ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high) * advantages

    # GSPO-specific secondary clip to limit extreme values
    surr2 = torch.clamp(surr2, -clip_ratio_c, clip_ratio_c)

    loss = -torch.min(surr1, surr2)
    return loss.mean()


def compute_dcpo_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_ratio_low: float = 0.16,
    clip_ratio_high: float = 0.20,
    dual_clip_ratio: float = 10.0,
    loss_agg_mode: str = "otm",
) -> torch.Tensor:
    """DCPO: DAC asymmetric clip + Dual Clip + OTM Loss aggregation.

    Three innovations over DAPO:
    1) DAC: Tighter asymmetric clip (ε_low=0.16, ε_high=0.20 vs DAPO's
       0.2/0.28).
    2) Dual Clip: For tokens with negative advantages, cap the loss at
       -dual_clip_ratio * A to prevent extreme negative gradients when
       ratio > 1.
    3) OTM Loss: Per-response token-mean → batch-mean aggregation.

    Args:
        log_probs: [batch, seq_len] log-probabilities from the policy model.
        ref_log_probs: [batch, seq_len] log-probabilities from the reference model.
        advantages: [batch] (SAS-smoothed) advantages per response.
        mask: [batch, seq_len] mask for response tokens (1 = valid).
        clip_ratio_low: DAC lower clip bound ε_low. Default 0.16.
        clip_ratio_high: DAC upper clip bound ε_high. Default 0.20.
        dual_clip_ratio: Dual clip upper bound r_max. Default 10.0.
        loss_agg_mode: Loss aggregation mode. Default "otm".

    Returns:
        Scalar loss tensor.

    Formula:
        r = exp(log_pi - log_ref)
        surr1 = r * A
        surr2 = clip(r, 1-ε_low, 1+ε_high) * A
        token_loss = -min(surr1, surr2)
        For negative A: token_loss = max(token_loss, -r_max * A)  # Dual Clip
        L = OTM_aggregate(token_loss, mask)
    """
    ratio = torch.exp(log_probs - ref_log_probs)  # [batch, seq_len]
    adv = advantages.unsqueeze(-1)  # [batch, 1]

    # (1) DAC: asymmetric clip
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high) * adv
    token_loss = -torch.min(surr1, surr2)

    # (2) Dual Clip: prevent extreme negative gradients
    if dual_clip_ratio is not None and dual_clip_ratio > 0:
        neg_adv_mask = (adv < 0).float()
        dual_loss = -dual_clip_ratio * adv
        token_loss = (
            torch.max(token_loss, dual_loss) * neg_adv_mask
            + token_loss * (1 - neg_adv_mask)
        )

    return _aggregate_loss(token_loss, mask, loss_agg_mode)
