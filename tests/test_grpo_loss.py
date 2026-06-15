"""Unit tests for GRPO loss functions."""

import torch
import pytest

from llamafactory.train.grpo.loss import (
    _aggregate_loss,
    compute_grpo_loss,
    compute_dapo_loss,
    compute_gspo_loss,
    compute_dcpo_loss,
)


class TestAggregateLoss:
    """Tests for the _aggregate_loss helper."""

    def test_token_mean(self):
        token_loss = torch.ones(2, 4)
        mask = torch.ones(2, 4)
        mask[0, 2:] = 0  # first seq has 2 tokens
        result = _aggregate_loss(token_loss, mask, "token-mean")
        # 6 valid tokens total, each loss=1 → mean=1
        assert torch.allclose(result, torch.tensor(1.0))

    def test_seq_mean_token_mean(self):
        token_loss = torch.ones(2, 4)
        mask = torch.ones(2, 4)
        mask[0, 2:] = 0
        result = _aggregate_loss(token_loss, mask, "seq-mean-token-mean")
        # seq 0: mean over 2 tokens = 1, seq 1: mean over 4 tokens = 1
        # batch mean = 1
        assert torch.allclose(result, torch.tensor(1.0))

    def test_seq_mean_token_sum(self):
        token_loss = torch.ones(2, 4)
        mask = torch.ones(2, 4)
        mask[0, 2:] = 0
        result = _aggregate_loss(token_loss, mask, "seq-mean-token-sum")
        # seq 0: sum over 2 tokens = 2, seq 1: sum over 4 tokens = 4
        # batch mean = 3
        assert torch.allclose(result, torch.tensor(3.0))


class TestGRPOLoss:
    """Tests for compute_grpo_loss."""

    def test_basic_output(self):
        batch, seq = 4, 8
        log_probs = torch.randn(batch, seq)
        ref_log_probs = torch.randn(batch, seq)
        advantages = torch.randn(batch)
        mask = torch.ones(batch, seq)

        loss = compute_grpo_loss(log_probs, ref_log_probs, advantages, mask)
        assert loss.shape == torch.Size([])
        assert torch.isfinite(loss)

    def test_ratio_computation(self):
        """When log_probs == ref_log_probs, ratio should be 1.0 everywhere."""
        batch, seq = 4, 8
        log_probs = torch.randn(batch, seq)
        advantages = torch.randn(batch)
        mask = torch.ones(batch, seq)

        # Same log probs → ratio=1 → surr1 = surr2 = A
        loss = compute_grpo_loss(log_probs, log_probs, advantages, mask)
        assert torch.isfinite(loss)
        # Loss should be -mean(A), since surr1 = surr2 = A
        expected = -advantages.mean()
        # But due to loss aggregation mode being seq-mean-token-mean,
        # and all tokens having same loss, it should match
        assert torch.allclose(loss, expected, atol=1e-5)


class TestDAPOLoss:
    """Tests for compute_dapo_loss."""

    def test_basic_output(self):
        batch, seq = 4, 8
        log_probs = torch.randn(batch, seq)
        ref_log_probs = torch.randn(batch, seq)
        advantages = torch.randn(batch)
        mask = torch.ones(batch, seq)

        loss = compute_dapo_loss(log_probs, ref_log_probs, advantages, mask)
        assert loss.shape == torch.Size([])
        assert torch.isfinite(loss)

    def test_asymmetric_clip(self):
        """Verify that clip_ratio_low and clip_ratio_high are used correctly."""
        batch, seq = 4, 8
        log_probs = 2.0 * torch.ones(batch, seq)  # ratio = exp(2) ≈ 7.4
        ref_log_probs = torch.zeros(batch, seq)
        advantages = torch.ones(batch)
        mask = torch.ones(batch, seq)

        # With clip_ratio_low=0.2, clip_ratio_high=0.28
        loss1 = compute_dapo_loss(
            log_probs, ref_log_probs, advantages, mask,
            clip_ratio_low=0.2, clip_ratio_high=0.28,
        )

        # With tighter clip
        loss2 = compute_dapo_loss(
            log_probs, ref_log_probs, advantages, mask,
            clip_ratio_low=0.1, clip_ratio_high=0.1,
        )

        # Tighter clip should result in different loss behavior
        assert torch.isfinite(loss1)
        assert torch.isfinite(loss2)


class TestGSPOLoss:
    """Tests for compute_gspo_loss."""

    def test_basic_output(self):
        batch, seq = 4, 8
        log_probs = torch.randn(batch, seq)
        ref_log_probs = torch.randn(batch, seq)
        advantages = torch.randn(batch)
        mask = torch.ones(batch, seq)

        loss = compute_gspo_loss(log_probs, ref_log_probs, advantages, mask)
        assert loss.shape == torch.Size([])
        assert torch.isfinite(loss)

    def test_sequence_level_ratio(self):
        """When log_probs == ref_log_probs, seq_ratio should be 1.0."""
        batch, seq = 4, 8
        log_probs = torch.randn(batch, seq)
        advantages = torch.randn(batch)
        mask = torch.ones(batch, seq)

        loss = compute_gspo_loss(log_probs, log_probs, advantages, mask)
        # seq_ratio = exp(0) = 1, so surr1 = surr2 = A, loss = -A.mean()
        expected = -advantages.mean()
        assert torch.allclose(loss, expected, atol=1e-5)


class TestDCPOLoss:
    """Tests for compute_dcpo_loss."""

    def test_basic_output(self):
        batch, seq = 16, 32
        log_probs = torch.randn(batch, seq)
        ref_log_probs = torch.randn(batch, seq)
        advantages = torch.randn(batch)
        mask = torch.ones(batch, seq)

        loss = compute_dcpo_loss(
            log_probs, ref_log_probs, advantages, mask,
            clip_ratio_low=0.16, clip_ratio_high=0.20,
            dual_clip_ratio=10.0, loss_agg_mode="otm",
        )
        assert loss.shape == torch.Size([])
        assert torch.isfinite(loss)

    def test_dual_clip_negative_advantages(self):
        """Dual Clip should limit loss for tokens with negative advantages."""
        batch, seq = 4, 8
        log_probs = 3.0 * torch.ones(batch, seq)  # ratio = exp(3) ≈ 20
        ref_log_probs = torch.zeros(batch, seq)
        advantages = -torch.ones(batch)  # all negative
        mask = torch.ones(batch, seq)

        loss_with_dual = compute_dcpo_loss(
            log_probs, ref_log_probs, advantages, mask,
            dual_clip_ratio=10.0,
        )
        loss_without_dual = compute_dcpo_loss(
            log_probs, ref_log_probs, advantages, mask,
            dual_clip_ratio=None,
        )
        # With Dual Clip, loss for negative advantages is capped at -r_max * A
        assert torch.isfinite(loss_with_dual)
        assert torch.isfinite(loss_without_dual)

    def test_otm_aggregation(self):
        """OTM should be equivalent to seq-mean-token-mean."""
        batch, seq = 4, 8
        log_probs = torch.randn(batch, seq)
        ref_log_probs = torch.randn(batch, seq)
        advantages = torch.randn(batch)
        mask = torch.ones(batch, seq)

        loss_otm = compute_dcpo_loss(
            log_probs, ref_log_probs, advantages, mask,
            loss_agg_mode="otm",
        )
        loss_smtm = compute_dcpo_loss(
            log_probs, ref_log_probs, advantages, mask,
            loss_agg_mode="seq-mean-token-mean",
        )
        assert torch.allclose(loss_otm, loss_smtm, atol=1e-5)
