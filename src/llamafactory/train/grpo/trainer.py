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

"""CustomGRPOTrainer: unified trainer for GRPO/DAPO/GSPO/DCPO algorithms."""

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from transformers import Trainer
from transformers.utils import is_peft_available

from ...extras import logging
from .advantage import compute_group_relative_advantage, compute_smoothed_advantage
from .loss import compute_dapo_loss, compute_dcpo_loss, compute_grpo_loss, compute_gspo_loss
from .reward_shaping import apply_overlong_penalty
from .sampling import filter_trivial_groups


if is_peft_available():
    from peft import PeftModel


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomGRPOTrainer(Trainer):
    """Unified trainer for GRPO-family algorithms.

    Supports GRPO, DAPO, GSPO, and DCPO algorithms through the `grpo_loss_mode`
    parameter. The trainer orchestrates the full training step:
    rollout → reward → advantage → loss → update.

    All four algorithms share the same `grpo` stage and are dispatched via
    the `grpo_loss_mode` parameter, avoiding routing logic modifications.
    """

    def __init__(
        self,
        ref_model,
        reward_manager,
        finetuning_args: "FinetuningArguments",
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.ref_model = ref_model
        self.grpo_args = finetuning_args
        self.reward_manager = reward_manager
        self.tokenizer = tokenizer
        self.processor = processor

        # Build loss function dispatch table
        self.loss_fn = {
            "grpo": compute_grpo_loss,
            "dapo": compute_dapo_loss,
            "gspo": compute_gspo_loss,
            "dcpo": compute_dcpo_loss,
        }[finetuning_args.grpo_loss_mode]

        # Track DAC clip state for DCPO scheduler (updated each step)
        self._current_dac_clip = None

        # For logging metrics
        self._metrics = {}

    def _rollout(
        self, model, prompts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate responses using the policy model.

        Placeholder implementation using HF model.generate().
        Will be optimized with vLLM async sampling in future iterations.

        Args:
            model: The policy model.
            prompts: [num_prompts, prompt_len] input token IDs.

        Returns:
            responses: [num_prompts * num_generations, max_response_len] generated tokens.
            log_probs: [num_prompts * num_generations, max_response_len] token log-probabilities.
            mask: [num_prompts * num_generations, max_response_len] response token mask.
        """
        # Rollout: for each prompt, generate `grpo_num_generations` responses
        num_generations = self.grpo_args.grpo_num_generations
        prompt_len = prompts.shape[-1]

        all_responses = []
        all_log_probs = []
        all_masks = []

        model.eval()
        with torch.no_grad():
            for _ in range(num_generations):
                gen_output = model.generate(
                    input_ids=prompts,
                    max_new_tokens=self.grpo_args.grpo_max_response_length,
                    temperature=self.grpo_args.grpo_temperature,
                    top_p=self.grpo_args.grpo_top_p,
                    top_k=self.grpo_args.grpo_top_k if self.grpo_args.grpo_top_k > 0 else None,
                    do_sample=self.grpo_args.grpo_temperature > 0,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.pad_token_id
                    or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                # Extract response tokens (after prompt)
                response_ids = gen_output.sequences[:, prompt_len:]
                response_mask = (response_ids != self.tokenizer.pad_token_id).float()

                # Compute log-probs from scores
                scores = torch.stack(gen_output.scores, dim=1)  # [batch, gen_len, vocab]
                log_probs = torch.log_softmax(scores, dim=-1)
                response_log_probs = torch.gather(
                    log_probs, dim=-1, index=response_ids.unsqueeze(-1)
                ).squeeze(-1)

                all_responses.append(response_ids)
                all_log_probs.append(response_log_probs)
                all_masks.append(response_mask)

        # Concatenate all generations
        responses = torch.cat(all_responses, dim=0)
        log_probs = torch.cat(all_log_probs, dim=0)
        mask = torch.cat(all_masks, dim=0)

        return responses, log_probs, mask

    def _get_log_probs(self, model, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute token-level log-probabilities for input sequences.

        Args:
            model: The model to use (policy or reference).
            input_ids: [batch, seq_len] input token IDs.

        Returns:
            log_probs: [batch, seq_len] token log-probabilities.
        """
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = (
                outputs.logits[:, :-1, :]
                if hasattr(outputs, "logits")
                else outputs[:, :-1, :]
            )
            log_probs = torch.log_softmax(logits, dim=-1)
            # Gather log-probs for the actual next tokens
            target_ids = input_ids[:, 1:]
            gathered = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1))
            return gathered.squeeze(-1)

    def _compute_rewards(
        self, prompts: torch.Tensor, responses: List[str], ground_truths: List[str]
    ) -> torch.Tensor:
        """Compute rewards using the RewardManager.

        Args:
            prompts: [batch] prompt token IDs.
            responses: List of decoded response strings.
            ground_truths: List of ground truth strings.

        Returns:
            rewards: [batch] tensor of scalar rewards.
        """
        from .reward.manager import RewardInput

        inputs = [
            RewardInput(
                response=r,
                ground_truth=g,
                prompt=self.tokenizer.decode(p, skip_special_tokens=True)
                if isinstance(p, torch.Tensor)
                else str(p),
            )
            for r, g, p in zip(responses, ground_truths, prompts)
        ]
        return self.reward_manager(inputs)

    def _decode_responses(self, responses: torch.Tensor) -> List[str]:
        """Decode response token IDs to strings."""
        return self.tokenizer.batch_decode(
            responses, skip_special_tokens=True
        )

    def _compute_kl(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence between policy and reference distributions.

        Args:
            log_probs: [batch, seq_len] policy log-probabilities.
            ref_log_probs: [batch, seq_len] reference log-probabilities.
            mask: [batch, seq_len] valid token mask.

        Returns:
            kl: Scalar KL divergence value.

        Supported kl_type:
            - "kl": Standard KL divergence E[log(p) - log(q)].
            - "abs": Mean absolute difference |log(p) - log(q)|.
            - "mse": Mean squared error (log(p) - log(q))^2.
            - "low_var_kl": Low-variance KL estimator.
            - "full": KL with softmax over full vocabulary.
        """
        kl_type = self.grpo_args.grpo_kl_type

        if kl_type == "kl":
            kl = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1
        elif kl_type == "abs":
            kl = torch.abs(log_probs - ref_log_probs)
        elif kl_type == "mse":
            diff = log_probs - ref_log_probs
            kl = 0.5 * diff * diff
        elif kl_type == "low_var_kl":
            diff = log_probs - ref_log_probs
            kl = diff * diff * 0.5
        elif kl_type == "full":
            diff = log_probs - ref_log_probs
            kl = torch.exp(diff) * diff
        else:
            raise ValueError(f"Unknown kl_type: {kl_type}")

        return (kl * mask).sum() / mask.sum().clamp(min=1)

    def _get_loss_kwargs(self) -> dict:
        """Get algorithm-specific keyword arguments for the loss function."""
        mode = self.grpo_args.grpo_loss_mode

        if mode == "grpo":
            return {
                "clip_ratio": self.grpo_args.grpo_clip_ratio,
                "loss_agg_mode": self.grpo_args.grpo_loss_agg_mode,
            }
        elif mode == "dapo":
            return {
                "clip_ratio_low": self.grpo_args.dapo_clip_ratio_low,
                "clip_ratio_high": self.grpo_args.dapo_clip_ratio_high,
            }
        elif mode == "gspo":
            return {
                "clip_ratio_low": self.grpo_args.dapo_clip_ratio_low,
                "clip_ratio_high": self.grpo_args.dapo_clip_ratio_high,
                "clip_ratio_c": self.grpo_args.gspo_clip_ratio_c,
            }
        elif mode == "dcpo":
            # Use DAC scheduler output if available, otherwise fall back to config
            if getattr(self, "_current_dac_clip", None) is not None:
                clip_low, clip_high = self._current_dac_clip
            else:
                clip_low = self.grpo_args.dcpo_clip_ratio_low
                clip_high = self.grpo_args.dcpo_clip_ratio_high
            return {
                "clip_ratio_low": clip_low,
                "clip_ratio_high": clip_high,
                "dual_clip_ratio": self.grpo_args.dcpo_dual_clip_ratio,
                "loss_agg_mode": self.grpo_args.dcpo_loss_agg_mode,
            }
        else:
            raise ValueError(f"Unknown grpo_loss_mode: {mode}")

    def _get_dac_clip_ratios(self) -> Tuple[float, float]:
        """DCPO DAC threshold scheduler."""
        from .dcpo import get_dac_clip_ratios

        return get_dac_clip_ratios(
            schedule=self.grpo_args.dcpo_clip_schedule,
            global_step=self.state.global_step,
            max_steps=self.state.max_steps,
            clip_ratio_low=self.grpo_args.dcpo_clip_ratio_low,
            clip_ratio_high=self.grpo_args.dcpo_clip_ratio_high,
        )

    def training_step(
        self, model, inputs
    ) -> torch.Tensor:
        """Single GRPO training step: rollout → reward → advantage → loss.

        This method orchestrates the complete GRPO training pipeline:

        1. Rollout: Generate n responses per prompt using the policy model.
        2. Reward: Score each response via the RewardManager.
        3. Overlong Shaping (DAPO/DCPO): Apply length penalty if enabled.
        4. Advantage: Compute group-relative advantages (with SAS for DCPO).
        5. Dynamic Sampling (DAPO/DCPO): Filter trivial groups.
        6. Ref Log Probs: Get reference model's log-probabilities.
        7. Policy Loss: Compute algorithm-specific loss.
        8. KL Loss: Add KL divergence penalty if enabled.

        Args:
            model: The policy model (unwrapped by HuggingFace Trainer).
            inputs: Batch dict with "input_ids" (prompts) and "ground_truth".

        Returns:
            loss: Scalar training loss.
        """
        prompts = inputs["input_ids"]
        ground_truths = inputs.get("ground_truth", [""] * len(prompts))

        # Step 1: Rollout — generate responses
        responses, log_probs, mask = self._rollout(model, prompts)

        # Step 2: Reward — score responses
        response_strs = self._decode_responses(responses)
        rewards = self._compute_rewards(prompts, response_strs, ground_truths)

        # Step 3: DAPO/DCPO Overlong Reward Shaping
        if self.grpo_args.dapo_overlong_shaping:
            lengths = mask.sum(dim=-1)
            rewards = apply_overlong_penalty(
                rewards,
                lengths,
                self.grpo_args.grpo_max_response_length,
                self.grpo_args.dapo_overlong_buffer_len,
                self.grpo_args.dapo_overlong_penalty_factor,
            )
            self._metrics["overlong_penalty"] = True

        # Step 4: Compute advantages
        if self.grpo_args.grpo_loss_mode == "dcpo" and self.grpo_args.dcpo_sas_enable:
            advantages = compute_smoothed_advantage(
                rewards,
                self.grpo_args.grpo_num_generations,
                threshold=self.grpo_args.dcpo_sas_threshold,
            )
        else:
            advantages = compute_group_relative_advantage(
                rewards,
                self.grpo_args.grpo_num_generations,
                self.grpo_args.grpo_norm_adv_by_std,
            )

        # Step 5: DAPO/DCPO Dynamic Sampling — filter trivial groups
        if self.grpo_args.dapo_dynamic_sampling:
            valid_mask = filter_trivial_groups(
                rewards,
                self.grpo_args.grpo_num_generations,
                self.grpo_args.dapo_filter_metric,
            )
            advantages = advantages * valid_mask.float()

        # Step 6: DCPO DAC threshold scheduling
        if self.grpo_args.grpo_loss_mode == "dcpo":
            self._current_dac_clip = self._get_dac_clip_ratios()

        # Step 7: Compute reference log-probabilities
        with torch.no_grad():
            response_full = torch.cat([prompts, responses], dim=-1)  # full sequences
            ref_log_probs = self._get_log_probs(
                self.ref_model, response_full
            )[:, prompts.shape[-1]:]  # keep only response portion

        # Step 8: Compute policy loss via algorithm-specific loss function
        loss = self.loss_fn(
            log_probs,
            ref_log_probs,
            advantages,
            mask,
            **self._get_loss_kwargs(),
        )

        # Step 9: Add KL divergence loss
        if self.grpo_args.grpo_use_kl_loss:
            kl = self._compute_kl(log_probs, ref_log_probs, mask)
            loss = loss + self.grpo_args.grpo_kl_coef * kl
            self._metrics["kl"] = kl.item()

        # Step 10: Add entropy bonus (if enabled)
        if self.grpo_args.grpo_entropy_coeff > 0:
            log_p = torch.log_softmax(log_probs, dim=-1)
            entropy = -(log_p.exp() * log_p).sum(dim=-1)
            entropy_loss = -(entropy * mask).sum() / mask.sum().clamp(min=1)
            loss = loss + self.grpo_args.grpo_entropy_coeff * entropy_loss
            self._metrics["entropy"] = entropy_loss.item()

        self._metrics["loss"] = loss.item()
        self._metrics["reward_mean"] = rewards.mean().item()

        return loss

    def log(self, logs: dict) -> None:
        """Override to inject GRPO-specific metrics."""
        logs.update(self._metrics)
        self._metrics = {}
        super().log(logs)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Save model checkpoint."""
        super()._save(output_dir, state_dict)
