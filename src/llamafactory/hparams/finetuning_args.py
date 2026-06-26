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

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class FreezeArguments:
    r"""Arguments pertaining to the freeze (partial-parameter) training."""

    freeze_trainable_layers: int = field(
        default=2,
        metadata={
            "help": (
                "The number of trainable layers for freeze (partial-parameter) fine-tuning. "
                "Positive numbers mean the last n layers are set as trainable, "
                "negative numbers mean the first n layers are set as trainable."
            )
        },
    )
    freeze_trainable_modules: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of trainable modules for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the available modules."
            )
        },
    )
    freeze_extra_modules: str | None = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from hidden layers to be set as trainable "
                "for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules."
            )
        },
    )


@dataclass
class LoraArguments:
    r"""Arguments pertaining to the LoRA training."""

    additional_target: str | None = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from LoRA layers to be set as trainable "
                "and saved in the final checkpoint. "
                "Use commas to separate multiple modules."
            )
        },
    )
    lora_alpha: int | None = field(
        default=None,
        metadata={"help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2)."},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."},
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."},
    )
    lora_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of target modules to apply LoRA. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    loraplus_lr_ratio: float | None = field(
        default=None,
        metadata={"help": "LoRA plus learning rate ratio (lr_B / lr_A)."},
    )
    loraplus_lr_embedding: float = field(
        default=1e-6,
        metadata={"help": "LoRA plus learning rate for lora embedding layers."},
    )
    use_rslora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the rank stabilization scaling factor for LoRA layer."},
    )
    use_dora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the weight-decomposed lora method (DoRA)."},
    )
    pissa_init: bool = field(
        default=False,
        metadata={"help": "Whether or not to initialize a PiSSA adapter."},
    )
    pissa_iter: int = field(
        default=16,
        metadata={"help": "The number of iteration steps performed by FSVD in PiSSA. Use -1 to disable it."},
    )
    pissa_convert: bool = field(
        default=False,
        metadata={"help": "Whether or not to convert the PiSSA adapter to a normal LoRA adapter."},
    )
    create_new_adapter: bool = field(
        default=False,
        metadata={"help": "Whether or not to create a new adapter with randomly initialized weight."},
    )


@dataclass
class OFTArguments:
    r"""Arguments pertaining to the OFT training."""

    additional_target: str | None = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from LoRA layers to be set as trainable "
                "and saved in the final checkpoint. "
                "Use commas to separate multiple modules."
            )
        },
    )
    module_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the OFT fine-tuning."},
    )
    oft_rank: int = field(
        default=0,
        metadata={"help": "The intrinsic dimension for OFT fine-tuning."},
    )
    oft_block_size: int = field(
        default=32,
        metadata={"help": "The intrinsic dimension for OFT fine-tuning."},
    )
    oft_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of target modules to apply OFT. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    create_new_adapter: bool = field(
        default=False,
        metadata={"help": "Whether or not to create a new adapter with randomly initialized weight."},
    )


@dataclass
class RLHFArguments:
    r"""Arguments pertaining to the PPO, DPO and KTO training."""

    pref_beta: float = field(
        default=0.1,
        metadata={"help": "The beta parameter in the preference loss."},
    )
    pref_ftx: float = field(
        default=0.0,
        metadata={"help": "The supervised fine-tuning loss coefficient in DPO training."},
    )
    pref_bco_weight: float = field(
        default=0.0,
        metadata={"help": "The Binary Classifier Optimization coefficient in DPO training."},
    )
    pref_loss: Literal["sigmoid", "hinge", "ipo", "kto_pair", "orpo", "simpo"] = field(
        default="sigmoid",
        metadata={"help": "The type of DPO loss to use."},
    )
    dpo_label_smoothing: float = field(
        default=0.0,
        metadata={"help": "The robust DPO label smoothing parameter in cDPO that should be between 0 and 0.5."},
    )
    kto_chosen_weight: float = field(
        default=1.0,
        metadata={"help": "The weight factor of the desirable losses in KTO training."},
    )
    kto_rejected_weight: float = field(
        default=1.0,
        metadata={"help": "The weight factor of the undesirable losses in KTO training."},
    )
    simpo_gamma: float = field(
        default=0.5,
        metadata={"help": "The target reward margin term in SimPO loss."},
    )
    ppo_buffer_size: int = field(
        default=1,
        metadata={"help": "The number of mini-batches to make experience buffer in a PPO optimization step."},
    )
    ppo_epochs: int = field(
        default=4,
        metadata={"help": "The number of epochs to perform in a PPO optimization step."},
    )
    ppo_score_norm: bool = field(
        default=False,
        metadata={"help": "Use score normalization in PPO training."},
    )
    ppo_target: float = field(
        default=6.0,
        metadata={"help": "Target KL value for adaptive KL control in PPO training."},
    )
    ppo_whiten_rewards: bool = field(
        default=False,
        metadata={"help": "Whiten the rewards before compute advantages in PPO training."},
    )
    ref_model: str | None = field(
        default=None,
        metadata={"help": "Path to the reference model used for the PPO or DPO training."},
    )
    ref_model_adapters: str | None = field(
        default=None,
        metadata={"help": "Path to the adapters of the reference model."},
    )
    ref_model_quantization_bit: int | None = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reference model."},
    )
    reward_model: str | None = field(
        default=None,
        metadata={"help": "Path to the reward model used for the PPO training."},
    )
    reward_model_adapters: str | None = field(
        default=None,
        metadata={"help": "Path to the adapters of the reward model."},
    )
    reward_model_quantization_bit: int | None = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reward model."},
    )
    reward_model_type: Literal["lora", "full", "api"] = field(
        default="lora",
        metadata={"help": "The type of the reward model in PPO training. Lora model only supports lora training."},
    )
    ld_alpha: float | None = field(
        default=None,
        metadata={
            "help": (
                "Alpha parameter from the LD-DPO paper, which controls the weighting of"
                " the verbose token log-probabilities in responses."
            )
        },
    )


@dataclass
class GaloreArguments:
    r"""Arguments pertaining to the GaLore algorithm."""

    use_galore: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the gradient low-Rank projection (GaLore)."},
    )
    galore_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of modules to apply GaLore. Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    galore_rank: int = field(
        default=16,
        metadata={"help": "The rank of GaLore gradients."},
    )
    galore_update_interval: int = field(
        default=200,
        metadata={"help": "Number of steps to update the GaLore projection."},
    )
    galore_scale: float = field(
        default=2.0,
        metadata={"help": "GaLore scaling coefficient."},
    )
    galore_proj_type: Literal["std", "reverse_std", "right", "left", "full"] = field(
        default="std",
        metadata={"help": "Type of GaLore projection."},
    )
    galore_layerwise: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable layer-wise update to further save memory."},
    )


@dataclass
class ApolloArguments:
    r"""Arguments pertaining to the APOLLO algorithm."""

    use_apollo: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the APOLLO optimizer."},
    )
    apollo_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of modules to apply APOLLO. Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    apollo_rank: int = field(
        default=16,
        metadata={"help": "The rank of APOLLO gradients."},
    )
    apollo_update_interval: int = field(
        default=200,
        metadata={"help": "Number of steps to update the APOLLO projection."},
    )
    apollo_scale: float = field(
        default=32.0,
        metadata={"help": "APOLLO scaling coefficient."},
    )
    apollo_proj: Literal["svd", "random"] = field(
        default="random",
        metadata={"help": "Type of APOLLO low-rank projection algorithm (svd or random)."},
    )
    apollo_proj_type: Literal["std", "right", "left"] = field(
        default="std",
        metadata={"help": "Type of APOLLO projection."},
    )
    apollo_scale_type: Literal["channel", "tensor"] = field(
        default="channel",
        metadata={"help": "Type of APOLLO scaling (channel or tensor)."},
    )
    apollo_layerwise: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable layer-wise update to further save memory."},
    )
    apollo_scale_front: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the norm-growth limiter in front of gradient scaling."},
    )


@dataclass
class BAdamArgument:
    r"""Arguments pertaining to the BAdam optimizer."""

    use_badam: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the BAdam optimizer."},
    )
    badam_mode: Literal["layer", "ratio"] = field(
        default="layer",
        metadata={"help": "Whether to use layer-wise or ratio-wise BAdam optimizer."},
    )
    badam_start_block: int | None = field(
        default=None,
        metadata={"help": "The starting block index for layer-wise BAdam."},
    )
    badam_switch_mode: Literal["ascending", "descending", "random", "fixed"] | None = field(
        default="ascending",
        metadata={"help": "the strategy of picking block to update for layer-wise BAdam."},
    )
    badam_switch_interval: int | None = field(
        default=50,
        metadata={
            "help": "Number of steps to update the block for layer-wise BAdam. Use -1 to disable the block update."
        },
    )
    badam_update_ratio: float = field(
        default=0.05,
        metadata={"help": "The ratio of the update for ratio-wise BAdam."},
    )
    badam_mask_mode: Literal["adjacent", "scatter"] = field(
        default="adjacent",
        metadata={
            "help": (
                "The mode of the mask for BAdam optimizer. "
                "`adjacent` means that the trainable parameters are adjacent to each other, "
                "`scatter` means that trainable parameters are randomly choosed from the weight."
            )
        },
    )
    badam_verbose: int = field(
        default=0,
        metadata={
            "help": (
                "The verbosity level of BAdam optimizer. "
                "0 for no print, 1 for print the block prefix, 2 for print trainable parameters."
            )
        },
    )


@dataclass
class SwanLabArguments:
    use_swanlab: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the SwanLab (an experiment tracking and visualization tool)."},
    )
    swanlab_project: str | None = field(
        default="llamafactory",
        metadata={"help": "The project name in SwanLab."},
    )
    swanlab_workspace: str | None = field(
        default=None,
        metadata={"help": "The workspace name in SwanLab."},
    )
    swanlab_run_name: str | None = field(
        default=None,
        metadata={"help": "The experiment name in SwanLab."},
    )
    swanlab_mode: Literal["cloud", "local"] = field(
        default="cloud",
        metadata={"help": "The mode of SwanLab."},
    )
    swanlab_api_key: str | None = field(
        default=None,
        metadata={"help": "The API key for SwanLab."},
    )
    swanlab_logdir: str | None = field(
        default=None,
        metadata={"help": "The log directory for SwanLab."},
    )
    swanlab_lark_webhook_url: str | None = field(
        default=None,
        metadata={"help": "The Lark(飞书) webhook URL for SwanLab."},
    )
    swanlab_lark_secret: str | None = field(
        default=None,
        metadata={"help": "The Lark(飞书) secret for SwanLab."},
    )


@dataclass
class GRPOArguments:
    r"""Arguments pertaining to GRPO/DAPO/GSPO/DCPO training."""

    # === Shared GRPO parameters ===
    grpo_loss_mode: Literal["grpo", "dapo", "gspo", "dcpo"] = field(
        default="grpo",
        metadata={"help": "Which GRPO-family algorithm to use. Affects loss computation and training logic."},
    )
    grpo_num_generations: int = field(
        default=8,
        metadata={"help": "Number of responses to generate per prompt during rollout (G in GRPO paper)."},
    )
    grpo_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for response generation during rollout."},
    )
    grpo_top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p (nucleus) sampling parameter for rollout."},
    )
    grpo_top_k: int = field(
        default=-1,
        metadata={"help": "Top-k sampling parameter for rollout (-1 = disabled)."},
    )
    grpo_max_response_length: int = field(
        default=2048,
        metadata={"help": "Maximum number of tokens to generate per response."},
    )
    grpo_ppo_epochs: int = field(
        default=1,
        metadata={"help": "Number of PPO epochs per batch of rollouts."},
    )
    grpo_mini_batch_size: int = field(
        default=8,
        metadata={"help": "Mini-batch size for PPO updates within a rollout batch."},
    )
    grpo_grad_clip: float = field(
        default=1.0,
        metadata={"help": "Gradient clipping value for GRPO training."},
    )
    grpo_use_kl_loss: bool = field(
        default=True,
        metadata={"help": "Whether to add KL divergence penalty to the loss."},
    )
    grpo_kl_coef: float = field(
        default=0.001,
        metadata={"help": "Coefficient for KL divergence penalty (higher for GSPO, ~0.1)."},
    )
    grpo_kl_type: Literal["kl", "abs", "mse", "low_var_kl", "full"] = field(
        default="kl",
        metadata={"help": "Type of KL divergence estimator to use."},
    )
    grpo_entropy_coeff: float = field(
        default=0.0,
        metadata={"help": "Entropy bonus coefficient (0 = disabled)."},
    )
    grpo_norm_adv_by_std: bool = field(
        default=True,
        metadata={"help": "Whether to normalize advantages by group standard deviation."},
    )

    # === GRPO-specific parameters ===
    grpo_clip_ratio: float = field(
        default=0.2,
        metadata={"help": "Symmetric clip ratio for GRPO policy loss (ε)."},
    )
    grpo_loss_agg_mode: Literal["token-mean", "seq-mean-token-sum", "seq-mean-token-mean"] = field(
        default="seq-mean-token-mean",
        metadata={"help": "Loss aggregation mode for GRPO."},
    )

    # === DAPO-specific parameters ===
    dapo_clip_ratio_low: float = field(
        default=0.2,
        metadata={"help": "DAPO asymmetric clip lower bound (ε_low)."},
    )
    dapo_clip_ratio_high: float = field(
        default=0.28,
        metadata={"help": "DAPO asymmetric clip upper bound (ε_high)."},
    )
    dapo_dynamic_sampling: bool = field(
        default=True,
        metadata={"help": "Whether to filter trivial (all-0 or all-1) groups."},
    )
    dapo_filter_metric: Literal["acc", "score", "seq_reward"] = field(
        default="acc",
        metadata={"help": "Metric used for filtering trivial groups."},
    )
    dapo_max_gen_batches: int = field(
        default=10,
        metadata={"help": "Maximum retry batches for dynamic sampling."},
    )
    dapo_overlong_shaping: bool = field(
        default=True,
        metadata={"help": "Whether to apply overlong reward penalty."},
    )
    dapo_overlong_buffer_len: int = field(
        default=256,
        metadata={"help": "Buffer length before max_response_length where penalty begins."},
    )
    dapo_overlong_penalty_factor: float = field(
        default=1.0,
        metadata={"help": "Strength of overlong penalty (higher = stronger penalty)."},
    )

    # === GSPO-specific parameters ===
    gspo_clip_ratio_c: float = field(
        default=3.0,
        metadata={"help": "GSPO secondary clip bound for loss (c parameter)."},
    )
    gspo_use_megatron: bool = field(
        default=False,
        metadata={"help": "Whether to use Megatron distributed strategy (reserved for future)."},
    )

    # === DCPO-specific parameters (DAPO's further improvement) ===
    # 1) DAC (Dynamic-Adaptive Clipping)
    dcpo_clip_ratio_low: float = field(
        default=0.16,
        metadata={"help": "DCPO DAC lower clip bound ε_low (tighter than DAPO's 0.2)."},
    )
    dcpo_clip_ratio_high: float = field(
        default=0.20,
        metadata={"help": "DCPO DAC upper clip bound ε_high (tighter than DAPO's 0.28)."},
    )
    dcpo_dual_clip_ratio: float = field(
        default=10.0,
        metadata={"help": "DCPO Dual Clip upper bound r_max (paper: r_max=10)."},
    )

    # 2) SAS (Smooth Advantage Standardization)
    dcpo_sas_enable: bool = field(
        default=True,
        metadata={"help": "Whether to use SAS tanh smoothing for advantages."},
    )
    dcpo_sas_threshold: float = field(
        default=3.0,
        metadata={"help": "SAS clipping threshold k (paper uses k=3)."},
    )

    # 3) OTM Loss
    dcpo_loss_agg_mode: Literal["otm", "token-mean", "seq-mean-token-mean"] = field(
        default="otm",
        metadata={"help": "Loss aggregation mode for DCPO."},
    )

    # Optional: DAC scheduler
    dcpo_clip_schedule: Literal["constant", "linear_decay"] = field(
        default="constant",
        metadata={"help": "DAC clip ratio schedule (constant or linear_decay)."},
    )

    # === RewardManager parameters ===
    grpo_reward_type: Literal["math", "multiple_choice", "string_match", "llm_judge"] = field(
        default="math",
        metadata={"help": "Type of reward scoring function to use."},
    )
    grpo_reward_score_mode: Literal["binary"] = field(
        default="binary",
        metadata={"help": "Score mode for reward functions."},
    )

    # Math scoring
    grpo_reward_math_extract_mode: Literal["boxed", "hash", "last_number"] = field(
        default="boxed",
        metadata={"help": "Answer extraction mode for math scoring."},
    )

    # Multiple choice scoring
    grpo_reward_mc_pattern: str = field(
        default=r"(?i)\\boxed\{\s*([A-D])\s*\}|answer\s*[:：]?\s*([A-D])",
        metadata={"help": "Regex pattern for multiple choice answer extraction."},
    )

    # String match scoring
    grpo_reward_strict_match: bool = field(
        default=False,
        metadata={"help": "If True, only collapse whitespace; if False, also strip punctuation and lowercase."},
    )

    # LLM-as-Judge scoring
    grpo_llm_judge_url: str = field(
        default="",
        metadata={"help": "API URL for LLM-as-Judge (OpenAI-compatible endpoint)."},
    )
    grpo_llm_judge_model: str = field(
        default="",
        metadata={"help": "Model name for LLM-as-Judge."},
    )
    grpo_llm_judge_api_key: str = field(
        default="",
        metadata={"help": "API key for LLM-as-Judge. Leave empty for local endpoints without authentication."},
    )
    grpo_llm_judge_max_tokens: int = field(
        default=256,
        metadata={"help": "Max tokens for judge response."},
    )
    grpo_llm_judge_temperature: float = field(
        default=0.0,
        metadata={"help": "Temperature for judge model (0.0 = deterministic)."},
    )
    grpo_llm_judge_timeout: int = field(
        default=30,
        metadata={"help": "Request timeout in seconds for LLM judge."},
    )
    grpo_llm_judge_concurrency: int = field(
        default=16,
        metadata={"help": "Maximum concurrent requests to LLM judge API."},
    )
    grpo_llm_judge_fallback_score: float = field(
        default=0.0,
        metadata={"help": "Fallback score when LLM judge API call fails."},
    )

    # Rule-based reward
    grpo_use_rule_based_reward: bool = field(
        default=False,
        metadata={"help": "Whether to combine rule-based reward with main scoring."},
    )
    grpo_rule_based_weight: float = field(
        default=0.3,
        metadata={"help": "Weight of rule-based reward when combining with main score."},
    )

    # DCPO hybrid mode (M06 advanced)
    dcpo_hybrid_mode: Literal["token-first", "seq-first"] = field(
        default="token-first",
        metadata={"help": "DCPO hybrid mode for combining token-level and sequence-level clipping."},
    )
    dcpo_hybrid_enable: bool = field(
        default=False,
        metadata={"help": "Whether to use DCPO hybrid mode (DAC + GSPO clip_c)."},
    )


@dataclass
class FinetuningArguments(
    GRPOArguments,
    SwanLabArguments,
    BAdamArgument,
    ApolloArguments,
    GaloreArguments,
    RLHFArguments,
    LoraArguments,
    OFTArguments,
    FreezeArguments,
):
    r"""Arguments pertaining to which techniques we are going to fine-tuning with."""

    pure_bf16: bool = field(
        default=False,
        metadata={"help": "Whether or not to train model in purely bf16 precision (without AMP)."},
    )
    stage: Literal["pt", "sft", "rm", "ppo", "dpo", "kto", "grpo"] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training."},
    )
    finetuning_type: Literal["lora", "oft", "freeze", "full"] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."},
    )
    use_llama_pro: bool = field(
        default=False,
        metadata={"help": "Whether or not to make only the parameters in the expanded blocks trainable."},
    )
    use_adam_mini: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the Adam-mini optimizer."},
    )
    use_mca: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to use MCA (Megatron Core Adapter) training. "
                "Controlled by USE_MCA environment variable."
            )
        },
    )
    use_hyper_parallel: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to use HyperParallel distributed training backend (FSDP/TP). "
                "Only supported for the 'sft' stage with full fine-tuning."
            )
        },
    )
    hyper_parallel_args: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to a JSON file containing HyperParallel strategy arguments "
                "(e.g., tp_size, param_dtype). Used when use_hyper_parallel=True."
            )
        },
    )
    use_muon: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the Muon optimizer."},
    )
    use_dft_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the DFT loss."},
    )
    use_asft_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the ASFT loss."},
    )
    asft_alpha: float = field(
        default=0.1,
        metadata={"help": "The alpha parameter for ASFT loss to control the power of adaptive weight."},
    )
    use_eaft_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the EAFT loss."},
    )
    eaft_alpha: float = field(
        default=1.0,
        metadata={"help": "The alpha parameter for EAFT loss to control the power of adaptive weight."},
    )
    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "Whether ot not to freeze the vision tower in MLLM training."},
    )
    freeze_multi_modal_projector: bool = field(
        default=True,
        metadata={"help": "Whether or not to freeze the multi modal projector in MLLM training."},
    )
    freeze_language_model: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the language model in MLLM training."},
    )
    compute_accuracy: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute the token-level accuracy at evaluation."},
    )
    disable_shuffling: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable the shuffling of the training set."},
    )
    early_stopping_steps: int | None = field(
        default=None,
        metadata={"help": "Number of steps to stop training if the `metric_for_best_model` does not improve."},
    )
    plot_loss: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the training loss curves."},
    )
    include_effective_tokens_per_second: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute effective tokens per second."},
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.freeze_trainable_modules: list[str] = split_arg(self.freeze_trainable_modules)
        self.freeze_extra_modules: list[str] | None = split_arg(self.freeze_extra_modules)
        self.lora_alpha: int = self.lora_alpha or self.lora_rank * 2
        self.lora_target: list[str] = split_arg(self.lora_target)
        self.oft_target: list[str] = split_arg(self.oft_target)
        self.additional_target: list[str] | None = split_arg(self.additional_target)
        self.galore_target: list[str] = split_arg(self.galore_target)
        self.apollo_target: list[str] = split_arg(self.apollo_target)
        self.use_ref_model = self.stage == "dpo" and self.pref_loss not in ["orpo", "simpo"]

        assert self.finetuning_type in ["lora", "oft", "freeze", "full"], "Invalid fine-tuning method."
        assert self.ref_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."
        assert self.reward_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."

        if self.stage == "ppo" and self.reward_model is None:
            raise ValueError("`reward_model` is necessary for PPO training.")

        if self.stage == "ppo" and self.reward_model_type == "lora" and self.finetuning_type != "lora":
            raise ValueError("`reward_model_type` cannot be lora for Freeze/Full PPO training.")

        if self.stage == "ppo" and self.reward_model_type == "oft" and self.finetuning_type != "oft":
            raise ValueError("`reward_model_type` cannot be oft for Freeze/Full PPO training.")

        if self.stage == "grpo":
            if self.grpo_num_generations < 2:
                raise ValueError("`grpo_num_generations` must be at least 2 for group-relative advantage.")
            if self.grpo_reward_type == "llm_judge" and not self.grpo_llm_judge_url:
                raise ValueError("`grpo_llm_judge_url` must be set when grpo_reward_type='llm_judge'.")
            if self.grpo_reward_type == "llm_judge" and not self.grpo_llm_judge_model:
                raise ValueError("`grpo_llm_judge_model` must be set when grpo_reward_type='llm_judge'.")
            if self.grpo_loss_mode not in ("grpo", "dapo", "gspo", "dcpo"):
                raise ValueError(f"Unknown grpo_loss_mode: {self.grpo_loss_mode}.")

        if self.stage == "dpo" and self.pref_loss != "sigmoid" and self.dpo_label_smoothing > 1e-6:
            raise ValueError("`dpo_label_smoothing` is only valid for sigmoid loss function.")

        if self.use_llama_pro and self.finetuning_type == "full":
            raise ValueError("`use_llama_pro` is only valid for Freeze or LoRA training.")

        if self.finetuning_type == "lora" and (self.use_galore or self.use_apollo or self.use_badam):
            raise ValueError("Cannot use LoRA with GaLore, APOLLO or BAdam together.")

        if int(self.use_galore) + int(self.use_apollo) + (self.use_badam) > 1:
            raise ValueError("Cannot use GaLore, APOLLO or BAdam together.")

        if self.pissa_init and (self.stage in ["ppo", "kto"] or self.use_ref_model):
            raise ValueError("Cannot use PiSSA for current training stage.")

        if self.finetuning_type != "lora":
            if self.loraplus_lr_ratio is not None:
                raise ValueError("`loraplus_lr_ratio` is only valid for LoRA training.")

            if self.use_rslora:
                raise ValueError("`use_rslora` is only valid for LoRA training.")

            if self.use_dora:
                raise ValueError("`use_dora` is only valid for LoRA training.")

            if self.pissa_init:
                raise ValueError("`pissa_init` is only valid for LoRA training.")

    def to_dict(self) -> dict[str, Any]:
        args = asdict(self)
        args = {k: f"<{k.upper()}>" if k.endswith("api_key") else v for k, v in args.items()}
        return args
