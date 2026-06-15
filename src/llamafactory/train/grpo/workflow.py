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

"""Workflow orchestration for GRPO/DAPO/GSPO/DCPO training."""

from typing import TYPE_CHECKING, Optional

from ...data import get_dataset, get_template_and_fix_tokenizer
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_ref_model
from .reward.manager import RewardManager
from .trainer import CustomGRPOTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


def create_reward_manager(finetuning_args: "FinetuningArguments") -> RewardManager:
    """Factory method: construct RewardManager based on grpo_reward_type.

    Args:
        finetuning_args: Finetuning arguments containing reward configuration.

    Returns:
        A configured RewardManager instance.
    """
    return RewardManager(finetuning_args)


def run_grpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    """Unified training entry point for GRPO/DAPO/GSPO/DCPO.

    All four algorithms share the same stage "grpo" and are dispatched via
    the `grpo_loss_mode` parameter. This function handles the full setup:
    tokenizer, dataset, model, reference model, reward manager, and trainer.

    Args:
        model_args: Model configuration arguments.
        data_args: Data configuration arguments.
        training_args: Training configuration arguments.
        finetuning_args: Finetuning arguments including GRPO algorithm settings.
        generating_args: Generation configuration arguments.
        callbacks: Optional list of TrainerCallbacks.
    """
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(
        template, model_args, data_args, training_args, stage="grpo", **tokenizer_module
    )
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    tokenizer.padding_side = "left"  # left-padding for generation

    # Create reference model (frozen copy of the initial policy)
    ref_model = create_ref_model(model_args, finetuning_args)

    # Create reward manager
    reward_manager = create_reward_manager(finetuning_args)

    # Initialize the GRPO trainer
    trainer = CustomGRPOTrainer(
        ref_model=ref_model,
        reward_manager=reward_manager,
        finetuning_args=finetuning_args,
        model=model,
        args=training_args,
        data_collator=dataset_module.get("data_collator"),
        train_dataset=dataset_module.get("train_dataset"),
        eval_dataset=dataset_module.get("eval_dataset"),
        tokenizer=tokenizer,
        processor=tokenizer_module.get("processor"),
        callbacks=callbacks,
    )

    # Training
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "reward_mean"])
