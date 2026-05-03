# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Code style (auto-fix)
make style

# Code quality check (no modifications)
make quality

# Run all tests
make test

# Run a single test file
WANDB_DISABLED=true pytest -vv --import-mode=importlib tests/path/to/test_file.py

# Run tests matching a pattern
WANDB_DISABLED=true pytest -vv --import-mode=importlib tests/ -k "test_name"

# License header check
make license

# Build package
make build
```

The project uses `uv` as the preferred package manager. Commands automatically use `uv run` / `uvx` if `uv` is available.

## Architecture

LlamaFactory has two parallel architectures controlled by the `USE_V1` environment variable:

- **v0 (default):** `api, webui > chat, eval, train > data, model > hparams > extras`
- **v1 (experimental, `USE_V1=1`):** `trainers > core > accelerator, plugins, config > utils`

Most active development happens in v0. The v1 architecture lives in `src/llamafactory/v1/`.

### Entry Points

CLI entry point is `llamafactory-cli` / `lmf` → `src/llamafactory/cli.py:main()`, which dispatches to `launcher.py` based on `USE_V1`.

Available subcommands: `train`, `chat`, `api`, `export`, `webchat`, `webui`, `env`, `version`, `help`.

### Training Flow (v0)

```
run_exp() [tuner.py]
  → read_args() → parse YAML/JSON config
  → get_train_args() → produces typed argument dataclasses
  → routes to: run_sft / run_dpo / run_ppo / run_rm / run_pt / run_kto
  → optional: export_model()
```

Training is invoked with a YAML config: `llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml`

### Configuration System

All training parameters are YAML/JSON config files. Argument parsing in `src/llamafactory/hparams/parser.py` produces four typed dataclasses:
- `ModelArguments` — model/tokenizer selection, quantization
- `DataArguments` — datasets, templates, preprocessing
- `FinetuningArguments` — LoRA rank/target, training method (sft/dpo/ppo/rm/pt/kto)
- `TrainingArguments` — extends HuggingFace's `TrainingArguments`

### Key Modules

| Module | Purpose |
|--------|---------|
| `src/llamafactory/model/loader.py` | Loads model + tokenizer; applies quantization, LoRA, patches |
| `src/llamafactory/model/patcher.py` | Model-specific compatibility patches |
| `src/llamafactory/data/template.py` | Prompt templates; `TEMPLATES` dict maps model family → format |
| `src/llamafactory/data/mm_plugin.py` | Multi-modal (image/video/audio) data handling |
| `src/llamafactory/data/processor/` | Per-stage data processors (supervised, pairwise, pretrain, etc.) |
| `src/llamafactory/train/sft/` | SFT trainer; other stages follow same structure |
| `src/llamafactory/chat/` | Inference engines: `hf_engine`, `vllm_engine`, `sglang_engine`, `kt_engine` |
| `src/llamafactory/extras/constants.py` | Enums and constants used across the project |

### Adding a New Training Stage (e.g., GRPO/GSPO/DAPO)

To add a new RL training method, follow this pattern:

1. **Define stage in `FinetuningArguments`** (`src/llamafactory/hparams/finetuning_args.py`):
   - Add to the `stage` Literal type: `Literal["pt", "sft", "rm", "ppo", "dpo", "kto", "grpo"]`
   - Add new hyperparameters in a new dataclass (e.g., `GRPOArguments`) or extend `RLHFArguments`

2. **Add routing in `tuner.py`** (`src/llamafactory/train/tuner.py`):
   - Import your workflow module: `from .grpo import run_grpo`
   - Add route: `elif finetuning_args.stage == "grpo": run_grpo(...)`

3. **Create workflow module** (`src/llamafactory/train/grpo/workflow.py`):
   - Follow `run_dpo()` pattern: load tokenizer, dataset, model, create trainer
   - Return train_result and handle logging/metrics

4. **Create trainer class** (`src/llamafactory/train/grpo/trainer.py`):
   - Extend `CustomDPOTrainer` or `CustomPPOTrainer` depending on whether you need a reference model
   - Key methods to override: `compute_loss()`, `get_batch_loss_metrics()`
   - GRPO typically uses group-relative reward normalization (see `get_batch_logps` in `trainer_utils.py`)

5. **Add data processor** (`src/llamafactory/data/processor/`):
   - If using prompts only (no pairwise data), use `UnsupervisedDatasetProcessor`
   - If using self-play/generation data, create a new processor following `PairwiseDatasetProcessor` pattern

6. **Add data collator** (`src/llamafactory/data/collator.py`):
   - Add `XXXDataCollatorWithPadding` following `PairwiseDataCollatorWithPadding` pattern

7. **Add example YAML** (`examples/train_lora/llama3_lora_grpo.yaml`):
   - Set `stage: grpo` and any new hyperparameters

### Training Methods Architecture

Each training stage follows an identical pattern:
- `src/llamafactory/train/<stage>/workflow.py` — orchestrates data loading, model init, trainer creation
- `src/llamafactory/train/<stage>/trainer.py` — implements the training logic (loss computation, metrics)
- `src/llamafactory/data/processor/<stage>.py` — preprocesses dataset to model inputs

Existing stages:
- **pt** (Pretrain): `pretrain.py` + `UnsupervisedDatasetProcessor`
- **sft** (Supervised Fine-tuning): `sft/trainer.py` + `SupervisedDatasetProcessor`
- **rm** (Reward Modeling): `rm/trainer.py` + `PairwiseDatasetProcessor`
- **dpo** (Direct Preference Optimization): `dpo/trainer.py` + `PairwiseDatasetProcessor` (uses chosen/rejected pairs)
- **ppo** (Proximal Policy Optimization): `ppo/trainer.py` + reward model for scoring
- **kto** (KTO): `kto/trainer.py` + `FeedbackDatasetProcessor`

### Key Utilities for RL Training

- `get_batch_logps()` in `trainer_utils.py`: Computes log probabilities from logits (used by DPO, GRPO)
- `create_ref_model()` in `trainer_utils.py`: Creates reference model for KL regularization
- `nested_detach()` in `trainer_utils.py`: Detaches tensors without losing structure

### Adding Support for a New Model

1. Add a prompt template to `src/llamafactory/data/template.py` in the `TEMPLATES` dict
2. Add any necessary model patches in `src/llamafactory/model/patcher.py`
3. Add multi-modal support in `src/llamafactory/data/mm_plugin.py` if needed

### Distributed Training

Multi-GPU automatically uses `torchrun`. Additional backends:
- **Ray:** Optional Ray cluster support
- **HyperParallel FSDP2:** `src/llamafactory/train/hyper_parallel/`
- **Megatron-core:** `src/llamafactory/train/mca/`

### Testing

- `tests/` — v0 tests; `tests_v1/` — v1 tests
- Most training tests require GPU hardware
- pytest markers: `@pytest.mark.slow`, `@pytest.mark.runs_on(['cuda'])`
- Always set `WANDB_DISABLED=true` when running tests

### Code Style

- Ruff for linting and formatting (line length 119, Google-style docstrings)
- Python 3.11+ syntax
- Double quotes for strings
- All new files must include Apache 2.0 license header (checked by `make license`)
