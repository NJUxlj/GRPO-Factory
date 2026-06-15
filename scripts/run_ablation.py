#!/usr/bin/env python3
"""DCPO Ablation Experiment Script.

Runs ablation experiments to validate the independent effects of DCPO's
three core technologies: DAC, SAS, and OTM Loss.

Experiment Groups:
    Baseline (DAPO):  Uses DAPO without DCPO features.
    DCPO-DAC:         DAC asymmetric clip only.
    DCPO-SAS:         SAS smooth advantage only.
    DCPO-OTM:         OTM loss aggregation only.
    DCPO-Full:        All three technologies enabled.
"""

import os
import subprocess
import sys
from typing import Dict, List


def generate_config(
    name: str,
    config_overrides: Dict[str, str],
    base_template: str = "examples/train_lora/qwen3_lora_dapo.yaml",
) -> str:
    """Generate a temporary config file for an ablation experiment.

    Args:
        name: Experiment name (used for output path).
        config_overrides: Dictionary of config key-value overrides.
        base_template: Path to the base YAML config template.

    Returns:
        Path to the generated temporary config file.
    """
    output_path = f"/tmp/ablation_{name}.yaml"
    with open(base_template, "r") as f:
        lines = f.readlines()

    modified_lines = []
    overridden_keys = set()

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            modified_lines.append(line)
            continue

        # Check if this line's key is in overrides
        for key, value in config_overrides.items():
            if stripped.startswith(f"{key}:"):
                modified_lines.append(f"{key}: {value}\n")
                overridden_keys.add(key)
                break
        else:
            modified_lines.append(line)

    # Append any new keys not found in the original
    for key, value in config_overrides.items():
        if key not in overridden_keys:
            modified_lines.append(f"{key}: {value}\n")

    with open(output_path, "w") as f:
        f.writelines(modified_lines)

    return output_path


def run_experiment(name: str, config_path: str, max_steps: int = 5) -> Dict[str, str]:
    """Run a single ablation experiment.

    Args:
        name: Experiment name.
        config_path: Path to the experiment config file.
        max_steps: Maximum training steps (small for quick ablation).

    Returns:
        Dictionary of experiment results.
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {name}")
    print(f"Config: {config_path}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "-m", "llamafactory.launcher",
        config_path,
        f"output_dir=saves/ablation/{name}",
        f"max_steps={max_steps}",
        "logging_steps=1",
        "save_steps=1000",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes timeout
        )
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        if result.returncode != 0:
            print(f"STDERR: {result.stderr[-500:]}")
            return {"name": name, "status": "FAILED", "error": result.stderr[-200:]}
    except subprocess.TimeoutExpired:
        return {"name": name, "status": "TIMEOUT"}
    except Exception as e:
        return {"name": name, "status": "ERROR", "error": str(e)}

    return {"name": name, "status": "COMPLETED"}


def main():
    """Run all ablation experiments."""

    # Define experiment groups
    experiments = {
        "dapo_baseline": {
            "grpo_loss_mode": "dapo",
            "output_dir": "saves/ablation/dapo_baseline",
        },
        "dcpo_dac_only": {
            "grpo_loss_mode": "dcpo",
            "dcpo_clip_ratio_low": "0.16",
            "dcpo_clip_ratio_high": "0.20",
            "dcpo_sas_enable": "false",
            "dcpo_loss_agg_mode": "token-mean",
            "output_dir": "saves/ablation/dcpo_dac_only",
        },
        "dcpo_sas_only": {
            "grpo_loss_mode": "dcpo",
            "dapo_clip_ratio_low": "0.2",
            "dapo_clip_ratio_high": "0.28",
            "dcpo_sas_enable": "true",
            "dcpo_sas_threshold": "3.0",
            "dcpo_loss_agg_mode": "token-mean",
            "output_dir": "saves/ablation/dcpo_sas_only",
        },
        "dcpo_otm_only": {
            "grpo_loss_mode": "dcpo",
            "dapo_clip_ratio_low": "0.2",
            "dapo_clip_ratio_high": "0.28",
            "dcpo_sas_enable": "false",
            "dcpo_loss_agg_mode": "otm",
            "output_dir": "saves/ablation/dcpo_otm_only",
        },
        "dcpo_full": {
            "grpo_loss_mode": "dcpo",
            "dcpo_clip_ratio_low": "0.16",
            "dcpo_clip_ratio_high": "0.20",
            "dcpo_sas_enable": "true",
            "dcpo_sas_threshold": "3.0",
            "dcpo_loss_agg_mode": "otm",
            "output_dir": "saves/ablation/dcpo_full",
        },
    }

    results = []
    for name, overrides in experiments.items():
        config_path = generate_config(
            name,
            overrides,
            base_template="examples/train_lora/qwen3_lora_dcpo.yaml",
        )
        result = run_experiment(name, config_path, max_steps=3)
        results.append(result)

    # Print summary table
    print("\n\n" + "=" * 60)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Experiment':<25} {'Status':<15}")
    print("-" * 40)
    for r in results:
        print(f"{r['name']:<25} {r['status']:<15}")

    # Cleanup temp files
    for name in experiments:
        tmp_path = f"/tmp/ablation_{name}.yaml"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    main()
