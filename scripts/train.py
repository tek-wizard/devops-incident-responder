from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _render_prompt(observation: dict) -> str:
    return (
        "You are an incident commander.\n"
        f"Task: {observation['task_id']}\n"
        f"Incident: {observation['incident_brief']}\n"
        f"Global signals: {json.dumps(observation['global_signals'])}\n"
        f"Services: {json.dumps(observation['service_status'])}\n"
        f"Recent events: {json.dumps(observation['recent_events'])}\n"
        f"Discovered facts: {json.dumps(observation['discovered_facts'])}\n"
        "Return the next action as JSON with keys command and target."
    )


def build_sft_dataset(rollout_path: Path, output_path: Path) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rollout_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as sink:
        for line in source:
            episode = json.loads(line)
            for step in episode["steps"]:
                record = {
                    "prompt": _render_prompt(step["observation"]),
                    "completion": json.dumps(step["action"]),
                    "task_id": episode["task_id"],
                    "seed": episode["seed"],
                    "reward": step["reward"]["value"],
                }
                sink.write(json.dumps(record) + "\n")
                count += 1
    return count


def maybe_run_trl(config: dict, dataset_path: Path) -> None:
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:  # pragma: no cover - only for hackathon runtime
        raise RuntimeError(
            "TRL / Transformers dependencies are not installed in this environment. "
            "Use this script onsite with the provided compute image."
        ) from exc

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    training_args = SFTConfig(
        output_dir=config["output_dir"],
        max_seq_length=config["max_seq_length"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        logging_steps=config["logging_steps"],
        save_strategy="epoch",
        report_to=[],
    )
    trainer = SFTTrainer(
        model=config["model_name"],
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(config["output_dir"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an SFT dataset and optionally launch TRL training.")
    parser.add_argument("--config", type=Path, default=Path("configs/train.json"))
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--dataset-output", type=Path, default=Path("artifacts/sft_dataset.jsonl"))
    parser.add_argument("--run-training", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    count = build_sft_dataset(args.rollouts, args.dataset_output)
    print(f"Prepared {count} SFT records at {args.dataset_output}")
    if args.run_training:
        maybe_run_trl(config, args.dataset_output)


if __name__ == "__main__":
    main()
