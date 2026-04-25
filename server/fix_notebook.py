from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("/home/prateeksingh/Desktop/devops_responder/devops_grpo_training.ipynb")


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip("\n").split("\n")],
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip("\n").split("\n")],
    }


def main() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    notebook["cells"] = [
        md(
            """
# GRPO Training — DevOps Incident Responder

Trains `Qwen2.5-1.5B-Instruct` with GRPO to produce incident-response action plans
for the DevOps OpenEnv benchmark.

This notebook follows the same hackathon-style flow as `abdur.ipynb`:

1. install dependencies
2. load model
3. define environment reward function
4. build dataset
5. run GRPO
6. plot reward curve
7. evaluate the trained agent
8. save results
            """
        ),
        md("## 1. Install Dependencies"),
        code(
            """
%%capture
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --upgrade "trl>=0.15.0" transformers datasets httpx matplotlib uvicorn fastapi
            """
        ),
        md("## 2. Load Model with Unsloth (4-bit quantisation)"),
        code(
            """
from unsloth import FastLanguageModel
import torch

MAX_SEQ_LENGTH = 512

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    fast_inference=False,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

print("Model loaded ✓")
            """
        ),
        md("## 3. Environment Reward Function"),
        code(
            """
import json
import os

import httpx

ENV_URL = os.environ.get("DEVOPS_ENV_URL", "http://127.0.0.1:7860").rstrip("/")

with httpx.Client(base_url=ENV_URL, timeout=20) as http:
    health = http.get("/health")
    assert health.status_code == 200, f"Server not reachable: {health.status_code}"
    catalog = http.get("/tasks")
    catalog.raise_for_status()
    catalog_payload = catalog.json()

TASKS = {task["id"]: task for task in catalog_payload["tasks"]}
SEED_SETS = catalog_payload["seed_sets"]


def fetch_observation(task_id: str, seed: int) -> dict:
    with httpx.Client(base_url=ENV_URL, timeout=20) as http:
        response = http.post("/reset", params={"task_id": task_id, "seed": int(seed)})
        response.raise_for_status()
        return response.json()


sample_seed = SEED_SETS["train"][0]
observations = {task_id: fetch_observation(task_id, sample_seed) for task_id in TASKS}
KNOWN_COMMANDS = sorted({command for obs in observations.values() for command in obs["available_commands"]})
KNOWN_TARGETS = sorted(
    {
        "system",
        *{
            service_name
            for obs in observations.values()
            for service_name in obs["service_status"].keys()
        },
    }
)

COMMAND_ALIASES = {
    "restart": "restart_service",
    "scale": "scale_service",
    "get_logs": "query_logs",
    "check_metrics": "get_metrics",
}
TARGET_ALIASES = {
    "db": "main-db",
    "system": "system",
    "main-db": "main-db",
}


def fetch_baseline_summary() -> dict | None:
    try:
        with httpx.Client(base_url=ENV_URL, timeout=60) as http:
            response = http.post("/baseline")
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results")
            if not isinstance(results, list) or not results:
                return None
            avg_score = sum(float(item["score"]) for item in results) / len(results)
            solve_rate = sum(1.0 if item.get("success") else 0.0 for item in results) / len(results)
            return {
                "episodes": results,
                "summary": {
                    "avg_score": round(avg_score, 4),
                    "solve_rate": round(solve_rate, 4),
                },
            }
    except Exception as exc:
        print(f"Baseline endpoint unavailable: {exc}")
        return None


baseline_summary = fetch_baseline_summary()

print(f"Connected to local DevOps environment ✓  ({ENV_URL})")
print(f"Loaded {len(TASKS)} tasks")
print(f"Train seeds: {SEED_SETS['train']}")
print(f"Validation seeds: {SEED_SETS['val']}")
print(f"Discovered commands: {KNOWN_COMMANDS}")
print(f"Discovered targets: {KNOWN_TARGETS}")
if baseline_summary is not None:
    print("Loaded baseline results from /baseline")
            """
        ),
        code(
            """
import re


def compact_json(value) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def normalize_action(action: dict) -> dict:
    command = str(action.get("command", "")).strip()
    target = str(action.get("target", "")).strip()
    params = action.get("params", {})
    if not isinstance(params, dict):
        params = {}
    return {
        "command": COMMAND_ALIASES.get(command, command),
        "target": TARGET_ALIASES.get(target, target),
        "params": {str(k): str(v) for k, v in params.items()},
    }


def extract_json_blob(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
    match = re.search(r"(\\{.*\\}|\\[.*\\])", cleaned, re.DOTALL)
    return match.group(1) if match else ""


def payload_to_actions(payload) -> list[dict]:
    if isinstance(payload, dict) and "actions" in payload:
        raw_actions = payload["actions"]
    elif isinstance(payload, dict) and "command" in payload and "target" in payload:
        raw_actions = [payload]
    elif isinstance(payload, list):
        raw_actions = payload
    else:
        return []

    actions = []
    for item in raw_actions[:6]:
        if not isinstance(item, dict):
            continue
        normalized = normalize_action(item)
        if normalized["command"] and normalized["target"]:
            actions.append(normalized)
    return actions


def fallback_actions(text: str) -> list[dict]:
    lines = [segment.strip().lower() for segment in re.split(r"[\\n;]+", text) if segment.strip()]
    alias_to_command = {**{cmd: cmd for cmd in KNOWN_COMMANDS}, **COMMAND_ALIASES}
    actions = []

    for line in lines:
        matched_command = None
        for alias in sorted(alias_to_command, key=len, reverse=True):
            if alias in line:
                matched_command = alias_to_command[alias]
                break
        if matched_command is None:
            continue

        matched_target = None
        for target in sorted(KNOWN_TARGETS, key=len, reverse=True):
            if target in line:
                matched_target = TARGET_ALIASES.get(target, target)
                break
        if matched_target is None:
            continue

        actions.append({"command": matched_command, "target": matched_target, "params": {}})

    if actions:
        return actions[:6]

    lowered = text.lower()
    command_hits = [alias_to_command[alias] for alias in sorted(alias_to_command, key=len, reverse=True) if alias in lowered]
    target_hits = [TARGET_ALIASES.get(target, target) for target in sorted(KNOWN_TARGETS, key=len, reverse=True) if target in lowered]
    if command_hits and target_hits:
        return [{"command": command_hits[0], "target": target_hits[0], "params": {}}]
    return []


def parse_actions(completion: str) -> tuple[list[dict], str]:
    blob = extract_json_blob(completion)
    if blob:
        try:
            payload = json.loads(blob)
            actions = payload_to_actions(payload)
            if actions:
                return actions, "json"
        except json.JSONDecodeError:
            pass

    actions = fallback_actions(completion)
    if actions:
        return actions, "fallback"
    return [], "unparsed"


def run_plan_episode(task_id: str, seed: int, completion: str) -> tuple[float, dict]:
    actions, parse_mode = parse_actions(completion)
    if not actions:
        return 0.0, {"task_id": task_id, "seed": seed, "parse_mode": parse_mode, "actions": []}

    with httpx.Client(base_url=ENV_URL, timeout=30) as http:
        reset = http.post("/reset", params={"task_id": task_id, "seed": int(seed)})
        reset.raise_for_status()
        observation = reset.json()
        session_id = observation["session_id"]

        executed = []
        done = False
        for action in actions:
            action_payload = dict(action)
            action_payload["session_id"] = session_id
            step = http.post("/step", json=action_payload)
            if step.status_code != 200:
                break
            executed.append(action_payload)
            payload = step.json()
            done = bool(payload.get("done", False))
            if done:
                break

        grader = http.get("/grader", params={"session_id": session_id})
        grader.raise_for_status()
        grade = grader.json()
        return float(grade["score"]), {
            "task_id": task_id,
            "seed": int(seed),
            "parse_mode": parse_mode,
            "actions": executed,
            "grade": grade,
            "done": done,
        }


def reward_fn(completions: list[str], task_id: list[str], seed: list[int], **kwargs) -> list[float]:
    rewards = []
    reward_fn.last_batch = []
    for completion, tid, seed_value in zip(completions, task_id, seed):
        score, details = run_plan_episode(tid, int(seed_value), completion.strip())
        rewards.append(score)
        reward_fn.last_batch.append(details)
    return rewards


reward_fn.last_batch = []

print("Reward smoke-test on one valid action per task:")
for task_id, observation in observations.items():
    first_action = {"command": observation["available_commands"][0], "target": "system"}
    if first_action["target"] not in KNOWN_TARGETS:
        first_action["target"] = sorted(observation["service_status"].keys())[0]
    score, details = run_plan_episode(task_id, SEED_SETS["train"][0], compact_json(first_action))
    print(f"  {task_id:26s} score={score:.4f} parse={details['parse_mode']} steps={len(details['actions'])}")

bad_score, bad_details = run_plan_episode("bad_auth_deploy", SEED_SETS["train"][0], "do something smart")
print(f"Malformed-output smoke-test: score={bad_score:.4f} parse={bad_details['parse_mode']}  (expect 0.0)")
            """
        ),
        md("## 4. Training Dataset"),
        code(
            """
from datasets import Dataset

SYSTEM_PROMPT = f\"\"\"You are a senior DevOps incident commander.
Return ONLY JSON in one of these forms:
{{"actions":[{{"command":"...", "target":"..."}}]}}
or
{{"command":"...", "target":"..."}}

Rules:
- Use only these commands: {", ".join(KNOWN_COMMANDS)}
- Use only these targets: {", ".join(KNOWN_TARGETS)}
- Prefer short plans of 2 to 6 actions.
- Prefer investigation before risky mitigation when evidence is incomplete.
- Include run_smoke_test near the end once service should be healthy.
- Do not add explanation outside JSON.\"\"\"


observation_cache = {}


def fetch_initial_observation(task_id: str, seed: int) -> dict:
    key = (task_id, int(seed))
    if key in observation_cache:
        return observation_cache[key]
    with httpx.Client(base_url=ENV_URL, timeout=20) as http:
        response = http.post("/reset", params={"task_id": task_id, "seed": int(seed)})
        response.raise_for_status()
        observation_cache[key] = response.json()
    return observation_cache[key]


def make_prompt(task_id: str, seed: int) -> str:
    task = TASKS[task_id]
    observation = fetch_initial_observation(task_id, seed)
    return (
        f"<|im_start|>system\\n{SYSTEM_PROMPT}<|im_end|>\\n"
        f"<|im_start|>user\\n"
        f"Task: {task_id}\\n"
        f"Difficulty: {task['difficulty']}\\n"
        f"Description: {task['description']}\\n"
        f"Incident brief: {observation['incident_brief']}\\n"
        f"Global signals: {compact_json(observation['global_signals'])}\\n"
        f"Service status: {compact_json(observation['service_status'])}\\n"
        f"Recent events: {compact_json(observation['recent_events'])}\\n"
        f"Discovered facts: {compact_json(observation['discovered_facts'])}\\n"
        "Return the remediation plan JSON now.<|im_end|>\\n"
        f"<|im_start|>assistant\\n"
    )


EXAMPLES_PER_TASK_SEED = 4

examples = []
for task_id in TASKS:
    for seed in SEED_SETS["train"]:
        prompt = make_prompt(task_id, seed)
        for _ in range(EXAMPLES_PER_TASK_SEED):
            examples.append({"prompt": prompt, "task_id": task_id, "seed": int(seed)})

dataset = Dataset.from_list(examples).shuffle(seed=42)
print(f"Dataset: {len(dataset)} examples")
print(dataset[0]["prompt"][:600])
            """
        ),
        md("## 5. GRPO Training"),
        code(
            """
from trl import GRPOConfig, GRPOTrainer

grpo_config = GRPOConfig(
    num_generations=4,
    max_completion_length=220,
    temperature=0.8,
    max_steps=120,
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    output_dir="./grpo_devops_output",
    logging_steps=5,
    save_steps=60,
    report_to="none",
    beta=0.04,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_fn],
    args=grpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

print("Starting GRPO training...")
print(f"  Steps: {grpo_config.max_steps}")
print(f"  Generations per prompt: {grpo_config.num_generations}")
print(f"  Effective batch size: {grpo_config.per_device_train_batch_size * grpo_config.gradient_accumulation_steps}")
trainer.train()
            """
        ),
        md("## 6. Plot Reward Curve"),
        code(
            """
import json
import matplotlib.pyplot as plt

log_history = trainer.state.log_history

steps = [row["step"] for row in log_history if "reward" in row]
rewards = [row["reward"] for row in log_history if "reward" in row]

print("Training log:")
for step, reward in zip(steps, rewards):
    print(f"  step {step:>4}: reward = {reward:.4f}")

if steps:
    plt.figure(figsize=(10, 5))
    plt.plot(steps, rewards, marker="o", linewidth=2, label="GRPO agent (training)")
    if baseline_summary is not None and "summary" in baseline_summary and "avg_score" in baseline_summary["summary"]:
        plt.axhline(
            y=baseline_summary["summary"]["avg_score"],
            color="red",
            linestyle="--",
            label=f"Heuristic baseline ({baseline_summary['summary']['avg_score']:.2f})",
        )
    plt.axhline(y=1.0, color="green", linestyle="--", label="Maximum score (1.00)")
    plt.xlabel("Training Step")
    plt.ylabel("Mean Episode Reward")
    plt.title("GRPO Training: DevOps Incident Responder")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reward_curve.png", dpi=150)
    plt.show()
    print("Saved reward_curve.png")
else:
    print("No reward logs found — check trainer.state.log_history")
            """
        ),
        md("## 7. Evaluate Trained Agent"),
        code(
            """
import torch

FastLanguageModel.for_inference(model)


def generate_plan(task_id: str, seed: int, *, temperature: float = 0.7, do_sample: bool = True) -> str:
    prompt = make_prompt(task_id, seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=220,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


print("Evaluating trained agent (3 attempts per task on the first validation seed)...\\n")

demo_seed = SEED_SETS["val"][0]
task_scores = {}
task_outputs = {}

for task_id in TASKS:
    scores = []
    outputs = []
    for trial in range(3):
        completion = generate_plan(task_id, demo_seed, temperature=0.7, do_sample=True)
        score, details = run_plan_episode(task_id, demo_seed, completion)
        scores.append(score)
        outputs.append({"completion": completion, "details": details})
        print(f"  [{task_id}] trial {trial + 1}: score={score:.4f} | output={completion[:120]}")
    task_scores[task_id] = max(scores)
    task_outputs[task_id] = outputs
    print(f"  → best score: {task_scores[task_id]:.4f}\\n")

overall_demo = sum(task_scores.values()) / len(task_scores)

print(f"\\n{'=' * 60}")
print(f"Demo-seed average score:    {overall_demo:.4f}")
if baseline_summary is not None and "summary" in baseline_summary and "avg_score" in baseline_summary["summary"]:
    print(f"Heuristic baseline score:   {baseline_summary['summary']['avg_score']:.4f}")
print(f"{'=' * 60}")


def evaluate_validation_set() -> dict:
    rows = []
    for task_id in TASKS:
        for seed in SEED_SETS["val"]:
            completion = generate_plan(task_id, seed, temperature=0.0, do_sample=False)
            score, details = run_plan_episode(task_id, seed, completion)
            rows.append(
                {
                    "task_id": task_id,
                    "seed": int(seed),
                    "score": score,
                    "resolved": bool(details.get("grade", {}).get("resolved", False)),
                    "parse_mode": details["parse_mode"],
                    "completion": completion,
                }
            )

    mean_score = sum(row["score"] for row in rows) / len(rows)
    solve_rate = sum(1.0 if row["resolved"] else 0.0 for row in rows) / len(rows)
    return {
        "episodes": rows,
        "overall": {
            "avg_score": round(mean_score, 4),
            "solve_rate": round(solve_rate, 4),
        },
    }


validation_results = evaluate_validation_set()
print("\\nDeterministic validation summary:")
print(json.dumps(validation_results["overall"], indent=2))
            """
        ),
        md("## 8. Save Results"),
        code(
            """
import json

results = {
    "model": "unsloth/Qwen2.5-1.5B-Instruct + GRPO LoRA",
    "training_steps": grpo_config.max_steps,
    "num_generations": grpo_config.num_generations,
    "heuristic_validation_baseline": baseline_summary,
    "demo_seed": int(demo_seed),
    "demo_seed_task_scores": task_scores,
    "demo_seed_overall": round(overall_demo, 4),
    "validation_results": validation_results,
    "reward_curve": {"steps": steps, "rewards": rewards},
}

model.save_pretrained_lora("grpo_devops_commander")
tokenizer.save_pretrained("grpo_devops_commander")

with open("training_results.json", "w", encoding="utf-8") as handle:
    json.dump(results, handle, indent=2)

print("Saved LoRA adapter to grpo_devops_commander")
print("Saved training_results.json")
print(json.dumps(results["validation_results"]["overall"], indent=2))
            """
        ),
    ]
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Notebook updated: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
