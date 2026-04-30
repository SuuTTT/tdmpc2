#!/usr/bin/env python3
import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def official_mean_by_step(rows):
    values = {}
    for row in rows:
        step = int(float(row["step"]))
        values.setdefault(step, []).append(float(row["reward"]))
    return {step: sum(rewards) / len(rewards) for step, rewards in values.items()}


def final_elapsed(run_log):
    text = Path(run_log).read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"T:\s*([0-9]+:[0-9]{2}:[0-9]{2})", text)
    return matches[-1] if matches else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", required=True)
    parser.add_argument("--eval-csv", required=True)
    parser.add_argument("--result-csv", required=True)
    parser.add_argument("--compare-csv", required=True)
    parser.add_argument("--scores-jsonl", required=True)
    parser.add_argument("--state-json", required=True)
    parser.add_argument("--run-log", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--iteration", type=int, default=6)
    parser.add_argument("--mode", default="cross_task_compile_abstraction")
    parser.add_argument("--idea-id", default="IDEA-004")
    parser.add_argument("--notes", default="Acrobot-swingup cross-task abstraction probe using PyTorch compile and official TD-MPC2 CSV output schema.")
    parser.add_argument("--compile", dest="compile_enabled", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-size", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--eval-episodes", type=int, required=True)
    parser.add_argument("--pre-commit", required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()

    official_rows = read_rows(args.official)
    eval_rows = read_rows(args.eval_csv)
    candidate_rows = [
        {
            "step": str(int(float(row["step"]))),
            "reward": f"{float(row['episode_reward']):.1f}",
            "seed": str(args.seed),
        }
        for row in eval_rows
    ]
    write_rows(args.result_csv, ["step", "reward", "seed"], candidate_rows)

    official_mean = official_mean_by_step(official_rows)
    compare_rows = []
    for row in candidate_rows:
        step = int(row["step"])
        candidate_reward = float(row["reward"])
        official_reward = official_mean.get(step)
        if official_reward is None:
            continue
        compare_rows.append({
            "step": str(step),
            "official_tdmpc2_mean_reward": f"{official_reward:.1f}",
            "abstraction_compile_reward": f"{candidate_reward:.1f}",
            "delta": f"{candidate_reward - official_reward:.1f}",
            "seed": str(args.seed),
        })
    write_rows(
        args.compare_csv,
        ["step", "official_tdmpc2_mean_reward", "abstraction_compile_reward", "delta", "seed"],
        compare_rows,
    )

    final_reward = float(candidate_rows[-1]["reward"]) if candidate_rows else None
    final_official = official_mean.get(args.steps)
    train_time = final_elapsed(args.run_log)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    if args.partial:
        state = {
            "updated_at": now,
            "phase": "in_progress",
            "iteration": args.iteration,
            "latest_step": int(candidate_rows[-1]["step"]) if candidate_rows else None,
            "latest_reward": final_reward,
            "train_time": train_time,
            "result_csv": str(args.result_csv),
            "compare_csv": str(args.compare_csv),
            "run_log": str(args.run_log),
        }
        Path(args.state_json).write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return

    decision = "PENDING_COMPARISON"
    if final_reward is not None and final_official is not None:
        decision = "ABSTRACTION_BEATS_OFFICIAL_MEAN" if final_reward >= final_official else "OFFICIAL_MEAN_AHEAD"

    record = {
        "iteration": args.iteration,
        "mode": args.mode,
        "idea_id": args.idea_id,
        "task": args.task,
        "metric": "episode_reward",
        "reward": final_reward,
        "train_time": train_time,
        "seed": args.seed,
        "eval_episodes": args.eval_episodes,
        "pre_commit": args.pre_commit,
        "protocol_tier": "official_step_grid_single_seed",
        "decision": decision,
        "timestamp": now,
        "config": {
            "compile": args.compile_enabled,
            "model_size": args.model_size,
            "steps": args.steps,
        },
        "comparison_target": {
            "csv": str(args.official),
            "model_size": 5,
            "steps": args.steps,
            "official_mean_reward_at_final_step": final_official,
        },
        "checkpoint": args.checkpoint,
        "result_csv": str(args.result_csv),
        "compare_csv": str(args.compare_csv),
        "notes": args.notes,
    }
    with Path(args.scores_jsonl).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")

    state = {
        "updated_at": now,
        "phase": "completed",
        "iteration": args.iteration,
        "decision": decision,
        "reward": final_reward,
        "official_mean_reward_at_final_step": final_official,
        "train_time": train_time,
        "result_csv": str(args.result_csv),
        "compare_csv": str(args.compare_csv),
        "run_log": str(args.run_log),
    }
    Path(args.state_json).write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
