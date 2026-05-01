#!/usr/bin/env python3
"""Attach summaries/artifacts to an existing W&B training run.

This avoids creating short, empty-looking W&B runs for post-training export.
The script finds the local W&B run id from a training run directory and resumes
that run before saving artifacts or updating summary fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_run_id(run_dir: Path) -> str:
	wandb_dir = run_dir / "wandb"
	candidates = sorted(wandb_dir.glob("run-*"))
	if not candidates:
		raise FileNotFoundError(f"No W&B run directory found under {wandb_dir}")
	latest = candidates[-1].name
	return latest.split("-")[-1]


def load_summary_json(path: Path | None) -> dict[str, object]:
	if path is None:
		return {}
	return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--project", required=True)
	parser.add_argument("--entity", default=None)
	parser.add_argument("--run-dir", type=Path, required=True)
	parser.add_argument("--name", required=True)
	parser.add_argument("--summary-json", type=Path, default=None)
	parser.add_argument("--artifact", action="append", default=[])
	args = parser.parse_args()

	run_id = find_run_id(args.run_dir)

	import wandb

	run = wandb.init(
		project=args.project,
		entity=args.entity,
		id=run_id,
		name=args.name,
		resume="allow",
	)
	for key, value in load_summary_json(args.summary_json).items():
		run.summary[key] = value
	for item in args.artifact:
		path = Path(item)
		if path.exists():
			wandb.save(str(path))
	run.finish()
	print(json.dumps({"attached_to_run_id": run_id, "name": args.name}, indent=2))


if __name__ == "__main__":
	main()
