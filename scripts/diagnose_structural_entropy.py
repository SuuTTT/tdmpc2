#!/usr/bin/env python3
"""Post-hoc structural diagnostics for a trained single-task TD-MPC2 checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tdmpc2"))

from common.seed import set_seed  # noqa: E402
from envs import make_env  # noqa: E402
from tdmpc2 import TDMPC2  # noqa: E402


def _assignments(z: torch.Tensor, simnorm_dim: int) -> torch.Tensor:
	z = z.view(*z.shape[:-1], -1, simnorm_dim)
	return z.mean(-2)


def _flow_metrics(source: torch.Tensor, target: torch.Tensor) -> tuple[np.ndarray, dict[str, float]]:
	flow = source.transpose(0, 1).matmul(target) / max(source.shape[0], 1)
	volume = flow.sum(0) + flow.sum(1)
	internal = flow.diag()
	cut = (volume - 2 * internal).clamp_min(0)
	total_volume = volume.sum().clamp_min(1e-8)
	ratio = (volume / total_volume).clamp_min(1e-8)
	entropy = -(cut / total_volume * ratio.log()).sum()
	flow_np = flow.detach().cpu().numpy()
	volume_share = (volume / total_volume).detach().cpu().numpy()
	metrics = {
		"structural_entropy": float(entropy.detach().cpu()),
		"diagonal_flow_mass": float(internal.sum().div(flow.sum().clamp_min(1e-8)).detach().cpu()),
		"active_modules_1pct": int((volume_share > 0.01).sum()),
		"max_volume_share": float(volume_share.max()),
		"flow_total": float(flow.sum().detach().cpu()),
	}
	return flow_np, metrics


def _flow_components(flow: np.ndarray) -> dict[str, np.ndarray]:
	volume = flow.sum(axis=0) + flow.sum(axis=1)
	internal = np.diag(flow)
	cut = np.maximum(volume - 2 * internal, 0.0)
	total_volume = max(volume.sum(), 1e-8)
	share = np.maximum(volume / total_volume, 1e-8)
	contrib = -(cut / total_volume) * np.log(share)
	return {"volume": volume, "cut": cut, "contrib": contrib}


def _save_heatmap(flow: np.ndarray, output_pdf: Path, output_png: Path, title: str) -> None:
	array = np.asarray(flow, dtype=float)
	array = array - array.min()
	scale = array.max() if array.max() > 0 else 1.0
	array = array / scale
	h, w = array.shape
	pix = np.zeros((h, w, 3), dtype=np.uint8)
	pix[..., 0] = (255 * np.clip(1.4 * array, 0, 1)).astype(np.uint8)
	pix[..., 1] = (255 * np.clip(1.1 * (1 - np.abs(array - 0.5) * 1.8), 0, 1)).astype(np.uint8)
	pix[..., 2] = (255 * np.clip(1 - array, 0, 1)).astype(np.uint8)
	img = Image.fromarray(pix, mode="RGB").resize((w * 28, h * 28), Image.Resampling.NEAREST)
	canvas = Image.new("RGB", (img.width + 220, img.height + 120), "white")
	draw = ImageDraw.Draw(canvas)
	font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
	font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
	draw.text((24, 16), title, font=font_title, fill=(30, 30, 30))
	canvas.paste(img, (100, 55))
	draw.text((img.width / 2 + 85, img.height + 66), "target module", font=font_label, fill=(60, 60, 60), anchor="mm")
	draw.text((34, img.height / 2 + 55), "source module", font=font_label, fill=(60, 60, 60), anchor="mm")
	canvas.save(output_png, dpi=(300, 300))
	canvas.save(output_pdf, "PDF", resolution=300.0)


def _save_weight_figure(pred_flow: np.ndarray, obs_flow: np.ndarray, output_pdf: Path, output_png: Path) -> None:
	pred = _flow_components(pred_flow)
	obs = _flow_components(obs_flow)
	font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
	font_panel = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
	font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
	font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
	canvas = Image.new("RGB", (2200, 900), "white")
	draw = ImageDraw.Draw(canvas)
	draw.text((30, 18), "Planned structural-weight visualization from checkpoint diagnostics", font=font_title, fill=(25, 25, 25))

	def heatmap(arr: np.ndarray, x: int, y: int, title: str):
		a = np.asarray(arr, dtype=float)
		a = a - a.min()
		scale = a.max() if a.max() > 0 else 1.0
		a = a / scale
		h, w = a.shape
		pix = np.zeros((h, w, 3), dtype=np.uint8)
		pix[..., 0] = (255 * np.clip(1.4 * a, 0, 1)).astype(np.uint8)
		pix[..., 1] = (255 * np.clip(1.15 * (1 - np.abs(a - 0.5) * 1.8), 0, 1)).astype(np.uint8)
		pix[..., 2] = (255 * np.clip(1 - a, 0, 1)).astype(np.uint8)
		img = Image.fromarray(pix, mode="RGB").resize((320, 320), Image.Resampling.NEAREST)
		canvas.paste(img, (x, y))
		draw.rectangle((x, y, x + 320, y + 320), outline=(210, 215, 220), width=2)
		draw.text((x, y - 28), title, font=font_panel, fill=(30, 30, 30))

	def bars(values: np.ndarray, x: int, y: int, title: str, fill: tuple[int, int, int]):
		draw.rectangle((x, y, x + 320, y + 320), outline=(210, 215, 220), width=2)
		draw.text((x, y - 28), title, font=font_panel, fill=(30, 30, 30))
		vals = np.asarray(values, dtype=float)
		maxv = float(np.max(vals)) if np.max(vals) > 0 else 1.0
		for i, v in enumerate(vals):
			bh = int(250 * (v / maxv))
			xx = x + 18 + i * 36
			draw.rectangle((xx, y + 280 - bh, xx + 22, y + 280), fill=fill, outline=fill)
			draw.text((xx + 11, y + 290), str(i), font=font_small, fill=(90, 90, 90), anchor="mt")

	heatmap(pred_flow, 40, 70, "Predicted transition flow")
	heatmap(obs_flow, 420, 70, "Observed encoded transition flow")
	bars(pred["volume"], 860, 70, "Module volume", (1, 115, 178))
	bars(pred["cut"], 1220, 70, "Cut / escape mass", (222, 143, 5))
	bars(pred["contrib"], 1580, 70, "Per-module SE contribution", (214, 31, 140))
	draw.text((40, 760), "This figure should be regenerated from the saved checkpoint after diagnosis; the above values come from the actual transition-flow arrays exported by the script.", font=font_label, fill=(120, 20, 20))
	canvas.save(output_png, dpi=(300, 300))
	canvas.save(output_pdf, "PDF", resolution=300.0)


def diagnose(args: argparse.Namespace) -> dict[str, object]:
	from omegaconf import OmegaConf

	cfg = OmegaConf.load(REPO / "tdmpc2" / "config.yaml")
	overrides = {
		"task": args.task,
		"model_size": args.model_size,
		"checkpoint": str(args.checkpoint),
		"eval_episodes": args.episodes,
		"save_video": False,
		"compile": False,
		"enable_wandb": False,
		"seed": args.seed,
	}
	for key, value in overrides.items():
		cfg[key] = value
	cfg.work_dir = REPO / "logs" / cfg.task / str(cfg.seed) / cfg.exp_name
	cfg.task_title = cfg.task.replace("-", " ").title()
	cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
	from common import MODEL_SIZE, TASK_SET
	if cfg.get("model_size", None) is not None:
		assert cfg.model_size in MODEL_SIZE.keys(), f"Invalid model size {cfg.model_size}"
		for k, v in MODEL_SIZE[cfg.model_size].items():
			cfg[k] = v
	if cfg.task == "mt30" and cfg.get("model_size", None) == 19:
		cfg.latent_dim = 512
	cfg.multitask = cfg.task in TASK_SET.keys()
	if cfg.multitask:
		cfg.task_title = cfg.task.upper()
		cfg.task_dim = 96 if cfg.task == "mt80" or cfg.get("model_size", 5) in {1, 317} else 64
	else:
		cfg.task_dim = 0
	cfg.tasks = TASK_SET.get(cfg.task, [cfg.task])
	set_seed(cfg.seed)
	cfg.device = "cpu"

	env = make_env(cfg)
	agent = TDMPC2(cfg)
	agent.load(args.checkpoint)
	agent.model.eval()

	pred_sources: list[torch.Tensor] = []
	pred_targets: list[torch.Tensor] = []
	obs_sources: list[torch.Tensor] = []
	obs_targets: list[torch.Tensor] = []
	episode_rewards: list[float] = []
	episode_lengths: list[int] = []

	with torch.no_grad():
		for _ in range(args.episodes):
			obs, done, ep_reward, t = env.reset(), False, 0.0, 0
			while not done and t < args.max_steps:
				obs_device = obs.to(agent.device, non_blocking=True).unsqueeze(0)
				z = agent.model.encode(obs_device, None)
				action = agent.act(obs, t0=t == 0, eval_mode=True)
				action_device = action.to(agent.device, non_blocking=True).unsqueeze(0)
				pred_next_z = agent.model.next(z, action_device, None)
				next_obs, reward, done, info = env.step(action)
				next_obs_device = next_obs.to(agent.device, non_blocking=True).unsqueeze(0)
				next_z = agent.model.encode(next_obs_device, None)

				pred_sources.append(_assignments(z, cfg.simnorm_dim).squeeze(0).detach().cpu())
				pred_targets.append(_assignments(pred_next_z, cfg.simnorm_dim).squeeze(0).detach().cpu())
				obs_sources.append(_assignments(z, cfg.simnorm_dim).squeeze(0).detach().cpu())
				obs_targets.append(_assignments(next_z, cfg.simnorm_dim).squeeze(0).detach().cpu())

				obs = next_obs
				ep_reward += float(reward)
				t += 1
			episode_rewards.append(ep_reward)
			episode_lengths.append(t)

	pred_source = torch.stack(pred_sources)
	pred_target = torch.stack(pred_targets)
	obs_source = torch.stack(obs_sources)
	obs_target = torch.stack(obs_targets)
	pred_flow, pred_metrics = _flow_metrics(pred_source, pred_target)
	obs_flow, obs_metrics = _flow_metrics(obs_source, obs_target)

	args.output_dir.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(
		args.output_dir / "structural_diagnostics.npz",
		predicted_flow=pred_flow,
		observed_encoded_flow=obs_flow,
		episode_rewards=np.array(episode_rewards),
		episode_lengths=np.array(episode_lengths),
	)

	summary = {
		"task": args.task,
		"checkpoint": str(args.checkpoint),
		"episodes": args.episodes,
		"transitions": int(pred_source.shape[0]),
		"mean_episode_reward": float(np.mean(episode_rewards)),
		"std_episode_reward": float(np.std(episode_rewards)),
		"mean_episode_length": float(np.mean(episode_lengths)),
		"predicted_transition_flow": pred_metrics,
		"observed_encoded_transition_flow": obs_metrics,
	}
	(args.output_dir / "structural_diagnostics.json").write_text(json.dumps(summary, indent=2) + "\n")

	with (args.output_dir / "structural_diagnostics_summary.csv").open("w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["flow_type", "metric", "value"])
		for flow_type, metrics in (
			("predicted_transition_flow", pred_metrics),
			("observed_encoded_transition_flow", obs_metrics),
		):
			for key, value in metrics.items():
				writer.writerow([flow_type, key, value])
		writer.writerow(["evaluation", "mean_episode_reward", summary["mean_episode_reward"]])
		writer.writerow(["evaluation", "std_episode_reward", summary["std_episode_reward"]])
		writer.writerow(["evaluation", "transitions", summary["transitions"]])

	_save_heatmap(
		pred_flow,
		args.output_dir / "predicted_transition_flow.pdf",
		args.output_dir / "predicted_transition_flow.png",
		"Predicted latent transition flow",
	)
	_save_heatmap(
		obs_flow,
		args.output_dir / "observed_encoded_transition_flow.pdf",
		args.output_dir / "observed_encoded_transition_flow.png",
		"Encoded rollout transition flow",
	)
	_save_weight_figure(
		pred_flow,
		obs_flow,
		args.output_dir / "structural_weights.pdf",
		args.output_dir / "structural_weights.png",
	)
	return summary


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--checkpoint", type=Path, required=True)
	parser.add_argument("--task", default="acrobot-swingup")
	parser.add_argument("--model-size", type=int, default=1)
	parser.add_argument("--episodes", type=int, default=10)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--max-steps", type=int, default=1000)
	parser.add_argument("--output-dir", type=Path, required=True)
	args = parser.parse_args()
	summary = diagnose(args)
	print(json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()
