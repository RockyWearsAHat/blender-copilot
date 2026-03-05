#!/usr/bin/env python3
"""Self-improvement (Phase 2) using Blender rollouts + reconstruction scoring.

This is the reconstruction-fidelity version of `training/self_improve_policy.py`.

Loop:
  1) Sample prompts from the tokenized geometry dataset (jsonl)
  2) Run closed-loop Blender rollouts for each prompt
  3) Score each rollout vs its prompt-specific reference mesh using
     Chamfer/F-score (dominant) + a small trajectory-quality term
  4) Keep best trajectories in a replay buffer
  5) Train the policy by imitation on buffer trajectories

Architecture compliance:
- state is compact numeric mesh stats + compact prompt fingerprint
- finite action grammar
- small policy transformer
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _detect_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _autocast_ctx(device: torch.device, mp: str):
    mp = (mp or "fp32").lower()
    if mp == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    if device.type == "mps":
        dtype = torch.bfloat16 if mp in {"bf16", "bfloat16"} else torch.float16
        return torch.autocast(device_type="mps", dtype=dtype)
    if device.type == "cuda":
        dtype = torch.bfloat16 if mp in {"bf16", "bfloat16"} else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return torch.autocast(device_type=device.type, enabled=False)


@dataclass
class Episode:
    states: np.ndarray       # (T, S)
    action_type: np.ndarray  # (T,)
    action_param: np.ndarray  # (T,)
    total_reward: float
    done: bool
    prompt: str
    out_dir: str


class ReplayBuffer:
    def __init__(self, capacity_episodes: int = 256):
        self._cap = int(capacity_episodes)
        self._eps: list[Episode] = []

    def add_many(self, episodes: list[Episode]):
        self._eps.extend(episodes)
        self._eps.sort(key=lambda e: (e.total_reward, float(e.done)), reverse=True)
        if len(self._eps) > self._cap:
            self._eps = self._eps[: self._cap]

    def __len__(self) -> int:
        return len(self._eps)

    def sample_batch(
        self,
        rng: np.random.Generator,
        batch_size: int,
        seq_len: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._eps:
            raise RuntimeError("ReplayBuffer is empty")

        bsz = int(batch_size)
        T = int(seq_len)
        S = int(self._eps[0].states.shape[-1])

        states = np.zeros((bsz, T, S), dtype=np.float32)
        a_type = np.zeros((bsz, T), dtype=np.int64)
        a_param = np.zeros((bsz, T), dtype=np.int64)

        for b in range(bsz):
            ep = self._eps[int(rng.integers(0, len(self._eps)))]
            n = int(ep.states.shape[0])
            if n >= T:
                start = int(rng.integers(0, n - T + 1))
                states[b] = ep.states[start : start + T]
                a_type[b] = ep.action_type[start : start + T]
                a_param[b] = ep.action_param[start : start + T]
            else:
                states[b, :n] = ep.states
                a_type[b, :n] = ep.action_type
                a_param[b, :n] = ep.action_param
                states[b, n:] = ep.states[n - 1]
                a_type[b, n:] = ep.action_type[n - 1]
                a_param[b, n:] = ep.action_param[n - 1]

        return states, a_type, a_param


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _stats_to_feat(stats: dict, prompt: str) -> np.ndarray:
    from policy.state import MeshState, normalize_goal, normalize_state_vector
    from policy.prompt_parser import parse_prompt_to_goal

    bb = stats.get("bounding_box", {}) if isinstance(stats, dict) else {}
    st = MeshState(
        vertex_count=int(stats.get("vertex_count", 0)),
        face_count=int(stats.get("face_count", 0)),
        edge_count=int(stats.get("edge_count", 0)),
        bbox_x=float(bb.get("x", 0.0)),
        bbox_y=float(bb.get("y", 0.0)),
        bbox_z=float(bb.get("z", 0.0)),
        avg_edge_length=float(stats.get("avg_edge_length", 0.0)),
        manifold_flag=float(stats.get("manifold_flag", 0.0)),
        symmetry_score=float(stats.get("symmetry_score", 0.0)),
        selected_face_count=int(stats.get("selected_face_count", 0)),
    )
    goal = parse_prompt_to_goal(prompt).goal
    svec = normalize_state_vector(st.as_vector())
    gvec = normalize_goal(goal)
    return np.concatenate([svec, gvec], axis=0).astype(np.float32, copy=False)


def _build_episode_from_rollout(out_dir: Path, prompt: str, total_reward: float) -> Episode | None:
    out_dir = Path(out_dir)
    ready = _read_json(out_dir / "ready.json")
    if not isinstance(ready, dict):
        return None
    stats0 = ready.get("stats")
    if not isinstance(stats0, dict):
        return None

    action_files = sorted(out_dir.glob("action_*.json"))
    state_files = sorted(out_dir.glob("state_*.json"))
    T = min(len(action_files), len(state_files))
    if T <= 0:
        return None

    # Inputs are the *pre-action* states:
    #  step 0 uses ready.stats; step i uses state_{i-1}.stats.
    feats: list[np.ndarray] = []
    at: list[int] = []
    ap: list[int] = []

    prev_stats = stats0
    for i in range(T):
        a = _read_json(action_files[i]) or {}
        feats.append(_stats_to_feat(prev_stats, prompt))
        at.append(int(a.get("action_type", 0)))
        ap.append(int(a.get("param", 0)))

        nxt = _read_json(state_files[i]) or {}
        nxt_stats = nxt.get("stats") if isinstance(nxt, dict) else None
        if not isinstance(nxt_stats, dict):
            break
        prev_stats = nxt_stats

    if not feats:
        return None

    return Episode(
        states=np.stack(feats, axis=0).astype(np.float32, copy=False),
        action_type=np.asarray(at, dtype=np.int64),
        action_param=np.asarray(ap, dtype=np.int64),
        total_reward=float(total_reward),
        done=True,
        prompt=str(prompt),
        out_dir=str(out_dir),
    )


def _run_rollout(
    *,
    ckpt: Path,
    out_dir: Path,
    prompt: str,
    steps: int,
    seed: int,
    blender: str,
    device: str,
    temperature: float,
    top_k: int,
    deterministic: bool,
) -> None:
    # Defensive cleanup: if the directory already exists (common when rerunning
    # with deterministic RNG), remove handshake/artifact files so the rollout
    # can't accidentally read a stale ready.json from a previous run.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("ready.json"):
        try:
            p.unlink()
        except Exception:
            pass
    for p in out_dir.glob("action_*.json"):
        try:
            p.unlink()
        except Exception:
            pass
    for p in out_dir.glob("state_*.json"):
        try:
            p.unlink()
        except Exception:
            pass
    for p in (out_dir / "stats_final.json", out_dir / "mesh.obj", out_dir / "scene.blend", out_dir / "early_stop.json"):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    exe = PROJECT_ROOT / "scripts" / "rollout_policy_closed_loop.py"
    cmd = [
        sys.executable,
        str(exe),
        "--ckpt",
        str(ckpt),
        "--out-dir",
        str(out_dir),
        "--steps",
        str(int(steps)),
        "--seed",
        str(int(seed)),
        "--prompt",
        str(prompt),
        "--blender",
        str(blender),
        "--device",
        str(device),
        "--temperature",
        str(float(temperature)),
        "--top-k",
        str(int(top_k)),
    ]
    if deterministic:
        cmd.append("--deterministic")

    # We don't need stdout; the rollout writes artifacts into out_dir.
    # But we capture output so we can persist it on failure.
    subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )


def main() -> int:
    from models.policy_transformer import PolicyTransformer
    from policy.dataset import SyntheticImitationStream
    from policy.goal_sampler import ListGoalSampler, load_goal_records_from_geometry_jsonl

    from evaluation.prompt_reference import find_reference_for_prompt
    from evaluation.reconstruction_scoring import (
        compute_reconstruction_metrics,
        score_rollout_trajectory,
    )

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("config.policy_m3_quick.yaml"))
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--blender", type=str, default="/Applications/Blender.app/Contents/MacOS/Blender")

    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--episodes-per-iter", type=int, default=8)
    p.add_argument("--keep-frac", type=float, default=0.25)
    p.add_argument("--buffer-episodes", type=int, default=128)

    p.add_argument("--rollout-steps", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.2)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--deterministic", action="store_true")

    p.add_argument("--train-steps", type=int, default=200)
    p.add_argument("--mix-teacher", type=float, default=0.3)

    p.add_argument(
        "--goals-geometry-jsonl",
        type=Path,
        default=Path("data/datasets/geometry/train.jsonl"),
        help="Prompt+reference source; uses prompt text as the rollout condition.",
    )
    p.add_argument("--goals-max-records", type=int, default=50_000)

    p.add_argument(
        "--reference-geometry-dir",
        type=Path,
        default=Path("data/datasets/geometry"),
        help="Directory containing train/val/test.jsonl reference tokens.",
    )
    p.add_argument("--reference-splits", type=str, default="train,val,test")
    p.add_argument("--surface-points", type=int, default=2048)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    device = _detect_device(args.device)

    recs = load_goal_records_from_geometry_jsonl(
        path=Path(args.goals_geometry_jsonl),
        max_records=int(args.goals_max_records),
    )
    # Only keep records with usable prompt text.
    recs = [r for r in recs if isinstance(r.text, str) and r.text.strip()]
    if not recs:
        raise RuntimeError("No usable prompt records found (missing text)")
    goal_sampler = ListGoalSampler(recs)

    payload = torch.load(args.ckpt, map_location="cpu")
    ckpt_cfg = payload.get("config") or cfg

    model = PolicyTransformer(
        state_dim=int(model_cfg.get("state_dim", 12)),
        hidden_size=int(model_cfg.get("hidden_size", 256)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        action_type_vocab=int(model_cfg.get("action_type_vocab", 11)),
        action_param_vocab=int(model_cfg.get("action_param_vocab", 32)),
        max_seq_len=int(model_cfg.get("max_seq_len", 128)),
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)

    lr = float(train_cfg.get("learning_rate", 3e-4))
    wd = float(train_cfg.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    type_loss_fn = torch.nn.CrossEntropyLoss()
    param_loss_fn = torch.nn.CrossEntropyLoss()

    seq_len = int(data_cfg.get("seq_len", 64))
    batch_size = int(data_cfg.get("batch_size", 32))
    state_dim = int(model_cfg.get("state_dim", 12))

    teacher = SyntheticImitationStream(
        seed=int(data_cfg.get("seed", 0)),
        seq_len=seq_len,
        batch_size=batch_size,
        state_dim=state_dim,
        goal_sampler=goal_sampler,
    )
    teacher_it = iter(teacher)

    mp = str(train_cfg.get("mixed_precision", "fp16"))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(capacity_episodes=int(args.buffer_episodes))

    out_ckpt_dir = PROJECT_ROOT / str(train_cfg.get("checkpoint_dir", "checkpoints/policy_goal"))
    out_ckpt_dir.mkdir(parents=True, exist_ok=True)
    rollout_root = out_ckpt_dir / "self_improve_reconstruction_rollouts"
    rollout_root.mkdir(parents=True, exist_ok=True)

    def save_latest(step: int) -> Path:
        path = out_ckpt_dir / "latest_self_improve_reconstruction.pt"
        torch.save(
            {"step": int(step), "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": ckpt_cfg},
            path,
        )
        return path

    current_ckpt = Path(args.ckpt)
    global_step = int(payload.get("step", 0))

    splits = tuple(s.strip() for s in str(args.reference_splits).split(",") if s.strip()) or ("train", "val", "test")

    for it in range(1, int(args.iters) + 1):
        model.eval()
        episodes: list[Episode] = []
        t_it0 = time.time()

        for epi in range(int(args.episodes_per_iter)):
            # Sample a prompt record.
            rec = recs[int(rng.integers(0, len(recs)))]
            prompt = str(rec.text)

            # Unique rollout dir (avoid collisions across reruns).
            out_dir = rollout_root / f"it{it:03d}_ep{epi:03d}_{time.time_ns()}"
            out_dir.mkdir(parents=True, exist_ok=True)

            seed = int(rng.integers(0, 2**31 - 1))
            try:
                _run_rollout(
                    ckpt=current_ckpt,
                    out_dir=out_dir,
                    prompt=prompt,
                    steps=int(args.rollout_steps),
                    seed=seed,
                    blender=str(args.blender),
                    device=str(args.device),
                    temperature=float(args.temperature),
                    top_k=int(args.top_k),
                    deterministic=bool(args.deterministic),
                )
            except subprocess.TimeoutExpired as e:
                (out_dir / "rollout_failed.txt").write_text(
                    f"TimeoutExpired: {e}\n"
                )
                continue
            except subprocess.CalledProcessError as e:
                # Persist stderr/stdout for debugging; then skip the episode.
                msg = [
                    f"CalledProcessError: returncode={e.returncode}",
                    "\n--- CMD ---\n" + " ".join(str(x) for x in (e.cmd or [])),
                ]
                if e.stdout:
                    msg.append("\n--- STDOUT (tail) ---\n" + e.stdout[-4000:])
                if e.stderr:
                    msg.append("\n--- STDERR (tail) ---\n" + e.stderr[-4000:])
                (out_dir / "rollout_failed.txt").write_text("\n".join(msg))
                continue

            obj_path = out_dir / "mesh.obj"
            ref = find_reference_for_prompt(prompt, geometry_dir=Path(args.reference_geometry_dir), splits=splits)
            expected = None
            expected_reason = ""
            if ref is not None:
                from evaluation.reconstruction_scoring import infer_expected_base_primitives_from_source

                expected, expected_reason = infer_expected_base_primitives_from_source(ref.source)

            traj = score_rollout_trajectory(
                out_dir,
                expected_base_primitives=expected,
                expected_base_reason=str(expected_reason),
            )

            total = 0.0
            if ref is not None and obj_path.exists():
                metrics = compute_reconstruction_metrics(
                    obj_path=obj_path,
                    reference_tokens=ref.tokens,
                    n_surface_points=int(args.surface_points),
                    normalize=True,
                )
                try:
                    f02 = float(metrics.get("f_score_002", {}).get("f_score", 0.0))
                except Exception:
                    f02 = 0.0
                total = float(0.85 * f02 + 0.15 * float(traj.path_score))
                (out_dir / "self_improve_score.json").write_text(
                    json.dumps(
                        {"prompt": prompt, "f_score_002": f02, "path_score": float(traj.path_score), "total": total},
                        indent=2,
                    )
                )
            else:
                total = float(0.15 * float(traj.path_score))
                (out_dir / "self_improve_score.json").write_text(
                    json.dumps(
                        {"prompt": prompt, "reference_found": bool(ref is not None), "path_score": float(traj.path_score), "total": total},
                        indent=2,
                    )
                )

            ep = _build_episode_from_rollout(out_dir, prompt, total_reward=total)
            if ep is not None:
                episodes.append(ep)

        episodes.sort(key=lambda e: e.total_reward, reverse=True)
        keep_n = max(1, int(round(float(args.keep_frac) * max(1, len(episodes)))))
        kept = episodes[:keep_n]
        buffer.add_many(kept)

        # Train on buffer (+ optional teacher)
        model.train()
        t0 = time.time()
        for _ in range(int(args.train_steps)):
            use_teacher = (float(rng.random()) < float(args.mix_teacher)) or (len(buffer) < 4)
            if use_teacher:
                batch = next(teacher_it)
                states = batch.states.to(device=device, dtype=torch.float32)
                a_type = batch.action_type.to(device=device)
                a_param = batch.action_param.to(device=device)
            else:
                s, at, ap = buffer.sample_batch(rng, batch_size=batch_size, seq_len=seq_len)
                states = torch.from_numpy(s).to(device=device, dtype=torch.float32)
                a_type = torch.from_numpy(at).to(device=device)
                a_param = torch.from_numpy(ap).to(device=device)

            optimizer.zero_grad(set_to_none=True)
            with _autocast_ctx(device, mp):
                type_logits, param_logits = model(states)
                bsz, T = a_type.shape
                type_loss = type_loss_fn(type_logits.reshape(bsz * T, -1), a_type.reshape(-1))
                param_loss = param_loss_fn(param_logits.reshape(bsz * T, -1), a_param.reshape(-1))
                loss = type_loss + 0.5 * param_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            global_step += 1

        dt_train = max(1e-6, time.time() - t0)
        dt_it = max(1e-6, time.time() - t_it0)

        best = kept[0] if kept else None
        msg = {
            "iter": int(it),
            "episodes": int(len(episodes)),
            "kept": int(len(kept)),
            "buffer_eps": int(len(buffer)),
            "best_total": float(best.total_reward) if best else 0.0,
            "best_prompt": str(best.prompt) if best else None,
            "rollout_sps": float(int(args.episodes_per_iter) / dt_it),
            "train_sps": float(int(args.train_steps) / dt_train),
            "device": device.type,
            "global_step": int(global_step),
            "ckpt": str(current_ckpt),
        }
        print(json.dumps(msg))

        current_ckpt = save_latest(global_step)

    print("OK: self-improvement (reconstruction) completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
