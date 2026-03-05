#!/usr/bin/env python3
"""Phase 2 (ARCHITECTURE.md): self-improvement via generate→score→keep-best.

This is a lightweight, local alternative to high-variance RL ("GRPO" etc).
It does:
  1) Roll out the *current* policy in the compact MeshStatsEnv
  2) Score trajectories with env rewards (proxy for mesh metrics)
  3) Keep top trajectories in a replay buffer
  4) Train the policy by imitation on the buffer (optionally mixed with
     the original synthetic teacher stream for stability)

It stays architecture-compliant:
- compact numeric state (no raw meshes)
- finite action grammar
- small transformer

Usage:
  /path/to/python training/self_improve_policy.py \
    --config config.policy_m3_quick.yaml \
    --ckpt checkpoints/policy_goal/latest.pt \
    --device auto \
    --iters 20
"""

from __future__ import annotations

import argparse
import json
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
    states: np.ndarray      # (T, S)
    action_type: np.ndarray  # (T,)
    action_param: np.ndarray  # (T,)
    total_reward: float
    done: bool


class ReplayBuffer:
    def __init__(self, capacity_episodes: int = 512):
        self._cap = int(capacity_episodes)
        self._eps: list[Episode] = []

    def add_many(self, episodes: list[Episode]):
        self._eps.extend(episodes)
        # Keep best by reward (deterministic ordering)
        self._eps.sort(key=lambda e: (e.total_reward, float(e.done)), reverse=True)
        if len(self._eps) > self._cap:
            self._eps = self._eps[: self._cap]

    def __len__(self) -> int:
        return len(self._eps)

    def sample_batch(self, rng: np.random.Generator, batch_size: int, seq_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            # If episode shorter than T, pad by repeating last state/action.
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


def _rollout_one(
    *,
    model: torch.nn.Module,
    device: torch.device,
    env_seed: int,
    goal_seed: int,
    horizon: int,
    epsilon: float,
    deterministic: bool,
    action_type_vocab: int,
    action_param_vocab: int,
    state_dim: int,
    goal_sampler,
) -> Episode:
    from policy.actions import Action
    from policy.env import MeshStatsEnv
    from policy.state import normalize_goal, normalize_state_vector

    rng = np.random.default_rng(goal_seed)
    env = MeshStatsEnv(seed=int(env_seed))

    goal = goal_sampler.sample(rng)
    gvec = normalize_goal(goal)

    st = env.reset()

    states_list: list[np.ndarray] = []
    at_list: list[int] = []
    ap_list: list[int] = []

    total_reward = 0.0
    done = False

    for t in range(int(horizon)):
        svec = normalize_state_vector(st.as_vector())
        feat = np.concatenate([svec, gvec], axis=0).astype(np.float32, copy=False)
        if feat.shape[0] != state_dim:
            raise RuntimeError(f"state_dim mismatch: feat={feat.shape[0]} expected={state_dim}")

        # Exploration: random action sometimes.
        if float(rng.random()) < float(epsilon):
            a_type = int(rng.integers(0, action_type_vocab))
            a_param = int(rng.integers(0, action_param_vocab))
        else:
            with torch.no_grad():
                x = torch.from_numpy(np.stack(states_list + [feat], axis=0)[None, :, :]).to(device=device, dtype=torch.float32)
                type_logits, param_logits = model(x)
                tl = type_logits[0, -1]
                pl = param_logits[0, -1]
                if deterministic:
                    a_type = int(torch.argmax(tl).item())
                    a_param = int(torch.argmax(pl).item())
                else:
                    a_type = int(torch.distributions.Categorical(logits=tl).sample().item())
                    a_param = int(torch.distributions.Categorical(logits=pl).sample().item())

        act = Action(action_type=a_type, param=a_param).clamp()

        sr = env.step(st, act, goal)
        st = sr.state
        total_reward += float(sr.reward)

        states_list.append(feat)
        at_list.append(int(act.action_type))
        ap_list.append(int(act.param))

        if sr.done:
            done = True
            break

    return Episode(
        states=np.stack(states_list, axis=0).astype(np.float32, copy=False),
        action_type=np.asarray(at_list, dtype=np.int64),
        action_param=np.asarray(ap_list, dtype=np.int64),
        total_reward=float(total_reward),
        done=bool(done),
    )


def main() -> int:
    from models.policy_transformer import PolicyTransformer
    from policy.dataset import SyntheticImitationStream
    from policy.goal_sampler import (
        ListGoalSampler,
        UniformGoalSampler,
        load_goal_records_from_geometry_jsonl,
        load_goal_records_from_mesh_cache_dir,
    )

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("config.policy_m3_quick.yaml"))
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])

    p.add_argument("--iters", type=int, default=10, help="Self-improve iterations (generate+train rounds).")
    p.add_argument("--episodes-per-iter", type=int, default=64)
    p.add_argument("--keep-frac", type=float, default=0.25)
    p.add_argument("--buffer-episodes", type=int, default=512)

    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--epsilon", type=float, default=0.10, help="Exploration probability for random actions.")
    p.add_argument("--deterministic", action="store_true", help="Use greedy argmax actions (no sampling).")

    p.add_argument("--train-steps", type=int, default=200, help="Gradient steps per iteration.")
    p.add_argument("--mix-teacher", type=float, default=0.5, help="Fraction of batches from teacher stream (0..1).")

    p.add_argument("--goals-geometry-jsonl", type=Path, default=None, help="Sample goals from data/datasets/geometry/*.jsonl")
    p.add_argument("--goals-mesh-cache-dir", type=Path, default=None, help="Sample goals from data/processed/.mesh_cache/*.pt")
    p.add_argument("--goals-max-records", type=int, default=50_000)

    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    device = _detect_device(args.device)

    goal_sampler = None
    if args.goals_geometry_jsonl is not None:
        recs = load_goal_records_from_geometry_jsonl(path=args.goals_geometry_jsonl, max_records=int(args.goals_max_records))
        goal_sampler = ListGoalSampler(recs)
    elif args.goals_mesh_cache_dir is not None:
        recs = load_goal_records_from_mesh_cache_dir(cache_dir=args.goals_mesh_cache_dir, max_files=int(args.goals_max_records))
        goal_sampler = ListGoalSampler(recs)
    else:
        goal_src = data_cfg.get("goal_source")
        if isinstance(goal_src, dict):
            src_type = str(goal_src.get("type") or "").strip().lower()
            src_path = goal_src.get("path")
            if src_type in {"geometry_jsonl", "geometry"} and isinstance(src_path, str):
                recs = load_goal_records_from_geometry_jsonl(
                    path=Path(src_path),
                    max_records=int(goal_src.get("max_records", 50_000)),
                )
                goal_sampler = ListGoalSampler(recs)
            elif src_type in {"mesh_cache", "mesh_cache_dir"} and isinstance(src_path, str):
                recs = load_goal_records_from_mesh_cache_dir(
                    cache_dir=Path(src_path),
                    max_files=int(goal_src.get("max_files", 20_000)),
                    min_quality_weight=(float(goal_src["min_quality_weight"]) if "min_quality_weight" in goal_src else None),
                )
                goal_sampler = ListGoalSampler(recs)
            elif src_type in {"uniform", "random"}:
                goal_sampler = UniformGoalSampler()
    if goal_sampler is None:
        goal_sampler = UniformGoalSampler()

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
    action_type_vocab = int(model_cfg.get("action_type_vocab", 11))
    action_param_vocab = int(model_cfg.get("action_param_vocab", 32))

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

    out_dir = PROJECT_ROOT / str(train_cfg.get("checkpoint_dir", "checkpoints/policy_goal"))
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_latest(step: int):
        torch.save(
            {
                "step": int(step),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": ckpt_cfg,
            },
            out_dir / "latest_self_improve.pt",
        )

    global_step = int(payload.get("step", 0))

    for it in range(1, int(args.iters) + 1):
        model.eval()
        episodes: list[Episode] = []
        base_env_seed = int(rng.integers(0, 2**31 - 1))

        for i in range(int(args.episodes_per_iter)):
            ep = _rollout_one(
                model=model,
                device=device,
                env_seed=base_env_seed + i,
                goal_seed=int(rng.integers(0, 2**31 - 1)),
                horizon=int(args.horizon),
                epsilon=float(args.epsilon),
                deterministic=bool(args.deterministic),
                action_type_vocab=action_type_vocab,
                action_param_vocab=action_param_vocab,
                state_dim=state_dim,
                goal_sampler=goal_sampler,
            )
            episodes.append(ep)

        episodes.sort(key=lambda e: (e.total_reward, float(e.done)), reverse=True)
        keep_n = max(1, int(round(float(args.keep_frac) * len(episodes))))
        kept = episodes[:keep_n]
        buffer.add_many(kept)

        # Train on buffer (+ optionally teacher) for stability
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

        dt = max(1e-6, time.time() - t0)
        best = kept[0]
        msg = {
            "iter": int(it),
            "buffer_eps": int(len(buffer)),
            "kept": int(keep_n),
            "best_reward": float(best.total_reward),
            "best_done": bool(best.done),
            "train_steps": int(args.train_steps),
            "train_sps": float(args.train_steps / dt),
            "device": device.type,
            "global_step": int(global_step),
        }
        print(json.dumps(msg))

        save_latest(global_step)

    print("OK: self-improvement completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
