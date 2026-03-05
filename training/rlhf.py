"""Minimal RLHF/feedback plumbing for the local inference server.

This is intentionally lightweight:
- Stores human feedback to a JSONL log under data/feedback/
- Supports A/B comparison generation by sampling model outputs
- Does NOT (yet) fine-tune the model online

Reason: keep the system modular and deterministic, while ensuring the
addon feedback UI and server endpoints function end-to-end.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class FeedbackStats:
    total_feedback_ever: int = 0
    total_pairwise: int = 0
    total_accept: int = 0
    total_reject: int = 0
    total_rating: int = 0


class FeedbackBuffer:
    def __init__(self, log_path: Path):
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._stats = FeedbackStats()

        # Best-effort: load existing counts
        try:
            if self._log_path.exists():
                with self._log_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        self._stats.total_feedback_ever += 1
                        kind = (rec.get("kind") or "").lower()
                        if kind == "pairwise":
                            self._stats.total_pairwise += 1
                        elif kind == "accept":
                            self._stats.total_accept += 1
                        elif kind == "reject":
                            self._stats.total_reject += 1
                        elif kind == "rating":
                            self._stats.total_rating += 1
        except Exception:
            pass

    def add(self, record: dict[str, Any]) -> None:
        with self._lock:
            kind = (record.get("kind") or "").lower()
            if kind == "pairwise":
                self._stats.total_pairwise += 1
            elif kind == "accept":
                self._stats.total_accept += 1
            elif kind == "reject":
                self._stats.total_reject += 1
            elif kind == "rating":
                self._stats.total_rating += 1

            self._stats.total_feedback_ever += 1

            record = dict(record)
            record.setdefault("ts", time.time())

            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_feedback_ever": int(self._stats.total_feedback_ever),
                "pairwise": int(self._stats.total_pairwise),
                "accept": int(self._stats.total_accept),
                "reject": int(self._stats.total_reject),
                "rating": int(self._stats.total_rating),
                "log_path": str(self._log_path),
            }


class RLHFTrainer:
    """Minimal trainer facade used by inference/server.py.

    For now, it only records feedback and can generate comparison pairs.
    Online updates are intentionally not implemented.
    """

    def __init__(
        self,
        model: Any,
        config: dict[str, Any],
        *,
        text_tokenizer: Any,
        mesh_tokenizer: Any,
        device: torch.device,
        feedback_log_path: str | None = None,
    ):
        self.model = model
        self.config = config or {}
        self.text_tokenizer = text_tokenizer
        self.mesh_tokenizer = mesh_tokenizer
        self.device = device

        default_log = Path("data") / "feedback" / "rlhf_feedback.jsonl"
        self.feedback_buffer = FeedbackBuffer(Path(feedback_log_path) if feedback_log_path else default_log)

        self._lock = threading.Lock()
        self._updates = 0

    # ---- feedback ingestion -------------------------------------------------

    def add_pairwise_feedback(self, prompt: str, chosen_tokens: list, rejected_tokens: list, metadata: dict | None = None) -> None:
        self.feedback_buffer.add({
            "kind": "pairwise",
            "prompt": prompt,
            "chosen_tokens": chosen_tokens,
            "rejected_tokens": rejected_tokens,
            "metadata": metadata or {},
        })

    def add_accept_feedback(self, prompt: str, tokens: list, metadata: dict | None = None) -> None:
        self.feedback_buffer.add({
            "kind": "accept",
            "prompt": prompt,
            "tokens": tokens,
            "metadata": metadata or {},
        })

    def add_reject_feedback(self, prompt: str, tokens: list, metadata: dict | None = None) -> None:
        self.feedback_buffer.add({
            "kind": "reject",
            "prompt": prompt,
            "tokens": tokens,
            "metadata": metadata or {},
        })

    def add_rating_feedback(self, prompt: str, tokens: list, rating: float, metadata: dict | None = None) -> None:
        self.feedback_buffer.add({
            "kind": "rating",
            "prompt": prompt,
            "tokens": tokens,
            "rating": float(rating),
            "metadata": metadata or {},
        })

    # ---- status/update ------------------------------------------------------

    def maybe_update(self) -> dict[str, Any]:
        # Placeholder for future online updates.
        return {"status": "noop", "updates": int(self._updates)}

    def get_status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "updates": int(self._updates),
            "feedback": self.feedback_buffer.get_stats(),
            "note": "Feedback is recorded; online updates not implemented yet.",
        }

    def evaluate(self) -> dict[str, Any]:
        # Minimal stub.
        return {"status": "ok", "feedback": self.feedback_buffer.get_stats()}

    # ---- comparison generation ---------------------------------------------

    def _encode_text(self, text: str, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        tt = self.text_tokenizer
        if tt is not None and hasattr(tt, "encode_padded"):
            ids, mask = tt.encode_padded(text, max_length=max_length)
        else:
            ids = [ord(c) % 32000 for c in (text or "")[:max_length]]
            mask = [1] * len(ids)
            ids += [0] * (max_length - len(ids))
            mask += [0] * (max_length - len(mask))

        return (
            torch.tensor([ids], dtype=torch.long, device=self.device),
            torch.tensor([mask], dtype=torch.float, device=self.device),
        )

    def _generate_tokens(self, prompt: str, *, temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9, max_faces: int = 2048) -> list[int]:
        text_max = getattr(self.model.text_encoder, "max_length", 256)
        if hasattr(self.model.text_encoder, "pos_embed"):
            text_max = self.model.text_encoder.pos_embed.num_embeddings

        text_ids, text_mask = self._encode_text(prompt, max_length=int(text_max))

        # Compute max token budget from tokenizer constraints
        geo_dec = getattr(self.model, "geometry_decoder", None)
        if geo_dec is not None:
            max_seq = getattr(geo_dec, "max_seq_length", 16202)
        else:
            max_seq = getattr(getattr(self.model, "mesh_decoder", None), "max_seq_length", 18432)

        effective_max = min(int(max_faces), int((max_seq - 2) // 9))
        max_tokens = effective_max * 9 + 2

        with torch.no_grad():
            if hasattr(self.model, "generate_geometry"):
                t = self.model.generate_geometry(
                    text_ids, text_mask,
                    max_tokens=max_tokens,
                    temperature=float(temperature),
                    top_k=int(top_k),
                    top_p=float(top_p),
                    cfg_scale=2.0,
                )
            else:
                t = self.model.generate(
                    text_ids, text_mask,
                    max_tokens=max_tokens,
                    temperature=float(temperature),
                    top_k=int(top_k),
                    top_p=float(top_p),
                    cfg_scale=2.0,
                )

        return [int(x) for x in t[0].detach().cpu().tolist()]

    def select_comparison_pair(self, prompt: str, n_candidates: int = 4) -> dict[str, Any]:
        # Generate N candidates and pick the first two non-empty decoded meshes.
        candidates: list[dict[str, Any]] = []

        with self._lock:
            for _ in range(max(2, int(n_candidates))):
                tokens = self._generate_tokens(prompt)
                verts, faces = self.mesh_tokenizer.decode_tokens(tokens)
                if verts and faces:
                    candidates.append({"tokens": tokens, "vertices": verts, "faces": faces})
                if len(candidates) >= 2:
                    break

        if len(candidates) < 2:
            return {"status": "error", "error": "Model produced insufficient geometry for comparison"}

        option_a, option_b = candidates[0], candidates[1]
        return {
            "status": "ok",
            "option_a": option_a,
            "option_b": option_b,
            "feedback_stats": self.feedback_buffer.get_stats(),
        }
