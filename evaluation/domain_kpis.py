"""Professionalization KPI utilities.

Aggregates geometry and workflow-adjacent metrics into domain-level
promotion signals used by the professionalization plan.
"""

from __future__ import annotations

import json
from pathlib import Path


GOLD_SET_PATH = Path("data/eval/professional_gold_set.json")


def load_professional_gold_set(path: Path | None = None) -> list[dict]:
    target = path or GOLD_SET_PATH
    if not target.exists():
        return []
    with open(target) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return []


def summarize_professional_kpis(eval_results: dict) -> dict:
    summary = eval_results.get("summary", {})
    by_domain = summary.get("by_domain", {})

    out = {
        "promotion_ready": False,
        "domains_passed": 0,
        "domains_total": len(by_domain),
        "domain_scores": {},
        "gates": {
            "generation_rate_min": 0.8,
            "expectations_rate_min": 0.65,
            "validity_score_mean_min": 0.55,
            "domain_success_min": 0.6,
            "domain_pass_count_min": 4,
        },
    }

    if not by_domain:
        return out

    for domain, stats in by_domain.items():
        total = max(1, int(stats.get("total", 0)))
        generated_rate = float(stats.get("generated", 0)) / total
        success_rate = float(stats.get("valid", 0)) / total
        domain_score = 0.4 * generated_rate + 0.6 * success_rate
        out["domain_scores"][domain] = {
            "generated_rate": generated_rate,
            "success_rate": success_rate,
            "domain_score": domain_score,
        }

    gate = out["gates"]
    generation_rate = float(summary.get("generation_rate", 0.0))
    expectations_rate = float(summary.get("expectations_rate", 0.0))
    validity_mean = float(summary.get("validity_score_mean", 0.0))

    domains_passed = 0
    for domain_data in out["domain_scores"].values():
        if domain_data["success_rate"] >= gate["domain_success_min"]:
            domains_passed += 1

    out["domains_passed"] = domains_passed
    out["promotion_ready"] = (
        generation_rate >= gate["generation_rate_min"]
        and expectations_rate >= gate["expectations_rate_min"]
        and validity_mean >= gate["validity_score_mean_min"]
        and domains_passed >= gate["domain_pass_count_min"]
    )

    out["global"] = {
        "generation_rate": generation_rate,
        "expectations_rate": expectations_rate,
        "validity_score_mean": validity_mean,
    }

    return out


def get_wandb_kpi_log(kpis: dict, prefix: str = "pro") -> dict:
    payload = {
        f"{prefix}/promotion_ready": float(bool(kpis.get("promotion_ready", False))),
        f"{prefix}/domains_passed": float(kpis.get("domains_passed", 0)),
    }

    for domain, metrics in kpis.get("domain_scores", {}).items():
        payload[f"{prefix}/{domain}_generated_rate"] = float(metrics.get("generated_rate", 0.0))
        payload[f"{prefix}/{domain}_success_rate"] = float(metrics.get("success_rate", 0.0))
        payload[f"{prefix}/{domain}_score"] = float(metrics.get("domain_score", 0.0))

    global_metrics = kpis.get("global", {})
    for key, value in global_metrics.items():
        payload[f"{prefix}/{key}"] = float(value)

    return payload
