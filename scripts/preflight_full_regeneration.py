#!/usr/bin/env python3
"""Preflight gates for full dataset regeneration.

Purpose:
- Catch structural/material/face-index issues BEFORE long regeneration runs.
- Exercise master/training cache builders in dry-run mode.
- Exit non-zero on any hard gate failure.

This cannot mathematically guarantee "100% correctness", but it provides
strict fail-fast gates over the current corpus and build pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

DEFAULT_SOURCES = [
    "objaverse",
    "blendswap",
    "smutbase",
    "blender_official",
    "open3dlab",
    "github",
    "youtube",
]


@dataclass
class GateResult:
    name: str
    passed: bool
    details: str
    seconds: float


def _run(cmd: list[str], timeout: int) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, output


def _iter_source_jsons(sources: list[str]):
    for src in sources:
        src_dir = PROCESSED / src
        if not src_dir.exists():
            continue
        for fp in sorted(src_dir.glob("*.json")):
            yield src, fp


def gate_processed_integrity(sources: list[str], strict_fmi: bool) -> GateResult:
    t0 = time.time()

    n_files = 0
    n_objects = 0
    index_errors = 0
    malformed = 0
    mats_multi_no_fmi = 0
    mats_bad_fmi_len = 0

    for _, fp in _iter_source_jsons(sources):
        n_files += 1
        try:
            data = json.loads(fp.read_text())
        except Exception:
            malformed += 1
            continue

        objects = data.get("objects", []) if isinstance(data, dict) else []
        if not isinstance(objects, list):
            malformed += 1
            continue

        for obj in objects:
            if not isinstance(obj, dict):
                malformed += 1
                continue

            mesh = obj.get("mesh", {})
            if not isinstance(mesh, dict):
                malformed += 1
                continue

            verts = mesh.get("vertices", [])
            faces = mesh.get("faces", [])
            if not isinstance(verts, list) or not isinstance(faces, list):
                malformed += 1
                continue
            if not verts or not faces:
                continue

            n_objects += 1
            n_verts = len(verts)

            for face in faces:
                if not isinstance(face, list):
                    index_errors += 1
                    continue
                for vi in face:
                    if not isinstance(vi, int) or vi < 0 or vi >= n_verts:
                        index_errors += 1

            materials = obj.get("materials", [])
            if not isinstance(materials, list):
                materials = []
            fmi = mesh.get("face_material_indices", [])
            if not isinstance(fmi, list):
                fmi = []

            if len(materials) > 1 and len(fmi) == 0:
                mats_multi_no_fmi += 1
            if len(fmi) > 0 and len(fmi) != len(faces):
                mats_bad_fmi_len += 1

    hard_fail = malformed > 0 or index_errors > 0 or mats_bad_fmi_len > 0
    if strict_fmi:
        hard_fail = hard_fail or mats_multi_no_fmi > 0

    details = (
        f"files={n_files}, objects={n_objects}, malformed={malformed}, "
        f"index_errors={index_errors}, multi_mat_no_fmi={mats_multi_no_fmi}, "
        f"bad_fmi_len={mats_bad_fmi_len}, strict_fmi={strict_fmi}"
    )
    return GateResult(
        name="processed_integrity",
        passed=not hard_fail,
        details=details,
        seconds=time.time() - t0,
    )


def gate_master_dry_run(source: str | None, timeout: int) -> GateResult:
    t0 = time.time()
    cmd = [sys.executable, "scripts/build_master_cache.py", "--dry-run", "--no-quality"]
    if source:
        cmd.extend(["--source", source])
    code, out = _run(cmd, timeout)
    passed = code == 0
    details = f"exit={code}; tail={(out.splitlines()[-3:] if out else [])}"
    return GateResult("master_dry_run", passed, details, time.time() - t0)


def gate_training_dry_run(source: str | None, timeout: int) -> GateResult:
    t0 = time.time()
    cmd = [
        sys.executable,
        "scripts/build_training_cache.py",
        "--dry-run",
        "--max-per-file",
        "1",
        "--max-per-label",
        "20",
    ]
    if source:
        cmd.extend(["--source", source])
    code, out = _run(cmd, timeout)
    passed = code == 0
    details = f"exit={code}; tail={(out.splitlines()[-3:] if out else [])}"
    return GateResult("training_dry_run", passed, details, time.time() - t0)


def main() -> int:
    p = argparse.ArgumentParser(description="Preflight gates for full regeneration")
    p.add_argument("--sources", type=str, default=",".join(DEFAULT_SOURCES),
                   help="Comma-separated processed sources to validate")
    p.add_argument("--strict-fmi", action="store_true",
                   help="Fail if objects with >1 materials have no face_material_indices")
    p.add_argument("--pilot-source", type=str, default="objaverse",
                   help="Source to use for dry-run build pilot (or empty for all)")
    p.add_argument("--master-timeout", type=int, default=1800)
    p.add_argument("--training-timeout", type=int, default=1800)
    args = p.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    pilot = args.pilot_source.strip() or None

    gates = [
        gate_processed_integrity(sources, args.strict_fmi),
        gate_master_dry_run(pilot, args.master_timeout),
        gate_training_dry_run(pilot, args.training_timeout),
    ]

    print("\n=== Preflight Results ===")
    ok = True
    for g in gates:
        status = "PASS" if g.passed else "FAIL"
        print(f"[{status}] {g.name} ({g.seconds:.1f}s) :: {g.details}")
        ok = ok and g.passed

    if ok:
        print("\nPREFLIGHT_STATUS=PASS")
        return 0

    print("\nPREFLIGHT_STATUS=FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
