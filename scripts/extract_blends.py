#!/usr/bin/env python3 -u
"""Wrapper to extract .blend files one at a time in isolated Blender processes.

Each .blend file gets its own Blender subprocess, so a crash on one file
doesn't kill the entire batch. This is essential for complex character
models that can segfault Blender.

Usage:
    python scripts/extract_blends.py --input data/raw/smutbase/files --output data/processed/smutbase
    python scripts/extract_blends.py --input data/raw/blendswap --output data/processed/blendswap
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
EXTRACTOR = Path(__file__).resolve().parent.parent / "processing" / "blend_extractor.py"


def extract_single_file(
    blend_path: Path,
    output_dir: Path,
    timeout: int = 420,
    retry_timeout: int | None = None,
    retry_immediate: bool = False,
    mark_timeout_invalid: bool = False,
) -> str:
    """Run Blender headless to extract a single .blend file.

    Returns: "ok", "skip", "invalid", "fail", "timeout"
    """
    output_file = output_dir / f"{blend_path.stem}.json"
    invalid_file = output_dir / f"{blend_path.stem}.invalid"
    timeout_file = output_dir / f"{blend_path.stem}.timeout"

    if invalid_file.exists():
        print(f"  SKIP (marked invalid): {blend_path.name}")
        return "skip"

    if timeout_file.exists():
        print(f"  RETRY (previous timeout): {blend_path.name}")

    if output_file.exists():
        print(f"  SKIP (already exists): {output_file.name}")
        return "skip"

    cmd = [
        BLENDER,
        "--background",
        "--python", str(EXTRACTOR),
        "--",
        "--input", str(blend_path),
        "--output", str(output_dir),
    ]

    try:
        effective_retry_timeout = (
            retry_timeout
            if retry_immediate and retry_timeout and retry_timeout > timeout
            else None
        )
        attempt_timeouts = [timeout]
        if effective_retry_timeout is not None:
            attempt_timeouts.append(effective_retry_timeout)

        for attempt_idx, current_timeout in enumerate(attempt_timeouts, start=1):
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=current_timeout,
                )
            except subprocess.TimeoutExpired:
                if attempt_idx < len(attempt_timeouts):
                    next_timeout = attempt_timeouts[attempt_idx]
                    print(
                        f"  TIMEOUT ({current_timeout}s) — retrying once with {next_timeout}s"
                    )
                    continue

                prev_attempts = 0
                if timeout_file.exists():
                    try:
                        first = timeout_file.read_text(errors="replace").splitlines()[0]
                        if first.startswith("attempts: "):
                            prev_attempts = int(first.split(":", 1)[1].strip())
                    except Exception:
                        prev_attempts = 0

                attempts = prev_attempts + 1
                reason = f"timeout after {current_timeout}s"
                timeout_file.write_text(
                    f"attempts: {attempts}\nreason: {reason}\nfile: {blend_path.name}\n"
                )

                if mark_timeout_invalid:
                    invalid_file.write_text(f"reason: {reason}\nfile: {blend_path.name}\n")
                    print(
                        f"  TIMEOUT ({current_timeout}s) — marked invalid (--mark-timeout-invalid)"
                    )
                    return "invalid"

                print(
                    f"  TIMEOUT ({current_timeout}s) — recorded timeout, will retry in future runs"
                )
                return "timeout"

            if output_file.exists():
                if timeout_file.exists():
                    timeout_file.unlink(missing_ok=True)
                size_mb = output_file.stat().st_size / 1e6
                print(f"  OK: {output_file.name} ({size_mb:.1f}MB)")
                return "ok"
            else:
                stderr_tail = result.stderr.strip().split("\n")[-5:] if result.stderr else []
                error_lines = [l.strip()[:150] for l in stderr_tail
                               if "Error" in l or "crash" in l.lower() or "Traceback" in l]

                is_corrupt = any(
                    x in result.stderr
                    for x in ["File format is not supported",
                               "incomplete header",
                               "invalid 'from' pointer",
                               "Loading \"/",
                               "read blend: premature"]
                ) if result.stderr else False

                is_segfault = result.returncode in (-11, 139, -6, 134)

                if is_corrupt or is_segfault:
                    if is_segfault:
                        reason = f"segfault/abort (exit code {result.returncode})"
                    else:
                        reason = error_lines[0] if error_lines else "corrupt or incompatible file"
                    invalid_file.write_text(f"reason: {reason}\nfile: {blend_path.name}\n")
                    print(f"  INVALID (marked, will not retry): {reason[:100]}")
                    return "invalid"
                else:
                    print(f"  FAIL: no output generated")
                    for line in error_lines:
                        print(f"    {line}")
                    return "fail"

    except Exception as e:
        print(f"  ERROR: {e}")
        return "fail"


def main():
    parser = argparse.ArgumentParser(
        description="Extract .blend files with crash isolation"
    )
    parser.add_argument("--input", required=True,
                        help="Input directory with .blend files")
    parser.add_argument("--output", required=True,
                        help="Output directory for JSON files")
    parser.add_argument("--timeout", type=int, default=420,
                        help="Primary timeout per file in seconds (default: 420)")
    parser.add_argument("--retry-timeout", type=int, default=None,
                        help="One-time retry timeout per file after primary timeout")
    parser.add_argument("--timeout-grace", type=int, default=180,
                        help="Extra seconds added for one retry if --retry-timeout is not set (default: 180)")
    parser.add_argument("--mark-timeout-invalid", action="store_true",
                        help="Mark timed-out files as .invalid (default: disabled; timeout stays retriable)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of parallel Blender extraction workers (default: 2)")
    parser.add_argument("--retry-immediate", action="store_true",
                        help="Immediately retry timed out files with --retry-timeout within same attempt")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max files to process")
    args = parser.parse_args()

    SUPPORTED_EXTS = {'.blend', '.glb', '.gltf', '.obj', '.fbx'}

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_dir.is_file():
        blend_files = [input_dir]
    else:
        blend_files = sorted(
            f for f in input_dir.rglob("*")
            if f.suffix.lower() in SUPPORTED_EXTS
            and not f.name.startswith("._")
        )

    if args.limit:
        blend_files = blend_files[:args.limit]

    print(f"Found {len(blend_files)} 3D files")
    print(f"Output: {output_dir}")
    print()

    counts = {"ok": 0, "skip": 0, "invalid": 0, "fail": 0, "timeout": 0}
    retry_timeout = args.retry_timeout if args.retry_timeout else (args.timeout + max(args.timeout_grace, 0))

    workers = max(1, int(args.workers or 1))

    def _run_one(idx: int, total: int, bf: Path, timeout_value: int) -> tuple[str, float, str]:
        print(f"[{idx}/{total}] {bf.name} ({bf.stat().st_size / 1e6:.0f}MB)")
        start = time.time()
        status = extract_single_file(
            bf,
            output_dir,
            timeout=timeout_value,
            retry_timeout=retry_timeout,
            retry_immediate=args.retry_immediate,
            mark_timeout_invalid=args.mark_timeout_invalid,
        )
        elapsed = time.time() - start
        return status, elapsed, bf.name

    timeout_queue: list[Path] = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_run_one, i + 1, len(blend_files), bf, args.timeout): bf
            for i, bf in enumerate(blend_files)
        }
        for fut in as_completed(futures):
            status, elapsed, _ = fut.result()
            counts[status] = counts.get(status, 0) + 1
            if status not in ("skip",):
                print(f"  Time: {elapsed:.0f}s")
            if status == "timeout" and not args.retry_immediate:
                timeout_queue.append(futures[fut])

    if timeout_queue and retry_timeout > args.timeout:
        print(f"\nSecond pass for {len(timeout_queue)} timed out files with timeout={retry_timeout}s")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(_run_one, i + 1, len(timeout_queue), bf, retry_timeout): bf
                for i, bf in enumerate(timeout_queue)
            }
            for fut in as_completed(futures):
                status, elapsed, _ = fut.result()
                counts[status] = counts.get(status, 0) + 1
                if status not in ("skip",):
                    print(f"  Time: {elapsed:.0f}s")

    print(f"\nDone! {counts['ok']} extracted, {counts['fail']} failed, "
          f"{counts['invalid']} invalid, {counts['timeout']} timed out, {counts['skip']} skipped "
          f"(total: {len(blend_files)} files)")


if __name__ == "__main__":
    main()
