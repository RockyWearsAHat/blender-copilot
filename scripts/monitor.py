#!/usr/bin/env python3
"""Live monitoring panel — BlendSwap/SmutBase rip + training progress.

Usage:
    python scripts/monitor.py
"""

import os
import re
import sys
import time
import argparse
import subprocess
import platform
import shutil
from pathlib import Path
from collections import deque

ROOT = Path(__file__).resolve().parent.parent

REFRESH = 4  # seconds

LOGS = {
    "rip":       "/tmp/rip_all.log",       # BlendSwap (all categories)
    "smutbase":  "/tmp/rip_smutbase.log",  # SmutBase only
    "open3dlab": "/tmp/rip_open3dlab.log", # Open3DLab only
    "fetch":     "/tmp/fetch_data.log",    # Objaverse
    "train":     "/tmp/semantic_v3.log",
    "inference": "/tmp/inference.log",
}

_COUNT_CACHE: dict[str, tuple[float, int]] = {}
_LOGIN_CACHE: dict[str, tuple[float, str]] = {}
_SMOKE_CACHE: dict[str, tuple[float, dict]] = {}


def _resolve_train_log(explicit: str | None = None) -> str:
    if explicit:
        return explicit

    candidates = [
        ROOT / "logs" / "train_latest.log",
    ]
    candidates.extend(sorted((ROOT / "logs").glob("train_resume_*.log"), reverse=True))
    candidates.extend(sorted((ROOT / "logs").glob("train_*.log"), reverse=True))
    candidates.append(Path("/tmp/semantic_v3.log"))

    for c in candidates:
        if c.exists():
            return str(c)
    return str(ROOT / "logs" / "train_latest.log")

PIDS = {
    "rip":      "rip_blendswap_smutbase",
    "fetch":    "fetch_diverse_data",
    "train":    "run.py",
}


def is_alive(keyword: str) -> bool:
    r = subprocess.run(
        ["pgrep", "-f", keyword],
        capture_output=True, text=True
    )
    return bool(r.stdout.strip())


def _read_log_window(path: str, max_bytes: int = 256_000) -> str:
    p = Path(path)
    if not p.exists() or max_bytes <= 0:
        return ""

    try:
        size = p.stat().st_size
        with p.open("rb") as f:
            if size <= max_bytes:
                data = f.read()
            else:
                head = max_bytes // 2
                tail = max_bytes - head
                first = f.read(head)
                f.seek(max(size - tail, 0))
                last = f.read(tail)
                data = first + b"\n" + last
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def tail(path: str, n: int = 8, max_bytes: int = 160_000) -> list[str]:
    try:
        p = Path(path)
        if not p.exists():
            return []

        with p.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            read_size = min(max_bytes, size)
            f.seek(-read_size, os.SEEK_END)
            data = f.read().decode("utf-8", errors="replace")

        lines = data.splitlines()
        return lines[-n:]
    except Exception:
        return []


def last_match(
    path: str,
    pattern: str,
    search_lines: int = 1200,
    max_bytes: int | None = None,
) -> str:
    def _backscan_last_matching_line(
        file_path: str,
        rx: re.Pattern[str],
        *,
        chunk_size: int = 64_000,
        max_scan_bytes: int = 8_000_000,
    ) -> str:
        p = Path(file_path)
        if not p.exists():
            return ""

        try:
            size = p.stat().st_size
            if size <= 0:
                return ""

            scanned = 0
            pos = size
            carry = ""

            with p.open("rb") as f:
                while pos > 0 and scanned < max_scan_bytes:
                    read_len = min(chunk_size, pos)
                    pos -= read_len
                    f.seek(pos, os.SEEK_SET)
                    chunk = f.read(read_len)
                    scanned += read_len

                    text = chunk.decode("utf-8", errors="replace") + carry
                    # Keep the earliest partial line as carry; everything after
                    # the first newline is line-aligned.
                    first_nl = text.find("\n")
                    if first_nl == -1:
                        carry = text
                        continue

                    carry = text[:first_nl]
                    aligned = text[first_nl + 1 :]

                    for line in reversed(aligned.splitlines()):
                        if rx.search(line):
                            return line.strip()

            # Final attempt: whatever remains in carry may contain a match.
            if carry and rx.search(carry):
                return carry.strip()
        except Exception:
            return ""

        return ""

    try:
        rx = re.compile(pattern)

        # Important: tail() also limits by max_bytes. Some logs (especially
        # training during real-mesh preprocessing) can be extremely chatty
        # and push the last "Step ..." line outside a small byte window.
        if max_bytes is None:
            # ~400 bytes/line is conservative and keeps things responsive.
            max_bytes = max(160_000, int(search_lines) * 400)

        lines = tail(path, n=search_lines, max_bytes=max_bytes)
        for line in reversed(lines):
            if rx.search(line):
                return line.strip()

        # Fallback: scan backwards in chunks until we find a match (or we hit
        # a max scan budget). This avoids "Step: ?" when the log tail is
        # dominated by verbose preprocessing output.
        return _backscan_last_matching_line(path, rx)
    except Exception:
        return ""


def count_files(glob: str) -> int:
    try:
        now = time.time()
        cached = _COUNT_CACHE.get(glob)
        if cached and now - cached[0] < 20:
            return cached[1]

        count = sum(1 for _ in ROOT.glob(glob))
        _COUNT_CACHE[glob] = (now, count)
        return count
    except Exception:
        return 0


def login_status(log_path, login_keyword):
    p = Path(log_path)
    if not p.exists():
        return "\033[33m○ pending\033[0m"

    try:
        mtime = p.stat().st_mtime
    except Exception:
        mtime = 0.0

    cache_key = f"{log_path}:{login_keyword}"
    cached = _LOGIN_CACHE.get(cache_key)
    if cached and cached[0] == mtime:
        return cached[1]

    text = _read_log_window(log_path, max_bytes=320_000)
    logged_in = "Logged in" in text and login_keyword in text
    failed = "Login failed" in text and not logged_in
    if logged_in:
        status = "\033[1;32m✓ logged in (premium)\033[0m"
    elif failed:
        status = "\033[1;31m✗ login failed\033[0m"
    else:
        status = "\033[33m○ pending\033[0m"

    _LOGIN_CACHE[cache_key] = (mtime, status)
    return status


def bar(val: float, width: int = 20, max_val: float = 100.0) -> str:
    filled = int(width * min(val, max_val) / max_val)
    return "█" * filled + "░" * (width - filled)


def _bbox_max_extent(vertices: list[list[float]]) -> float:
    if not vertices:
        return 0.0
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    return float(max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)))


def _probe_cube_smoke(
    *,
    server_url: str,
    samples: int,
    cache_seconds: int,
    served_key: str | None = None,
    health: dict | None = None,
) -> dict:
    """Return cached cube-smoke results.

    This is intentionally lightweight and heuristic: it measures whether the
    model can consistently produce a non-degenerate, cube-ish mesh for the
    prompt "a cube".
    """

    now = time.time()
    # Cache semantics:
    # - If we know what checkpoint the server is hosting (via /health), cache
    #   by that identity so we only resample when the hosted model changes.
    # - Otherwise fall back to time-based caching.
    served_key_norm = (served_key or "").strip() or "unknown"
    cache_key = f"cube_smoke:{server_url}:{samples}:{served_key_norm}"
    cached = _SMOKE_CACHE.get(cache_key)
    if cached:
        if served_key_norm != "unknown":
            return cached[1]
        if (now - cached[0]) < max(1, int(cache_seconds)):
            return cached[1]

    result = {
        "ok": False,
        "server_up": False,
        "samples": int(samples),
        "success": 0,
        "cube_like": 0,
        "degenerate": 0,
        "faces_median": None,
        "faces_p90": None,
        "faces_max": None,
        "verts_median": None,
        "elapsed": 0.0,
        "error": None,
    }

    start = time.time()
    try:
        import urllib.request
        import json as _json

        # Quick health probe (unless caller already did it)
        if isinstance(health, dict) and health:
            result["server_up"] = True
        else:
            with urllib.request.urlopen(f"{server_url}/health", timeout=2) as _r:
                _ = _json.loads(_r.read())
            result["server_up"] = True

        faces_counts: list[int] = []
        verts_counts: list[int] = []

        payload_base = {
            "prompt": "a cube",
            # Slightly exploratory but bounded.
            "temperature": 0.85,
            "top_p": 0.95,
            "top_k": 32,
            "cfg_scale": 3.5,
            # Keep generation fast for dashboard usage.
            "max_faces": 64,
        }

        for _i in range(int(samples)):
            req = urllib.request.Request(
                f"{server_url}/generate/mesh",
                data=_json.dumps(payload_base).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=12) as _r:
                resp = _json.loads(_r.read())

            objs = resp.get("objects") or []
            if not objs:
                faces_counts.append(0)
                verts_counts.append(0)
                continue

            mesh = (objs[0] or {}).get("mesh") or {}
            vertices = mesh.get("vertices") or []
            faces = mesh.get("faces") or []

            v_n = int(len(vertices))
            f_n = int(len(faces))
            verts_counts.append(v_n)
            faces_counts.append(f_n)

            if v_n > 0 and f_n > 0:
                result["success"] += 1

            max_extent = _bbox_max_extent(vertices) if vertices else 0.0
            if max_extent < 1e-3:
                result["degenerate"] += 1

            # Cube-ish heuristic:
            # - Triangulated cube: ~8 verts, ~12 faces (tris)
            # - Quad cube:         ~8 verts, ~6 faces (quads)
            face_arity = None
            if faces:
                try:
                    arities = sorted(len(ff) for ff in faces if ff)
                    face_arity = arities[len(arities) // 2] if arities else None
                except Exception:
                    face_arity = None

            if max_extent > 0.1 and (6 <= v_n <= 14):
                if face_arity and face_arity >= 4:
                    cubeish = (5 <= f_n <= 14)
                else:
                    cubeish = (10 <= f_n <= 40)
                if cubeish:
                    result["cube_like"] += 1

        def _pct(a: int, b: int) -> float:
            return (100.0 * float(a) / float(max(1, b)))

        result["faces_median"] = int(sorted(faces_counts)[len(faces_counts) // 2])
        result["faces_p90"] = int(sorted(faces_counts)[max(0, int(0.9 * (len(faces_counts) - 1)))])
        result["faces_max"] = int(max(faces_counts) if faces_counts else 0)
        result["verts_median"] = int(sorted(verts_counts)[len(verts_counts) // 2])
        result["ok"] = True
        result["success_pct"] = _pct(result["success"], int(samples))
        result["cube_like_pct"] = _pct(result["cube_like"], int(samples))
        result["degenerate_pct"] = _pct(result["degenerate"], int(samples))
    except Exception as e:
        result["error"] = str(e)
    finally:
        result["elapsed"] = round(time.time() - start, 2)
        _SMOKE_CACHE[cache_key] = (now, result)

    return result

def clear():
    # If stdout isn't a TTY (e.g. redirected to logs/monitor_latest.log),
    # don't emit terminal control codes. This keeps logs readable.
    try:
        if not sys.stdout.isatty():
            return
    except Exception:
        return

    if os.name == 'nt':
        # Windows
        command = 'cls'
    else:
        # Unix/Linux/Mac
        command = "clear && printf '\\033[3J'"

    os.system(command)
    print("\033[H\033[J", end="")


def render(refresh_seconds: int, *, enable_smoke: bool, smoke_every: int, smoke_samples: int):
    now = time.strftime("%H:%M:%S")

    clear()
    # ── Header ────────────────────────────────────────────────────────
    print(f"\033[1;36m{'─'*62}\033[0m")
    print(f"\033[1;36m  BLENDER COPILOT — MONITOR          {now}\033[0m")
    print(f"\033[1;36m{'─'*62}\033[0m")

    # ── Training ──────────────────────────────────────────────────────
    # Prefer a robust signal over fancy regexes: BSD `pgrep` on macOS doesn't
    # reliably support POSIX character classes like `[[:space:]]`.
    # Consider training "alive" if the train log is being written recently,
    # and fall back to a simple pgrep pattern.
    alive_train = False
    try:
        p = Path(LOGS["train"])
        if p.exists():
            alive_train = (time.time() - p.stat().st_mtime) < max(20, 3 * REFRESH)
    except Exception:
        alive_train = False
    alive_train = alive_train or is_alive(r"run\.py.* train")
    status_train = "\033[1;32m● RUNNING\033[0m" if alive_train else "\033[1;31m○ STOPPED\033[0m"

    last_step = last_match(LOGS["train"], r"Step \d+")
    step_num, geom_loss, real_loss, lr, speed = "?", "?", "?", "?", "?"
    if last_step:
        m = re.search(r"Step (\d+)", last_step)
        if m: step_num = m.group(1)
        m = re.search(r"geom=([\d.]+)", last_step)
        if m: geom_loss = m.group(1)
        m = re.search(r"real=([\d.]+)", last_step)
        if m: real_loss = m.group(1)
        m = re.search(r"LR: ([\d.e+-]+)", last_step)
        if m: lr = m.group(1)
        m = re.search(r"([\d.]+) it/s", last_step)
        if m: speed = m.group(1)

    print(f"\n\033[1;33m▶ TRAINING\033[0m  {status_train}")
    print(f"  Step: \033[1m{step_num}\033[0m  |  geom loss: \033[1m{geom_loss}\033[0m  |  real loss: \033[1m{real_loss}\033[0m")
    print(f"  LR: {lr}  |  Speed: {speed} it/s")
    print(f"  Log: {LOGS['train']}")

    # Show the active run/output dir (written by run.py train)
    try:
        import json as _json
        at = ROOT / "logs" / "active_train.json"
        if at.exists():
            data = _json.loads(at.read_text(encoding="utf-8"))
            rn = data.get("run_name") or "?"
            od = data.get("output_dir") or "?"
            cfg = data.get("config") or "?"
            print(f"  Run: {rn}  |  Output: {od}")
            print(f"  Config: {cfg}  |  Active: checkpoints/_active/train_latest.pt")
    except Exception:
        pass

    # ── BlendSwap/SmutBase Rip ────────────────────────────────────────
    alive_rip = is_alive("rip_blendswap_smutbase")
    status_rip = "\033[1;32m● RUNNING\033[0m" if alive_rip else "\033[1;31m○ STOPPED\033[0m"

    def last_download(log_path):
        for l in reversed(tail(log_path, 140)):
            if any(x in l for x in ["Downloaded", "Extracted", "WARN", "ERROR", "Page", "category"]):
                return l.strip()[-72:]
        return ""

    bs_blends  = count_files("data/raw/blendswap/**/*.blend")
    sm_blends  = count_files("data/raw/smutbase/**/*.blend")
    o3d_blends = count_files("data/raw/open3dlab/**/*.blend")

    bs_pages  = last_match(LOGS["rip"],       r"Page \d+/\d+")
    sm_pages  = last_match(LOGS["smutbase"],  r"Page \d+/\d+")
    o3d_pages = last_match(LOGS["open3dlab"], r"Page \d+/\d+")

    def page_str(match_line):
        m = re.search(r"Page (\d+)/(\d+)", match_line)
        return f"p{m.group(1)}/{m.group(2)}" if m else "p?"

    n_scrapers = sum(1 for _ in __import__('subprocess').run(['pgrep','-f','rip_blendswap'],capture_output=True,text=True).stdout.strip().splitlines())
    print(f"\n\033[1;33m▶ DATA RIPS\033[0m  ({n_scrapers} scraper processes running)")
    print(f"  BlendSwap   {login_status(LOGS['rip'],       'BlendSwap')}  {page_str(bs_pages)}   {bs_blends} files")
    print(f"  SmutBase    {login_status(LOGS['smutbase'],  'smutba.se')}  {page_str(sm_pages)}   {sm_blends} files")
    print(f"  Open3DLab   {login_status(LOGS['open3dlab'], 'open3dlab.com')}  {page_str(o3d_pages)}  {o3d_blends} files")

    for src, log in [("BS", LOGS["rip"]), ("SM", LOGS["smutbase"]), ("O3D", LOGS["open3dlab"])]:
        last = last_download(log)
        if last:
            print(f"  \033[2m[{src}] {last}\033[0m")

    # Inference server — hit /health directly (log-based check misses after hot-reload)
    health_url = "http://localhost:8420/health"
    try:
        import json as _json
        asv = ROOT / "logs" / "active_serve.json"
        if asv.exists():
            sdata = _json.loads(asv.read_text(encoding="utf-8"))
            if sdata.get("health_url"):
                health_url = str(sdata.get("health_url"))
    except Exception:
        pass

    try:
        import urllib.request, json as _json
        with urllib.request.urlopen(health_url, timeout=2) as _r:
            _h = _json.loads(_r.read())
        _step = _h.get("step", "?")
        _ckpt = _h.get("checkpoint") or "?"
        _served_key = f"{_ckpt}|{_step}"
        try:
            _port = health_url.split(":")[2].split("/")[0]
            infer_status = f"\033[1;32m● UP :{_port}  step={_step}\033[0m"
        except Exception:
            infer_status = f"\033[1;32m● UP  step={_step}\033[0m"
    except Exception:
        infer_alive = is_alive(r"run\.py.*[[:space:]]serve([[:space:]]|$)")
        infer_status = ("\033[33m⏳ starting\033[0m" if infer_alive
                        else "\033[1;31m○ DOWN\033[0m")
        _ckpt = "?"
        _served_key = None
        _h = None

    # If available, also show the stable served-checkpoint pointer.
    served_ptr = "checkpoints/_active/served_checkpoint.pt"
    try:
        import json as _json
        asv = ROOT / "logs" / "active_serve.json"
        if asv.exists():
            sdata = _json.loads(asv.read_text(encoding="utf-8"))
            # Prefer the explicit checkpoint path written by run.py serve.
            ck = sdata.get("checkpoint") or _ckpt
            _ckpt = ck
    except Exception:
        pass

    print(f"\n\033[1;33m▶ INFERENCE SERVER\033[0m  {infer_status}  (checkpoint: {_ckpt})")
    print(f"  Active: {served_ptr}")

    if enable_smoke:
        smoke = _probe_cube_smoke(
            server_url="http://localhost:8420",
            samples=int(smoke_samples),
            cache_seconds=int(smoke_every),
            served_key=_served_key,
            health=_h if isinstance(_h, dict) else None,
        )
        if not smoke.get("server_up"):
            print("\033[2m  smoke: server down\033[0m")
        elif not smoke.get("ok"):
            err = smoke.get("error") or "unknown error"
            print(f"\033[2m  smoke: error ({err[:70]})\033[0m")
        else:
            print(
                "\033[2m"
                f"  smoke(a cube): cube_like={smoke['cube_like']}/{smoke['samples']} ({smoke['cube_like_pct']:.1f}%) "
                f"| success={smoke['success']}/{smoke['samples']} ({smoke['success_pct']:.1f}%) "
                f"| degenerate={smoke['degenerate']}/{smoke['samples']} ({smoke['degenerate_pct']:.1f}%) "
                f"| faces med/p90/max={smoke['faces_median']}/{smoke['faces_p90']}/{smoke['faces_max']} "
                f"| verts med={smoke['verts_median']} "
                f"| {smoke['elapsed']}s"
                "\033[0m"
            )

    # ── Objaverse Fetch ───────────────────────────────────────────────
    alive_fetch = is_alive("fetch_diverse_data")
    status_fetch = "\033[1;32m● RUNNING\033[0m" if alive_fetch else "\033[1;90m○ done/stopped\033[0m"
    last_fetch = last_match(LOGS["fetch"], r"extracted|ERROR|complete|Progress")
    objaverse_extracted = 0
    if last_fetch:
        # "→ N extracted"  or  "Progress: N downloaded, N extracted"
        m = re.search(r"→ (\d+) extracted", last_fetch)
        if m:
            objaverse_extracted = int(m.group(1))
        else:
            m = re.search(r"(\d+) extracted", last_fetch)
            if m: objaverse_extracted = int(m.group(1))

    print(f"\n\033[1;33m▶ OBJAVERSE FETCH\033[0m  {status_fetch}")
    print(f"  Models extracted: \033[1m{objaverse_extracted}\033[0m / ~720 target")
    if last_fetch:
        print(f"  \033[2m{last_fetch[-80:]}\033[0m")

    # ── Processed JSONs ───────────────────────────────────────────────
    json_bs  = count_files("data/processed/blendswap/*.json")
    json_sm  = count_files("data/processed/smutbase/*.json")
    json_o3d = count_files("data/processed/open3dlab/*.json")
    json_obj = count_files("data/processed/objaverse/*.json")
    json_tot = json_bs + json_sm + json_o3d + json_obj

    print(f"\n\033[1;33m▶ PROCESSED JSONs\033[0m  (training-ready)")
    print(f"  blendswap: {json_bs}  |  smutbase: {json_sm}  |  open3dlab: {json_o3d}  |  objaverse: {json_obj}")
    print(f"  Total: \033[1m{json_tot}\033[0m  {bar(json_tot, max_val=2000)}")

    # ── Footer ────────────────────────────────────────────────────────
    print(f"\n\033[2m  Refreshing every {refresh_seconds}s — Ctrl+C to exit\033[0m")
    print(f"\033[1;36m{'─'*62}\033[0m")


def main():
    parser = argparse.ArgumentParser(description="Blender Copilot live monitor")
    parser.add_argument("--refresh", type=int, default=REFRESH, help="Refresh interval in seconds")
    parser.add_argument("--train-log", type=str, default=None, help="Explicit path to train log")
    parser.add_argument("--once", action="store_true", help="Render one snapshot and exit")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Enable a small prompt-based smoke test against the inference server",
    )
    parser.add_argument(
        "--smoke-every",
        type=int,
        default=300,
        help="How often to refresh smoke-test results (seconds, cached)",
    )
    parser.add_argument(
        "--smoke-samples",
        type=int,
        default=2,
        help="Number of /generate/mesh samples per smoke test",
    )
    parser.add_argument(
        "--keep-awake",
        action="store_true",
        default=(platform.system() == "Darwin"),
        help="On macOS, use caffeinate to prevent sleep/display sleep while monitor runs",
    )
    parser.add_argument(
        "--no-keep-awake",
        dest="keep_awake",
        action="store_false",
        help="Disable keep-awake behavior",
    )
    args = parser.parse_args()

    LOGS["train"] = _resolve_train_log(args.train_log)

    keepawake_proc = None
    win_keepawake_enabled = False
    if args.keep_awake:
        system = platform.system()
        if system == "Darwin":
            caffeinate_bin = shutil.which("caffeinate")
            if caffeinate_bin:
                try:
                    # -d: prevent display sleep, -i: prevent idle sleep, -m: prevent disk idle
                    keepawake_proc = subprocess.Popen([caffeinate_bin, "-dim"])
                except Exception:
                    keepawake_proc = None
        elif system == "Linux":
            inhibit_bin = shutil.which("systemd-inhibit")
            if inhibit_bin:
                try:
                    keepawake_proc = subprocess.Popen(
                        [
                            inhibit_bin,
                            "--why=Blender Copilot monitor",
                            "--what=sleep:idle",
                            "bash",
                            "-lc",
                            "while true; do sleep 3600; done",
                        ]
                    )
                except Exception:
                    keepawake_proc = None
        elif system == "Windows":
            try:
                import ctypes
                ES_CONTINUOUS = 0x80000000
                ES_SYSTEM_REQUIRED = 0x00000001
                ES_DISPLAY_REQUIRED = 0x00000002
                res = ctypes.windll.kernel32.SetThreadExecutionState(
                    ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
                )
                win_keepawake_enabled = bool(res)
            except Exception:
                win_keepawake_enabled = False

    print("\033[?25l", end="")  # hide cursor
    try:
        os.system('cls' if os.name == 'nt' else 'clear')
        if args.once:
            render(
                max(1, int(args.refresh)),
                enable_smoke=bool(args.smoke),
                smoke_every=int(args.smoke_every),
                smoke_samples=int(args.smoke_samples),
            )
            print("\033[?25h\033[0m", end="")
            return

        while True:
            render(
                max(1, int(args.refresh)),
                enable_smoke=bool(args.smoke),
                smoke_every=int(args.smoke_every),
                smoke_samples=int(args.smoke_samples),
            )
            time.sleep(max(1, int(args.refresh)))
    except KeyboardInterrupt:
        print("\033[?25h\033[0m")  # restore cursor + colors
        print("\nMonitor stopped.")
        sys.exit(0)
    finally:
        if keepawake_proc is not None:
            try:
                keepawake_proc.terminate()
            except Exception:
                pass
        if win_keepawake_enabled:
            try:
                import ctypes
                ES_CONTINUOUS = 0x80000000
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            except Exception:
                pass


if __name__ == "__main__":
    main()
