#!/usr/bin/env python3
"""BlenderModelTraining — Interactive CLI Controller.

Launch with:  python cli.py

A single interactive menu for managing the entire ML pipeline:
scraping, extraction, dataset building, training, serving, and testing.
"""

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DATASET_DIR = DATA_DIR / "datasets"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CONFIG_PATH = PROJECT_ROOT / "config_test.yaml"
FULL_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
VENV_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python3"

# ── Colors ────────────────────────────────────────────────────────────
class C:
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"
    END     = "\033[0m"
    BG_BLUE = "\033[44m"

    @staticmethod
    def ok(msg):   return f"{C.GREEN}✓{C.END} {msg}"
    @staticmethod
    def warn(msg): return f"{C.YELLOW}⚠{C.END} {msg}"
    @staticmethod
    def err(msg):  return f"{C.RED}✗{C.END} {msg}"
    @staticmethod
    def info(msg): return f"{C.BLUE}ℹ{C.END} {msg}"


def clear():
    os.system("clear" if os.name != "nt" else "cls")


def count_files(directory: Path, pattern: str = "*") -> int:
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def dir_size_mb(directory: Path) -> float:
    if not directory.exists():
        return 0
    total = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"cuda ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps (Apple Silicon)"
        return "cpu"
    except ImportError:
        return "cpu (torch not installed)"


def load_config(path: Path | None = None) -> dict:
    import yaml
    p = path or CONFIG_PATH
    if not p.exists():
        p = FULL_CONFIG_PATH
    with open(p) as f:
        return yaml.safe_load(f)


def check_server(port: int = 8420) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:
        return False


# ── Background process tracker ────────────────────────────────────────
_bg_processes: dict[str, subprocess.Popen] = {}


def _start_bg(name: str, cmd: list[str], logfile: str | None = None):
    """Start a background subprocess, track it by name."""
    if name in _bg_processes and _bg_processes[name].poll() is None:
        print(C.warn(f"{name} already running (PID {_bg_processes[name].pid})"))
        return
    log_fh = open(logfile, "a") if logfile else subprocess.DEVNULL
    proc = subprocess.Popen(
        cmd, stdout=log_fh, stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
    )
    _bg_processes[name] = proc
    print(C.ok(f"{name} started (PID {proc.pid})"
               + (f"  log → {logfile}" if logfile else "")))


def _stop_bg(name: str):
    proc = _bg_processes.get(name)
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print(C.ok(f"{name} stopped"))
    else:
        print(C.warn(f"{name} not running"))
    _bg_processes.pop(name, None)


def _bg_status() -> list[str]:
    alive = []
    for name, proc in list(_bg_processes.items()):
        if proc.poll() is None:
            alive.append(f"{name} (PID {proc.pid})")
        else:
            del _bg_processes[name]
    return alive


# ══════════════════════════════════════════════════════════════════════
# Menu Screens
# ══════════════════════════════════════════════════════════════════════

def _header():
    """Print the top banner."""
    bg = _bg_status()
    server_up = check_server()

    print(f"\n{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════╗{C.END}")
    print(f"{C.BOLD}{C.CYAN}║   🧊  Blender Model Training — Control Panel    ║{C.END}")
    print(f"{C.BOLD}{C.CYAN}╚══════════════════════════════════════════════════╝{C.END}")
    status_parts = []
    status_parts.append(f"{C.GREEN}● Server UP{C.END}" if server_up else f"{C.DIM}○ Server down{C.END}")
    if bg:
        status_parts.append(f"{C.YELLOW}⟳ {len(bg)} bg task(s){C.END}")
    print(f"  {' │ '.join(status_parts)}    {C.DIM}Device: {detect_device()}{C.END}")
    print()


def main_menu():
    """Top-level interactive menu."""
    clear()
    _header()

    options = [
        ("1", "📊  Dashboard",           "View current progress & stats"),
        ("2", "📥  Data Collection",      "Scrape / download training data"),
        ("3", "⚙️   Process & Build",      "Extract meshes → build dataset"),
        ("4", "🏋️   Train Model",          "Start or resume training"),
        ("5", "🚀  Serve Model",          "Start / stop inference server"),
        ("6", "🧪  Test Generation",      "Send a test prompt to the server"),
        ("7", "📦  Package Plugin",       "Build Blender addon ZIP"),
        ("8", "🔧  Settings",             "Config, paths, device info"),
        ("b", "📋  Background Tasks",     "View / stop running tasks"),
        ("q", "👋  Quit",                 ""),
    ]

    for key, label, desc in options:
        print(f"  {C.BOLD}{C.CYAN}[{key}]{C.END}  {label:.<34s} {C.DIM}{desc}{C.END}")
    print()

    return input(f"  {C.BOLD}▶ {C.END}").strip().lower()


# ── 1. Dashboard ──────────────────────────────────────────────────────
def screen_dashboard():
    clear()
    _header()
    print(f"  {C.BOLD}📊 Dashboard{C.END}")
    print(f"  {'─' * 46}\n")

    # Raw data
    print(f"  {C.BOLD}📁 Raw Data{C.END}")
    sources = [
        ("Objaverse GLB",        RAW_DIR / "objaverse" / "models",          "*.glb"),
        ("Blender Official",     RAW_DIR / "blender_official" / "models",   "*.blend"),
        ("YouTube Transcripts",  RAW_DIR / "youtube",                       "*.json"),
    ]
    for name, path, pat in sources:
        n = count_files(path, pat)
        sz = dir_size_mb(path)
        s = C.ok(f"{n:,} files ({sz:.0f} MB)") if n else C.warn("none")
        print(f"     {name:.<32s} {s}")
    print()

    # Processed
    print(f"  {C.BOLD}⚙️  Processed Meshes{C.END}")
    for name, path in [("Objaverse", PROCESSED_DIR / "objaverse"),
                       ("Blender Official", PROCESSED_DIR / "blender_official")]:
        n = count_files(path, "*.json")
        s = C.ok(f"{n:,} objects") if n else C.warn("not extracted")
        print(f"     {name:.<32s} {s}")
    print()

    # Datasets
    print(f"  {C.BOLD}📊 Datasets{C.END}")
    geo_dir = DATASET_DIR / "geometry"
    for split in ["train", "val", "test"]:
        p = geo_dir / f"{split}.jsonl"
        if p.exists():
            n = sum(1 for _ in open(p))
            print(f"     {split:.<32s} {C.ok(f'{n:,} examples')}")
        else:
            print(f"     {split:.<32s} {C.warn('not built')}")
    print()

    # Checkpoints
    print(f"  {C.BOLD}🏋️  Checkpoints{C.END}")
    if CHECKPOINT_DIR.exists():
        for cp in sorted(CHECKPOINT_DIR.iterdir()):
            if cp.is_dir():
                best = cp / "best.pt"
                if best.exists():
                    sz = best.stat().st_size / (1024 ** 2)
                    steps = len(list(cp.glob("step_*.pt")))
                    print(f"     {cp.name:.<32s} {C.ok(f'best.pt ({sz:.0f}MB), {steps} step checkpoints')}")
                else:
                    print(f"     {cp.name:.<32s} {C.warn('no best.pt')}")
    else:
        print(f"     {C.warn('No checkpoints yet')}")
    print()

    # Config summary
    if CONFIG_PATH.exists():
        cfg = load_config()
        geo = cfg.get("models", {}).get("geometry", {})
        tr = cfg.get("training", {})
        print(f"  {C.BOLD}⚙️  Active Config{C.END}  ({CONFIG_PATH.name})")
        print(f"     Model: {geo.get('hidden_size', '?')}d, "
              f"{geo.get('num_layers', '?')} layers, "
              f"{geo.get('num_heads', '?')} heads")
        print(f"     Training: {tr.get('max_steps', '?')} steps, "
              f"batch={tr.get('batch_size', '?')}, "
              f"lr={tr.get('learning_rate', '?')}")

    print()
    input(f"  {C.DIM}Press Enter to go back...{C.END}")


# ── 2. Data Collection ───────────────────────────────────────────────
def screen_data():
    clear()
    _header()
    print(f"  {C.BOLD}📥 Data Collection{C.END}")
    print(f"  {'─' * 46}\n")

    # Current counts
    n_obj  = count_files(RAW_DIR / "objaverse" / "models", "*.glb")
    n_blen = count_files(RAW_DIR / "blender_official" / "models", "*.blend")
    n_yt   = count_files(RAW_DIR / "youtube", "*.json")
    print(f"  Current: {n_obj} Objaverse │ {n_blen} Blender │ {n_yt} YouTube")
    print()

    options = [
        ("1", "Download Objaverse models (GLB)"),
        ("2", "Download Blender official files"),
        ("3", "Scrape YouTube tutorials"),
        ("4", "Download ALL sources"),
        ("5", "Check scraper log (tail)"),
        ("0", "← Back"),
    ]
    for k, v in options:
        print(f"  {C.BOLD}{C.CYAN}[{k}]{C.END}  {v}")
    print()

    choice = input(f"  {C.BOLD}▶ {C.END}").strip()
    python = str(VENV_PYTHON)

    if choice == "1":
        max_m = input(f"  Max models [{C.DIM}1000{C.END}]: ").strip() or "1000"
        _start_bg("objaverse-scrape",
                   [python, "-m", "scrapers.objaverse_scraper",
                    "--output", "data/raw/objaverse",
                    "--max-models", max_m],
                   "objaverse_scrape.log")
        input(f"\n  {C.DIM}Press Enter...{C.END}")

    elif choice == "2":
        _start_bg("blender-scrape",
                   [python, "-c",
                    "from scrapers.blender_official import download_blender_official; "
                    "from pathlib import Path; "
                    "download_blender_official(Path('data/raw/blender_official'), max_size_mb=500)"],
                   "blender_scrape.log")
        input(f"\n  {C.DIM}Press Enter...{C.END}")

    elif choice == "3":
        _start_bg("youtube-scrape",
                   [python, "-m", "scrapers.youtube_scraper",
                    "--output", "data/raw/youtube"],
                   "youtube_scrape.log")
        input(f"\n  {C.DIM}Press Enter...{C.END}")

    elif choice == "4":
        max_m = input(f"  Max Objaverse models [{C.DIM}1000{C.END}]: ").strip() or "1000"
        # Start all three
        _start_bg("objaverse-scrape",
                   [python, "-m", "scrapers.objaverse_scraper",
                    "--output", "data/raw/objaverse", "--max-models", max_m],
                   "objaverse_scrape.log")
        _start_bg("blender-scrape",
                   [python, "-c",
                    "from scrapers.blender_official import download_blender_official; "
                    "from pathlib import Path; "
                    "download_blender_official(Path('data/raw/blender_official'), max_size_mb=500)"],
                   "blender_scrape.log")
        _start_bg("youtube-scrape",
                   [python, "-m", "scrapers.youtube_scraper",
                    "--output", "data/raw/youtube"],
                   "youtube_scrape.log")
        input(f"\n  {C.DIM}Press Enter...{C.END}")

    elif choice == "5":
        print(f"\n  {C.BOLD}Recent scraper output:{C.END}\n")
        for log in ["objaverse_scrape.log", "blender_scrape.log", "youtube_scrape.log"]:
            p = PROJECT_ROOT / log
            if p.exists():
                print(f"  {C.CYAN}── {log} ──{C.END}")
                lines = p.read_text().strip().split("\n")[-8:]
                for l in lines:
                    print(f"  {C.DIM}{l}{C.END}")
                print()
        input(f"  {C.DIM}Press Enter...{C.END}")


# ── 3. Process & Build ────────────────────────────────────────────────
def screen_process():
    clear()
    _header()
    print(f"  {C.BOLD}⚙️  Process & Build Dataset{C.END}")
    print(f"  {'─' * 46}\n")

    n_raw_obj  = count_files(RAW_DIR / "objaverse" / "models", "*.glb")
    n_raw_blen = count_files(RAW_DIR / "blender_official" / "models", "*.blend")
    n_proc     = count_files(PROCESSED_DIR / "objaverse", "*.json") + \
                 count_files(PROCESSED_DIR / "blender_official", "*.json")
    ds_train   = DATASET_DIR / "geometry" / "train.jsonl"
    n_dataset  = sum(1 for _ in open(ds_train)) if ds_train.exists() else 0

    print(f"  Raw:       {n_raw_obj + n_raw_blen:,} files")
    print(f"  Processed: {n_proc:,} objects")
    print(f"  Dataset:   {n_dataset:,} training examples")
    print()

    options = [
        ("1", "Extract meshes from raw files"),
        ("2", "Build training dataset from processed data"),
        ("3", "Extract + Build (both steps)"),
        ("0", "← Back"),
    ]
    for k, v in options:
        print(f"  {C.BOLD}{C.CYAN}[{k}]{C.END}  {v}")
    print()

    choice = input(f"  {C.BOLD}▶ {C.END}").strip()
    python = str(VENV_PYTHON)

    if choice in ("1", "3"):
        print(f"\n  {C.info('Extracting meshes...')}")
        r = subprocess.run([python, "run.py", "extract"],
                           cwd=str(PROJECT_ROOT), capture_output=False)
        if r.returncode != 0:
            print(C.err("Extraction failed"))

    if choice in ("2", "3"):
        print(f"\n  {C.info('Building dataset...')}")
        r = subprocess.run([python, "run.py", "build"],
                           cwd=str(PROJECT_ROOT), capture_output=False)
        if r.returncode != 0:
            print(C.err("Dataset build failed"))

    if choice in ("1", "2", "3"):
        input(f"\n  {C.DIM}Press Enter...{C.END}")


# ── 4. Train ──────────────────────────────────────────────────────────
def screen_train():
    clear()
    _header()
    print(f"  {C.BOLD}🏋️  Training{C.END}")
    print(f"  {'─' * 46}\n")

    # Show existing checkpoints
    if CHECKPOINT_DIR.exists():
        for cp in sorted(CHECKPOINT_DIR.iterdir()):
            if cp.is_dir() and (cp / "best.pt").exists():
                sz = (cp / "best.pt").stat().st_size / (1024**2)
                print(f"  {C.ok(f'{cp.name}: best.pt ({sz:.0f}MB)')}")
    print()

    options = [
        ("1", "Train new model (test config — fast)"),
        ("2", "Train new model (full config — slow)"),
        ("3", "Resume from checkpoint"),
        ("4", "Custom training (set steps, batch, name)"),
        ("0", "← Back"),
    ]
    for k, v in options:
        print(f"  {C.BOLD}{C.CYAN}[{k}]{C.END}  {v}")
    print()

    choice = input(f"  {C.BOLD}▶ {C.END}").strip()
    python = str(VENV_PYTHON)

    if choice == "1":
        name = input(f"  Run name [{C.DIM}auto{C.END}]: ").strip() or None
        cmd = [python, "run.py", "train", "--config", str(CONFIG_PATH)]
        if name:
            cmd += ["--name", name]
        print(f"\n  {C.info('Starting training in foreground...')}\n")
        subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        input(f"\n  {C.DIM}Press Enter...{C.END}")

    elif choice == "2":
        name = input(f"  Run name [{C.DIM}auto{C.END}]: ").strip() or None
        cmd = [python, "run.py", "train", "--config", str(FULL_CONFIG_PATH)]
        if name:
            cmd += ["--name", name]
        bg = input(f"  Run in background? [{C.DIM}y/N{C.END}]: ").strip().lower()
        if bg == "y":
            _start_bg("training", cmd, "training.log")
        else:
            print(f"\n  {C.info('Starting training...')}\n")
            subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        input(f"\n  {C.DIM}Press Enter...{C.END}")

    elif choice == "3":
        # List checkpoints
        cps = []
        if CHECKPOINT_DIR.exists():
            for cp in sorted(CHECKPOINT_DIR.iterdir()):
                best = cp / "best.pt"
                if best.exists():
                    cps.append(best)
                    print(f"    [{len(cps)}] {best}")
        if not cps:
            print(C.warn("No checkpoints found"))
            input(f"\n  {C.DIM}Press Enter...{C.END}")
            return
        idx = input(f"\n  Which checkpoint? [1-{len(cps)}]: ").strip()
        try:
            cp_path = cps[int(idx) - 1]
        except (ValueError, IndexError):
            print(C.err("Invalid selection"))
            input(f"\n  {C.DIM}Press Enter...{C.END}")
            return
        cmd = [python, "run.py", "train", "--resume", str(cp_path)]
        subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        input(f"\n  {C.DIM}Press Enter...{C.END}")

    elif choice == "4":
        steps = input(f"  Max steps [{C.DIM}2000{C.END}]: ").strip() or "2000"
        batch = input(f"  Batch size [{C.DIM}2{C.END}]: ").strip() or "2"
        name  = input(f"  Run name [{C.DIM}auto{C.END}]: ").strip() or None
        config = input(f"  Config [{C.DIM}test{C.END}] (test/full): ").strip() or "test"
        cfg = str(CONFIG_PATH) if config == "test" else str(FULL_CONFIG_PATH)
        cmd = [python, "run.py", "train", "--config", cfg,
               "--steps", steps, "--batch-size", batch]
        if name:
            cmd += ["--name", name]
        bg = input(f"  Run in background? [{C.DIM}y/N{C.END}]: ").strip().lower()
        if bg == "y":
            logname = f"training_{name or 'custom'}.log"
            _start_bg("training", cmd, logname)
        else:
            subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        input(f"\n  {C.DIM}Press Enter...{C.END}")


# ── 5. Serve ──────────────────────────────────────────────────────────
def screen_serve():
    clear()
    _header()
    print(f"  {C.BOLD}🚀 Inference Server{C.END}")
    print(f"  {'─' * 46}\n")

    server_up = check_server()
    if server_up:
        print(f"  {C.GREEN}● Server is RUNNING on http://127.0.0.1:8420{C.END}")
    else:
        print(f"  {C.DIM}○ Server is not running{C.END}")
    print()

    options = [
        ("1", "Start server (foreground)"),
        ("2", "Start server (background)"),
        ("3", "Stop server"),
        ("4", "Health check"),
        ("0", "← Back"),
    ]
    for k, v in options:
        print(f"  {C.BOLD}{C.CYAN}[{k}]{C.END}  {v}")
    print()

    choice = input(f"  {C.BOLD}▶ {C.END}").strip()
    python = str(VENV_PYTHON)

    if choice in ("1", "2"):
        # Find best checkpoint
        checkpoint = None
        if CHECKPOINT_DIR.exists():
            for cp in sorted(CHECKPOINT_DIR.iterdir(), reverse=True):
                best = cp / "best.pt"
                if best.exists():
                    checkpoint = str(best)
                    break
        if not checkpoint:
            print(C.err("No checkpoint found — train a model first"))
            input(f"\n  {C.DIM}Press Enter...{C.END}")
            return

        port = input(f"  Port [{C.DIM}8420{C.END}]: ").strip() or "8420"
        cmd = [python, "-m", "inference.server",
               "--model", checkpoint,
               "--config", str(CONFIG_PATH),
               "--port", port]

        print(f"\n  {C.info(f'Checkpoint: {checkpoint}')}")
        print(f"  {C.info(f'Port: {port}')}")

        if choice == "2":
            _start_bg("server", cmd, "server.log")
            time.sleep(2)
            if check_server(int(port)):
                print(f"\n  {C.GREEN}● Server is UP at http://127.0.0.1:{port}{C.END}")
                print(f"  {C.info('In Blender → AI Copilot → set server to http://127.0.0.1:' + port)}")
            else:
                print(f"\n  {C.warn('Server may still be loading... check: tail -f server.log')}")
        else:
            print(f"\n  {C.info('Starting server (Ctrl+C to stop)...')}\n")
            try:
                subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            except KeyboardInterrupt:
                print(f"\n  {C.ok('Server stopped')}")

        input(f"\n  {C.DIM}Press Enter...{C.END}")

    elif choice == "3":
        _stop_bg("server")
        # Also try to kill any server process on port 8420
        try:
            result = subprocess.run(
                ["lsof", "-ti", ":8420"], capture_output=True, text=True)
            if result.stdout.strip():
                for pid in result.stdout.strip().split("\n"):
                    os.kill(int(pid), signal.SIGTERM)
                    print(C.ok(f"Killed process {pid} on port 8420"))
        except Exception:
            pass
        input(f"\n  {C.DIM}Press Enter...{C.END}")

    elif choice == "4":
        import urllib.request
        try:
            with urllib.request.urlopen("http://127.0.0.1:8420/health", timeout=3) as r:
                data = json.loads(r.read())
                print(f"\n  {C.GREEN}● Server healthy{C.END}")
                print(f"     Status:  {data.get('status')}")
                print(f"     Device:  {data.get('device')}")
                print(f"     Params:  {data.get('model_params', 0):,}")
        except Exception as e:
            print(f"\n  {C.RED}✗ Server not reachable: {e}{C.END}")
        input(f"\n  {C.DIM}Press Enter...{C.END}")


# ── 6. Test Generation ────────────────────────────────────────────────
def screen_test():
    clear()
    _header()
    print(f"  {C.BOLD}🧪 Test Generation{C.END}")
    print(f"  {'─' * 46}\n")

    if not check_server():
        print(f"  {C.RED}✗ Server not running. Start it first (option 5).{C.END}")
        input(f"\n  {C.DIM}Press Enter...{C.END}")
        return

    print(f"  {C.GREEN}● Server is running{C.END}")
    print(f"  {C.DIM}Type a prompt to generate a mesh, or 'q' to go back.{C.END}\n")

    while True:
        prompt = input(f"  {C.BOLD}Prompt:{C.END} ").strip()
        if not prompt or prompt.lower() == "q":
            break

        print(f"  {C.DIM}Generating...{C.END}", end="", flush=True)
        import urllib.request
        try:
            payload = json.dumps({
                "prompt": prompt,
                "temperature": 0.8,
                "top_k": 50,
                "max_faces": 512,
            }).encode()
            req = urllib.request.Request(
                "http://127.0.0.1:8420/generate/mesh",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            start = time.time()
            with urllib.request.urlopen(req, timeout=120) as r:
                result = json.loads(r.read())
            elapsed = time.time() - start

            print(f"\r  {C.GREEN}✓ Generated in {elapsed:.1f}s{C.END}     ")
            for obj in result.get("objects", []):
                mesh = obj.get("mesh", {})
                print(f"    Name: {obj.get('name')}")
                print(f"    Vertices: {mesh.get('num_vertices', 0)}")
                print(f"    Faces: {mesh.get('num_faces', 0)}")
            print(f"    Tokens: {result.get('token_count', 0)}")
            print()

            # Offer to save
            save = input(f"  {C.DIM}Save result? (y/N): {C.END}").strip().lower()
            if save == "y":
                out = PROJECT_ROOT / "test_outputs"
                out.mkdir(exist_ok=True)
                fname = out / f"gen_{int(time.time())}.json"
                with open(fname, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"  {C.ok(f'Saved to {fname.name}')}")
            print()

        except Exception as e:
            print(f"\r  {C.RED}✗ Error: {e}{C.END}     ")
            print()


# ── 7. Package Plugin ─────────────────────────────────────────────────
def screen_package():
    clear()
    _header()
    print(f"  {C.BOLD}📦 Package Blender Plugin{C.END}")
    print(f"  {'─' * 46}\n")

    addon_dir = PROJECT_ROOT.parent / "AIHouseGenerator"
    if not addon_dir.exists():
        print(C.err(f"Addon not found at {addon_dir}"))
        input(f"\n  {C.DIM}Press Enter...{C.END}")
        return

    print(f"  Addon source: {addon_dir}")
    print()

    options = [
        ("1", "Build ZIP (original plugin — OpenAI backend)"),
        ("2", "Build ZIP with local model support (adds local_client.py)"),
        ("0", "← Back"),
    ]
    for k, v in options:
        print(f"  {C.BOLD}{C.CYAN}[{k}]{C.END}  {v}")
    print()

    choice = input(f"  {C.BOLD}▶ {C.END}").strip()

    if choice in ("1", "2"):
        zip_name = "AIHouseGenerator.zip"
        zip_path = PROJECT_ROOT.parent / zip_name

        # Collect files
        files = [
            "__init__.py", "properties.py", "preferences.py", "panels.py",
            "operators.py", "ai_engine.py", "blender_tools.py",
            "materials.py", "oauth.py", "tool_defs.py", "README.md",
        ]

        if choice == "2":
            # Copy local_client to addon dir
            src = PROJECT_ROOT / "inference" / "local_client.py"
            dst = addon_dir / "local_client.py"
            if src.exists():
                import shutil
                shutil.copy2(src, dst)
                files.append("local_client.py")
                print(f"  {C.ok('Copied local_client.py into addon')}")

        # Build zip
        import zipfile
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in files:
                fpath = addon_dir / fname
                if fpath.exists():
                    zf.write(fpath, f"AIHouseGenerator/{fname}")
                else:
                    print(C.warn(f"Missing: {fname}"))

        sz = zip_path.stat().st_size / 1024
        print(f"\n  {C.ok(f'Built {zip_path} ({sz:.0f} KB)')}")
        print(f"  {C.info('Install in Blender: Edit → Preferences → Add-ons → Install...')}")

        input(f"\n  {C.DIM}Press Enter...{C.END}")


# ── 8. Settings ───────────────────────────────────────────────────────
def screen_settings():
    clear()
    _header()
    print(f"  {C.BOLD}🔧 Settings & Info{C.END}")
    print(f"  {'─' * 46}\n")

    print(f"  {C.BOLD}Paths{C.END}")
    print(f"     Project:     {PROJECT_ROOT}")
    print(f"     Data:        {DATA_DIR}")
    print(f"     Checkpoints: {CHECKPOINT_DIR}")
    print(f"     Python:      {VENV_PYTHON}")
    print()

    print(f"  {C.BOLD}Device{C.END}")
    print(f"     {detect_device()}")
    print()

    print(f"  {C.BOLD}Config Files{C.END}")
    for cp in [CONFIG_PATH, FULL_CONFIG_PATH]:
        exists = "✓" if cp.exists() else "✗"
        print(f"     {exists} {cp.name}")
    print()

    print(f"  {C.BOLD}Disk Usage{C.END}")
    for name, path in [("Raw data", RAW_DIR), ("Processed", PROCESSED_DIR),
                       ("Datasets", DATASET_DIR), ("Checkpoints", CHECKPOINT_DIR)]:
        sz = dir_size_mb(path)
        print(f"     {name:.<24s} {sz:,.0f} MB")
    print()

    input(f"  {C.DIM}Press Enter...{C.END}")


# ── b. Background tasks ──────────────────────────────────────────────
def screen_background():
    clear()
    _header()
    print(f"  {C.BOLD}📋 Background Tasks{C.END}")
    print(f"  {'─' * 46}\n")

    alive = _bg_status()
    if not alive:
        print(f"  {C.DIM}No background tasks running.{C.END}")
    else:
        for i, name in enumerate(alive, 1):
            print(f"  {C.YELLOW}⟳{C.END} {name}")
    print()

    # Also check for any process on port 8420
    try:
        r = subprocess.run(["lsof", "-ti", ":8420"], capture_output=True, text=True)
        if r.stdout.strip():
            pids = r.stdout.strip().split("\n")
            print(f"  {C.info(f'Port 8420: {len(pids)} process(es) bound')}")
    except Exception:
        pass

    # Show recent log files
    print(f"\n  {C.BOLD}Log files:{C.END}")
    for log in sorted(PROJECT_ROOT.glob("*.log")):
        age = time.time() - log.stat().st_mtime
        if age < 86400:
            h, m = divmod(int(age), 3600)
            mins = m // 60
            ago = f"{h}h{mins}m ago" if h else f"{mins}m ago"
        else:
            ago = f"{age/86400:.0f}d ago"
        print(f"     {log.name:.<32s} {C.DIM}{ago}{C.END}")
    print()

    if alive:
        stop = input(f"  Stop a task? Enter name (or Enter to go back): ").strip()
        if stop:
            for name in list(_bg_processes):
                if stop.lower() in name.lower():
                    _stop_bg(name)
                    break
            else:
                print(C.warn(f"No task matching '{stop}'"))
        input(f"\n  {C.DIM}Press Enter...{C.END}")
    else:
        input(f"  {C.DIM}Press Enter...{C.END}")


# ══════════════════════════════════════════════════════════════════════
# Main Loop
# ══════════════════════════════════════════════════════════════════════

def run():
    """Main interactive loop."""
    screens = {
        "1": screen_dashboard,
        "2": screen_data,
        "3": screen_process,
        "4": screen_train,
        "5": screen_serve,
        "6": screen_test,
        "7": screen_package,
        "8": screen_settings,
        "b": screen_background,
    }

    while True:
        try:
            choice = main_menu()
            if choice in ("q", "quit", "exit"):
                # Warn about bg tasks
                alive = _bg_status()
                if alive:
                    print(f"\n  {C.YELLOW}⚠ {len(alive)} background task(s) still running:{C.END}")
                    for a in alive:
                        print(f"    {a}")
                    kill = input(f"  Stop them? [y/N]: ").strip().lower()
                    if kill == "y":
                        for name in list(_bg_processes):
                            _stop_bg(name)
                print(f"\n  {C.DIM}Goodbye! 👋{C.END}\n")
                break
            elif choice in screens:
                screens[choice]()
            else:
                pass  # Invalid — just redraw menu
        except KeyboardInterrupt:
            print(f"\n  {C.DIM}(Ctrl+C) Back to menu...{C.END}")
            time.sleep(0.3)
        except EOFError:
            break


if __name__ == "__main__":
    run()
