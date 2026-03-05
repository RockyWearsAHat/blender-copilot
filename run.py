#!/usr/bin/env python3
"""blender-copilot — unified CLI.

    python run.py status              Show progress dashboard
    python run.py data                Full pipeline: download→extract→render→VL-label→cache (scraping ON)
    python run.py data --local        Disk-only reprocess (no network, no scraping)
    python run.py data --test         Test mode: 1 item per source
    python run.py data --pull-max 50  Process 50 items then stop
    python run.py data --pull-behave batch   One source at a time
    python run.py scrape              Download training data
    python run.py extract             Extract mesh data from .blend files
    python run.py build               Build .pt training cache
    python run.py render              Batch render cached meshes via Blender
    python run.py train               Train the unified model
    python run.py trace-cache        Generate real collapse traces (Milestone 1)
    python run.py trace-terrain      Generate synthetic terrain traces (Milestone 1)
    python run.py train-policy       Train compact policy transformer
    python run.py rollout-policy     Closed-loop policy rollout in Blender
    python run.py clean               Clean artifacts for a fresh training restart
    python run.py serve               Start inference server (hot-reload)
    python run.py pipeline            Run scrape → extract → build → render → train
    python run.py lambda-setup HOST   Set up a cloud GPU instance
    python run.py lambda-train HOST   Start training on cloud GPU
    python run.py lambda-sync HOST    Sync checkpoints from cloud
    python run.py lambda-kill HOST    Kill training on cloud GPU
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional
import os

# ── Constants ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DATASET_DIR = DATA_DIR / "datasets"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

BLENDER_EXE = "/Applications/Blender.app/Contents/MacOS/Blender"
EXTRACT_SCRIPT = PROJECT_ROOT / "scripts" / "extract_blends.py"
CACHE_SCRIPT = PROJECT_ROOT / "scripts" / "rebuild_cache.py"
RENDER_SCRIPT = PROJECT_ROOT / "scripts" / "render_cache.py"
TRACE_CACHE_SCRIPT = PROJECT_ROOT / "scripts" / "generate_collapse_traces_from_cache.py"
TRACE_TERRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "generate_collapse_traces_synthetic_terrain.py"
TRAIN_POLICY_SCRIPT = PROJECT_ROOT / "training" / "train_policy.py"
ROLLOUT_POLICY_SCRIPT = PROJECT_ROOT / "scripts" / "rollout_policy_closed_loop.py"

CLOUD_REMOTE_DIR = "/home/ubuntu/BlenderGPT/blender-copilot"

logger = logging.getLogger("run")


# ── Terminal colors ───────────────────────────────────────────────────────
class C:
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    DIM = "\033[2m"
    END = "\033[0m"

    @staticmethod
    def ok(msg): return f"{C.GREEN}✓{C.END} {msg}"
    @staticmethod
    def warn(msg): return f"{C.YELLOW}⚠{C.END} {msg}"
    @staticmethod
    def err(msg): return f"{C.RED}✗{C.END} {msg}"
    @staticmethod
    def info(msg): return f"{C.BLUE}ℹ{C.END} {msg}"
    @staticmethod
    def step(n, msg): return f"{C.CYAN}[{n}]{C.END} {C.BOLD}{msg}{C.END}"


def setup_logging(verbose: bool = False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(config_path: Optional[Path] = None) -> dict:
    import yaml
    path = config_path or CONFIG_PATH
    if not path.exists():
        print(C.err(f"Config not found: {path}"))
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


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


# ── Active status helpers ────────────────────────────────────────────────
AUTO_STATUS_BEGIN = "<!-- AUTO-STATUS:BEGIN -->"
AUTO_STATUS_END = "<!-- AUTO-STATUS:END -->"


def _read_json_if_exists(path: Path) -> Optional[dict]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _resolve_checkpoint_arg(checkpoint: str) -> tuple[str, Optional[str]]:
    """Resolve a checkpoint argument to an on-disk checkpoint path.

    This is mainly to make `--checkpoint checkpoints/_active/train_latest.pt`
    reliable even if the symlink hasn't been written yet, was broken by an
    earlier bug, or was removed during cleanup.
    """
    p = Path(checkpoint)
    if p.exists():
        return checkpoint, None

    note: Optional[str] = None
    active_dir = CHECKPOINT_DIR / "_active"
    # If user points at an _active symlink that is missing, try to regenerate
    # pointers from logs/active_train.json and logs/active_serve.json.
    try:
        if active_dir in p.parents:
            train = _read_json_if_exists(PROJECT_ROOT / "logs" / "active_train.json")
            serve = _read_json_if_exists(PROJECT_ROOT / "logs" / "active_serve.json")
            if train or serve:
                write_active_status(train=train, serve=serve)
                note = "regenerated checkpoints/_active pointers from logs"
    except Exception:
        pass

    if p.exists():
        return checkpoint, note

    # If it's a known _active alias, fall back to the actual run output dir.
    if p.name in {"train_latest.pt", "train_best.pt"}:
        train = _read_json_if_exists(PROJECT_ROOT / "logs" / "active_train.json")
        if train:
            out = Path(train.get("output_dir", ""))
            if out and not out.is_absolute():
                out = PROJECT_ROOT / out
            if out:
                cand = out / ("latest.pt" if p.name == "train_latest.pt" else "best.pt")
                try:
                    if cand.exists():
                        # Also heal the missing symlink for next time.
                        _safe_symlink(active_dir / p.name, cand)
                        healed = "healed _active symlink"
                        return str(cand), f"{note}; {healed}" if note else healed
                except Exception:
                    pass

    if p.name == "served_checkpoint.pt":
        serve = _read_json_if_exists(PROJECT_ROOT / "logs" / "active_serve.json")
        ck = (serve or {}).get("checkpoint")
        if ck and Path(ck).exists():
            try:
                _safe_symlink(active_dir / "served_checkpoint.pt", Path(ck))
            except Exception:
                pass
            healed = "healed served pointer from active_serve.json"
            return ck, f"{note}; {healed}" if note else healed

    return checkpoint, note


def _safe_symlink(link_path: Path, target_path: Path) -> None:
    """Best-effort symlink creation without clobbering real files/dirs."""
    try:
        if link_path.exists() or link_path.is_symlink():
            if link_path.is_symlink() or link_path.is_file():
                link_path.unlink(missing_ok=True)
            else:
                # Don't delete real directories.
                return
        # Resolve the target to an absolute path rooted at the repo.
        # This avoids broken links when `target_path` is relative.
        try:
            abs_target = target_path if target_path.is_absolute() else (PROJECT_ROOT / target_path)
            abs_target = abs_target.resolve()
        except Exception:
            abs_target = target_path

        # Prefer a relative symlink for portability.
        try:
            rel = os.path.relpath(str(abs_target), start=str(link_path.parent.resolve()))
            link_path.symlink_to(rel)
        except Exception:
            link_path.symlink_to(abs_target)
    except Exception:
        pass


def _write_json(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass


def _format_auto_status(train: Optional[dict], serve: Optional[dict]) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    served_suffix = ""
    if serve and serve.get("checkpoint"):
        served_suffix = f" (→ {serve.get('checkpoint')})"
    health_url = "http://127.0.0.1:8420/health"
    if serve and serve.get("health_url"):
        health_url = str(serve.get("health_url"))
    lines = [
        f"Last updated: {now}",
        "",
        "Training (active):",
        f"- Run: {train.get('run_name') if train else '(unknown)'}",
        f"- Config: {train.get('config') if train else '(unknown)'}",
        f"- Output dir: {train.get('output_dir') if train else '(unknown)'}",
        "- Checkpoint being written: checkpoints/_active/train_latest.pt",
        "- Train log: logs/train_latest.log",
        "- Monitor log: logs/monitor_latest.log",
        "",
        "Serving (Blender):",
        f"- URL: {health_url}",
        f"- Served checkpoint: checkpoints/_active/served_checkpoint.pt{served_suffix}",
    ]
    return "\n".join(lines) + "\n"


def _update_readme_auto_status(
    *,
    train: Optional[dict] = None,
    serve: Optional[dict] = None,
    readme_path: Path | None = None,
) -> None:
    """Replace README auto-status block between marker comments."""
    rp = readme_path or (PROJECT_ROOT / "README.md")
    try:
        text = rp.read_text(encoding="utf-8")
    except Exception:
        return

    if AUTO_STATUS_BEGIN not in text or AUTO_STATUS_END not in text:
        return

    pre, rest = text.split(AUTO_STATUS_BEGIN, 1)
    _, post = rest.split(AUTO_STATUS_END, 1)
    block = _format_auto_status(train, serve)
    new_text = pre + AUTO_STATUS_BEGIN + "\n" + block + AUTO_STATUS_END + post
    try:
        rp.write_text(new_text, encoding="utf-8")
    except Exception:
        pass


def write_active_status(*, train: Optional[dict] = None, serve: Optional[dict] = None) -> None:
    """Write stable pointers for 'what is active right now'.

    - logs/active_train.json and logs/active_serve.json
    - checkpoints/_active/* symlinks
    - README auto-status block
    """
    logs_dir = PROJECT_ROOT / "logs"
    active_dir = CHECKPOINT_DIR / "_active"
    try:
        active_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    if train:
        _write_json(logs_dir / "active_train.json", train)
        out = Path(train.get("output_dir", ""))
        if out.is_absolute():
            try:
                out_rel = out.relative_to(PROJECT_ROOT)
                out = PROJECT_ROOT / out_rel
            except Exception:
                pass
        if out:
            _safe_symlink(active_dir / "train_run", out)
            _safe_symlink(active_dir / "train_latest.pt", out / "latest.pt")
            try:
                best_path = (out / "best.pt")
                best_abs = best_path if best_path.is_absolute() else (PROJECT_ROOT / best_path)
                if best_abs.exists():
                    _safe_symlink(active_dir / "train_best.pt", best_path)
            except Exception:
                pass

    if serve:
        _write_json(logs_dir / "active_serve.json", serve)
        ckpt = serve.get("checkpoint")
        if ckpt:
            _safe_symlink(active_dir / "served_checkpoint.pt", Path(ckpt))

    # Load what we have (so README always reflects both sides if present)
    train_final = train
    serve_final = serve
    try:
        if train_final is None:
            train_final = json.loads((logs_dir / "active_train.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    try:
        if serve_final is None:
            serve_final = json.loads((logs_dir / "active_serve.json").read_text(encoding="utf-8"))
    except Exception:
        pass

    _update_readme_auto_status(train=train_final, serve=serve_final)


def _rel_to_root(p: Path) -> Path:
    try:
        return p.resolve().relative_to(PROJECT_ROOT)
    except Exception:
        return Path(p.name)


def _unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.name
    parent = dest.parent
    for i in range(1, 10_000):
        cand = parent / f"{stem}__{i}"
        if not cand.exists():
            return cand
    raise RuntimeError(f"Could not find unique destination for: {dest}")


def _iter_pycache_dirs(root: Path) -> list[Path]:
    skip_names = {".git", ".venv", "venv", "node_modules"}
    hits: list[Path] = []
    for d in root.rglob("__pycache__"):
        if not d.is_dir():
            continue
        parts = set(d.parts)
        if parts & skip_names:
            continue
        hits.append(d)
    return hits


def _rm_path(p: Path) -> None:
    if not p.exists() and not p.is_symlink():
        return
    if p.is_dir() and not p.is_symlink():
        shutil.rmtree(p)
    else:
        p.unlink(missing_ok=True)


def _archive_or_delete(
    target: Path,
    *,
    archive_root: Path,
    delete_instead: bool,
    allow_delete_without_archive: bool = False,
) -> str:
    """Return a short status string describing the action."""
    if not target.exists() and not target.is_symlink():
        return "skip (missing)"

    if delete_instead or allow_delete_without_archive:
        _rm_path(target)
        return "deleted"

    rel = _rel_to_root(target)
    dest = _unique_dest(archive_root / rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(target), str(dest))
    return f"moved → {dest.relative_to(PROJECT_ROOT)}"


# ═══════════════════════════════════════════════════════════════════════════
# STATUS
# ═══════════════════════════════════════════════════════════════════════════

def cmd_status(args):
    """Show current progress of data collection and training."""
    print(f"\n{C.BOLD}{'═' * 60}{C.END}")
    print(f"{C.BOLD}  blender-copilot — Status{C.END}")
    print(f"{C.BOLD}{'═' * 60}{C.END}\n")

    # Device
    print(f"  {C.info(f'Device: {detect_device()}')}")
    print()

    # ── Raw Data ──
    print(f"  {C.BOLD}📁 Raw Data (downloaded .blend files){C.END}")
    raw_sources = [
        ("Blender Official",  RAW_DIR / "blender_official", "*.blend"),
        ("BlendSwap",          RAW_DIR / "blendswap",        "*.blend"),
        ("SmutBase",           RAW_DIR / "smutbase",         "*.blend"),
        ("GitHub",             RAW_DIR / "github",           "*.blend"),
        ("Objaverse (GLB)",    RAW_DIR / "objaverse" / "models", "*.glb"),
        ("Thingiverse",        RAW_DIR / "thingiverse",      "*.obj"),
        ("Sketchfab",          RAW_DIR / "sketchfab",        "*.glb"),
        ("Wikimedia",          RAW_DIR / "wikimedia",        "*.glb"),
        ("Smithsonian (XL)",   Path.home() / ".objaverse" / "smithsonian" / "objects", "*.glb"),
    ]
    total_raw = 0
    for name, path, pattern in raw_sources:
        if path.exists():
            # Count with multiple extensions for new sources
            if name in ("Thingiverse", "Sketchfab", "Wikimedia"):
                exts = ["*.blend", "*.glb", "*.obj", "*.stl", "*.fbx"]
                files = []
                for ext in exts:
                    files.extend(list(path.rglob(ext)))
            else:
                files = list(path.rglob(pattern))
            n = len(files)
        else:
            n = 0
        total_raw += n
        sz = dir_size_mb(path) if path.exists() else 0
        status = C.ok(f"{n:,} files ({sz:.0f} MB)") if n > 0 else C.warn("0 files")
        print(f"     {name:.<30s} {status}")
    print(f"     {'TOTAL':.<30s} {C.BOLD}{total_raw:,} files{C.END}")
    print()

    # ── Processed Data ──
    print(f"  {C.BOLD}⚙️  Processed Data (extracted JSON){C.END}")
    total_proc = 0
    total_objects = 0
    for proc_subdir in sorted(PROCESSED_DIR.iterdir()) if PROCESSED_DIR.exists() else []:
        if not proc_subdir.is_dir() or proc_subdir.name.startswith("."):
            continue
        n_json = count_files(proc_subdir, "*.json")
        n_invalid = count_files(proc_subdir, "*.invalid")
        total_proc += n_json

        # Count mesh objects inside JSONs (sample first 10 for speed)
        obj_count = 0
        sampled = 0
        for jf in list(proc_subdir.glob("*.json"))[:10]:
            try:
                with open(jf) as f:
                    data = json.load(f)
                obj_count += len(data.get("objects", []))
                sampled += 1
            except Exception:
                pass
        if n_json > 10 and sampled > 0:
            avg = obj_count / sampled
            est_total = int(avg * n_json)
            total_objects += est_total
            obj_str = f"~{est_total:,} objects"
        elif obj_count > 0:
            total_objects += obj_count
            obj_str = f"{obj_count:,} objects"
        else:
            obj_str = ""

        parts = [f"{n_json:,} JSON"]
        if n_invalid > 0:
            parts.append(f"{n_invalid} invalid")
        if obj_str:
            parts.append(obj_str)
        status = C.ok(", ".join(parts)) if n_json > 0 else C.warn("empty")
        print(f"     {proc_subdir.name:.<30s} {status}")

    print(f"     {'TOTAL':.<30s} {C.BOLD}{total_proc:,} files, ~{total_objects:,} objects{C.END}")
    print()

    # ── Training Cache ──
    print(f"  {C.BOLD}📊 Training Cache (.pt files){C.END}")
    cache_dir = PROCESSED_DIR / ".mesh_cache"
    if cache_dir.exists():
        n_pt = count_files(cache_dir, "*.pt")
        sz = dir_size_mb(cache_dir)
        print(f"     {'Cache':.<30s} {C.ok(f'{n_pt:,} samples ({sz:.0f} MB)')}")
    else:
        print(f"     {'Cache':.<30s} {C.warn('not built — run: python run.py build')}")
    print()

    # ── Checkpoints ──
    print(f"  {C.BOLD}🏋️  Model Checkpoints{C.END}")
    if CHECKPOINT_DIR.exists():
        for cp_dir in sorted(CHECKPOINT_DIR.glob("*")):
            if not cp_dir.is_dir():
                continue
            parts = []
            for fname in ["latest.pt", "best.pt", "final.pt"]:
                p = cp_dir / fname
                if p.exists():
                    sz = p.stat().st_size / (1024 * 1024)
                    parts.append(f"{fname} ({sz:.0f}MB)")
            steps = list(cp_dir.glob("step_*.pt"))
            if steps:
                parts.append(f"{len(steps)} step checkpoints")
            if parts:
                print(f"     {cp_dir.name:.<30s} {C.ok(', '.join(parts))}")
            else:
                print(f"     {cp_dir.name:.<30s} {C.warn('empty')}")
    else:
        print(f"     {'(none)':.<30s} {C.warn('no training yet')}")
    print()

    # ── Config ──
    print(f"  {C.BOLD}⚙️  Config{C.END}")
    try:
        cfg = load_config()
        uni = cfg.get("unified", {})
        tr = cfg.get("training", {})
        print(f"     Model: {uni.get('embed_dim', '?')}d, "
              f"{uni.get('num_geometry_layers', '?')} layers, "
              f"{uni.get('num_heads', '?')} heads")
        print(f"     Training: batch={tr.get('batch_size', '?')}, "
              f"lr={tr.get('learning_rate', '?')}, "
              f"max_steps={tr.get('max_steps', '?')}")
    except Exception:
        print(f"     {C.warn('Could not load config.yaml')}")

    print(f"\n{C.BOLD}{'═' * 60}{C.END}\n")


# ═══════════════════════════════════════════════════════════════════════════
# SCRAPE
# ═══════════════════════════════════════════════════════════════════════════

def cmd_scrape(args):
    """Download training data from all sources."""
    print(f"\n{C.step(1, 'Downloading training data...')}\n")

    sources = args.sources or ["blender", "blendswap"]
    config = load_config(args.config)

    for source in sources:
        if source in ("blender", "blender_official"):
            print(f"  {C.step('→', 'Blender Official demo files...')}")
            from scrapers.blender_official import download_blender_official
            output = Path(args.output or "data/raw/blender_official")
            output.mkdir(parents=True, exist_ok=True)
            download_blender_official(
                output,
                max_size_mb=getattr(args, "max_size", 500) or 500,
                crawl=not getattr(args, "no_crawl", False),
            )

        elif source == "blendswap":
            print(f"  {C.step('→', 'BlendSwap models...')}")
            try:
                from scrapers.blendswap_scraper import scrape_blendswap
                output = Path(args.output or "data/raw/blendswap")
                output.mkdir(parents=True, exist_ok=True)
                scrape_blendswap(str(output), config=config)
            except Exception as e:
                print(C.err(f"BlendSwap failed: {e}"))

        elif source == "smutbase":
            print(f"  {C.step('→', 'SmutBase character models...')}")
            try:
                from scrapers.smutbase_scraper import scrape_site
                scrape_site(
                    site_key="smutbase",
                    output_dir=args.output or "data/raw/smutbase",
                    max_pages=getattr(args, "max_pages", 250) or 250,
                )
            except Exception as e:
                print(C.err(f"SmutBase failed: {e}"))

        elif source == "open3dlab":
            print(f"  {C.step('→', 'Open3DLab models...')}")
            try:
                from scrapers.smutbase_scraper import scrape_site
                scrape_site(
                    site_key="open3dlab",
                    output_dir=args.output or "data/raw/open3dlab",
                    max_pages=getattr(args, "max_pages", 250) or 250,
                )
            except Exception as e:
                print(C.err(f"Open3DLab failed: {e}"))

        elif source == "github":
            print(f"  {C.step('→', 'GitHub .blend repos...')}")
            try:
                from scrapers.github_scraper import scrape_github
                output = Path(args.output or "data/raw/github")
                output.mkdir(parents=True, exist_ok=True)
                scrape_github(output, config=config)
            except Exception as e:
                print(C.err(f"GitHub failed: {e}"))

        elif source == "objaverse":
            print(f"  {C.step('→', 'Objaverse-XL models...')}")
            from scrapers.objaverse_scraper import download_all_sources
            output = Path(args.output or "data/raw/objaverse")
            output.mkdir(parents=True, exist_ok=True)
            max_per = None
            if getattr(args, "max_models", None):
                max_per = {s: args.max_models for s in
                           ["sketchfab", "github", "smithsonian", "thingiverse"]}
            download_all_sources(output, max_per_source=max_per, processes=4)

        elif source == "all":
            print(C.info("Scraping ALL sources..."))
            args.sources = [
                "blender", "blendswap", "smutbase",
                "open3dlab", "github", "objaverse",
                "thingiverse", "sketchfab",
                "wikimedia", "terminology",
            ]
            cmd_scrape(args)
            return

        elif source == "thingiverse":
            print(f"  {C.step('→', 'Thingiverse 3D models...')}")
            try:
                from scrapers.thingiverse_scraper import scrape_thingiverse
                output = Path(args.output or "data/raw/thingiverse")
                output.mkdir(parents=True, exist_ok=True)
                scrape_thingiverse(
                    str(output),
                    max_pages=getattr(args, "max_pages", 50) or 50,
                    config=config,
                )
            except Exception as e:
                print(C.err(f"Thingiverse failed: {e}"))

        elif source == "sketchfab":
            print(f"  {C.step('→', 'Sketchfab 3D models...')}")
            try:
                from scrapers.sketchfab_scraper import scrape_sketchfab
                output = Path(args.output or "data/raw/sketchfab")
                output.mkdir(parents=True, exist_ok=True)
                scrape_sketchfab(
                    str(output),
                    max_pages=getattr(args, "max_pages", 20) or 20,
                    config=config,
                )
            except Exception as e:
                print(C.err(f"Sketchfab failed: {e}"))

        elif source == "wikimedia":
            print(f"  {C.step('→', 'Wikimedia Commons 3D assets...')}")
            try:
                from scrapers.wikimedia_scraper import scrape_wikimedia
                output = Path(args.output or "data/raw/wikimedia")
                output.mkdir(parents=True, exist_ok=True)
                scrape_wikimedia(str(output), max_per_query=getattr(args, "max_pages", 20) or 20)
            except Exception as e:
                print(C.err(f"Wikimedia failed: {e}"))

        elif source == "terminology":
            print(f"  {C.step('→', 'CG terminology glossary...')}")
            try:
                from scrapers.terminology_scraper import scrape_terminology
                output = Path(args.output or "data/raw/terminology")
                output.mkdir(parents=True, exist_ok=True)
                scrape_terminology(str(output))
            except Exception as e:
                print(C.err(f"Terminology failed: {e}"))

    print(f"\n{C.ok('Scraping complete!')}\n")


# ═══════════════════════════════════════════════════════════════════════════
# EXTRACT — crash-isolated per-file extraction
# ═══════════════════════════════════════════════════════════════════════════

def cmd_extract(args):
    """Extract mesh data from downloaded .blend files.

    Uses crash-isolated subprocess extraction — each .blend runs in its own
    Blender process so a segfault on one file doesn't kill the batch.
    """
    print(f"\n{C.step(2, 'Extracting mesh data (crash-isolated)...')}\n")

    source_dirs = [
        ("blender_official", RAW_DIR / "blender_official"),
        ("blendswap",        RAW_DIR / "blendswap"),
        ("smutbase",         RAW_DIR / "smutbase"),
        ("open3dlab",        RAW_DIR / "open3dlab"),
        ("github",           RAW_DIR / "github"),
    ]

    timeout = getattr(args, "timeout", 420) or 420
    retry_timeout = getattr(args, "retry_timeout", None)
    timeout_grace = getattr(args, "timeout_grace", 180)
    workers = max(1, int(getattr(args, "workers", 2) or 2))
    retry_immediate = bool(getattr(args, "retry_immediate", False))
    mark_timeout_invalid = bool(getattr(args, "mark_timeout_invalid", False))

    for name, raw_dir in source_dirs:
        if not raw_dir.exists():
            continue

        blend_files = list(raw_dir.rglob("*.blend"))
        blend_files = [f for f in blend_files if not f.name.startswith("._")]
        if not blend_files:
            continue

        output_dir = PROCESSED_DIR / name
        output_dir.mkdir(parents=True, exist_ok=True)

        pending = []
        for bf in blend_files:
            out_json = output_dir / f"{bf.stem}.json"
            out_invalid = output_dir / f"{bf.stem}.invalid"
            if not out_json.exists() and not out_invalid.exists():
                pending.append(bf)

        done = len(blend_files) - len(pending)
        print(f"  {C.BOLD}{name}{C.END}: {len(blend_files)} total, "
              f"{done} done, {len(pending)} pending")

        if not pending:
            continue

        if EXTRACT_SCRIPT.exists():
            cmd = [
                sys.executable,
                str(EXTRACT_SCRIPT),
                "--input", str(raw_dir),
                "--output", str(output_dir),
                "--timeout", str(timeout),
                "--workers", str(workers),
            ]
            if retry_timeout:
                cmd += ["--retry-timeout", str(retry_timeout)]
            else:
                cmd += ["--timeout-grace", str(timeout_grace)]
            if retry_immediate:
                cmd.append("--retry-immediate")
            if mark_timeout_invalid:
                cmd.append("--mark-timeout-invalid")
            subprocess.run(cmd)
        else:
            extractor = PROJECT_ROOT / "processing" / "blend_extractor.py"
            for bf in pending:
                print(f"    {bf.name}...")
                try:
                    subprocess.run([
                        BLENDER_EXE, "--background", "--python",
                        str(extractor), "--",
                        "--input", str(bf), "--output", str(output_dir),
                    ], capture_output=True, timeout=timeout)
                except subprocess.TimeoutExpired:
                    print(f"      {C.warn('timeout')}")
                except Exception as e:
                    print(f"      {C.err(str(e))}")

    # Objaverse GLB files (no Blender needed)
    objaverse_raw = RAW_DIR / "objaverse" / "models"
    if objaverse_raw.exists() and count_files(objaverse_raw, "*.glb") > 0:
        n = count_files(objaverse_raw, "*.glb")
        print(f"\n  {C.BOLD}objaverse{C.END}: {n} GLB files (mesh_extractor)")
        from processing.mesh_extractor import process_directory
        config = load_config(args.config)
        process_directory(
            objaverse_raw, PROCESSED_DIR / "objaverse",
            metadata_dir=RAW_DIR / "objaverse" / "metadata",
            config=config,
        )

    # Smithsonian Institution GLBs from ObjaverseXL (stored in ~/.objaverse/smithsonian)
    smithsonian_cache = Path.home() / ".objaverse" / "smithsonian" / "objects"
    if smithsonian_cache.exists():
        sm_glbs = list(smithsonian_cache.glob("*.glb"))
        if sm_glbs:
            print(f"\n  {C.BOLD}smithsonian{C.END}: {len(sm_glbs)} CC-0 GLB files (mesh_extractor)")
            try:
                from processing.mesh_extractor import process_directory
                config = load_config(args.config)
                output_dir = PROCESSED_DIR / "smithsonian"
                output_dir.mkdir(parents=True, exist_ok=True)
                # Build a metadata stub for each GLB using the Smithsonian parquet
                sm_meta_dir = PROCESSED_DIR / "smithsonian" / "metadata"
                sm_meta_dir.mkdir(parents=True, exist_ok=True)
                try:
                    import objaverse.xl as oxl
                    import pandas as pd
                    ann = oxl.smithsonian.SmithsonianDownloader.get_annotations()
                    # Write per-file metadata JSONL stubs
                    sm_meta_file = sm_meta_dir / "smithsonian_annotations.jsonl"
                    if not sm_meta_file.exists():
                        with open(sm_meta_file, "w") as f:
                            for _, row in ann.iterrows():
                                meta = {
                                    "uid": str(row.get("fileIdentifier", "")),
                                    "label": str(row.get("title", row.get("name", "smithsonian object"))),
                                    "source": "smithsonian",
                                    "license": "CC0",
                                    "tags": ["smithsonian", "natural history", "museum"],
                                }
                                f.write(json.dumps(meta) + "\n")
                        print(f"    Wrote {len(ann)} metadata stubs")
                except Exception as e:
                    print(f"    {C.warn(f'Could not write metadata stubs: {e}')}")

                process_directory(
                    smithsonian_cache,
                    output_dir,
                    metadata_dir=sm_meta_dir,
                    config=config,
                )
            except Exception as e:
                print(C.err(f"  Smithsonian extraction failed: {e}"))


    # New sources — Thingiverse, Sketchfab, Wikimedia (GLB/OBJ/STL files)
    for new_source, glb_patterns in [
        ("thingiverse", ["*.blend", "*.glb", "*.obj", "*.stl"]),
        ("sketchfab",   ["*.glb", "*.fbx", "*.obj"]),
        ("wikimedia",   ["*.glb", "*.obj", "*.stl"]),
    ]:
        src_dir = RAW_DIR / new_source
        if not src_dir.exists():
            continue
        all_files = []
        for pat in glb_patterns:
            all_files.extend(list(src_dir.rglob(pat)))
        all_files = [f for f in all_files if f.is_file()]
        if not all_files:
            continue
        print(f"\n  {C.BOLD}{new_source}{C.END}: {len(all_files)} files (mesh_extractor)")
        try:
            from processing.mesh_extractor import process_directory
            config = load_config(args.config)
            output_dir = PROCESSED_DIR / new_source
            output_dir.mkdir(parents=True, exist_ok=True)
            process_directory(
                src_dir,
                output_dir,
                metadata_dir=src_dir,
                config=config,
            )
        except Exception as e:
            print(C.err(f"  {new_source} extraction failed: {e}"))

    # Summary
    total = 0
    for d in PROCESSED_DIR.iterdir() if PROCESSED_DIR.exists() else []:
        if d.is_dir() and not d.name.startswith("."):
            total += count_files(d, "*.json")
    print(f"\n{C.ok(f'Extraction complete! {total} processed JSON files.')}\n")


# ═══════════════════════════════════════════════════════════════════════════
# BUILD — build .pt training cache
# ═══════════════════════════════════════════════════════════════════════════

def cmd_build(args):
    """Build .pt training cache from processed JSONs."""
    print(f"\n{C.step(3, 'Building training cache...')}\n")

    if CACHE_SCRIPT.exists():
        cmd = [sys.executable, str(CACHE_SCRIPT)]
        if getattr(args, "dry_run", False):
            cmd.append("--dry-run")
        subprocess.run(cmd)
    else:
        print(C.err(f"Cache build script not found: {CACHE_SCRIPT}"))
        print(C.info("Expected: scripts/rebuild_cache.py"))

    cache_dir = PROCESSED_DIR / ".mesh_cache"
    if cache_dir.exists():
        n_pt = count_files(cache_dir, "*.pt")
        sz = dir_size_mb(cache_dir)
        print(f"\n{C.ok(f'Cache built: {n_pt:,} samples ({sz:.0f} MB)')}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# TRAIN
# ═══════════════════════════════════════════════════════════════════════════

def cmd_train(args):
    """Train the unified model. Runs until Ctrl+C."""
    import torch
    import os

    print(f"\n{C.step(4, 'Training unified model...')}\n")

    config = load_config(args.config)
    unified_cfg = config.get("unified", {})

    if torch.cuda.is_available():
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_name = "Apple Silicon GPU (MPS)"
    else:
        device_name = "CPU"

    run_name = getattr(args, "name", None) or "unified"
    output_dir = CHECKPOINT_DIR / run_name

    if getattr(args, "batch_size", None):
        config.setdefault("training", {})["batch_size"] = args.batch_size

    print(f"  {C.BOLD}Unified Training{C.END}")
    print(f"  {'─' * 50}")
    print(f"  Device:       {device_name}")
    print(f"  Model:        {unified_cfg.get('embed_dim', 512)}d unified")
    print(f"  Output:       {output_dir}")
    print("  Resume:       auto (from latest checkpoint)")
    print("  Stop:         Ctrl+C (saves gracefully)")
    print(f"  {'─' * 50}\n")

    from training.train_unified import train as run_training

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    train_log_path = logs_dir / f"train_runpy_{ts}.log"
    latest_link = logs_dir / "train_latest.log"

    # Mirror Python logging to a stable train log so monitor can auto-detect.
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(train_log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    root_logger.addHandler(file_handler)

    try:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(train_log_path.name)
    except Exception:
        # Best effort only; monitor can still use explicit path.
        pass

    print(f"  Train log:     {train_log_path}")
    print(f"  Monitor cmd:   {sys.executable} scripts/monitor.py --train-log {train_log_path}")

    try:
        cfg_rel = str(Path(args.config).resolve().relative_to(PROJECT_ROOT))
    except Exception:
        cfg_rel = str(getattr(args, "config", "config.yaml"))

    train_status = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "run_name": run_name,
        "config": cfg_rel,
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "resume": getattr(args, "resume", "latest") or "latest",
        "train_log": str(train_log_path.relative_to(PROJECT_ROOT)),
        "train_log_latest": "logs/train_latest.log",
        "monitor_log": "logs/monitor_latest.log" if getattr(args, "monitor", True) else None,
    }
    write_active_status(train=train_status)

    if getattr(args, "monitor", True):
        monitor_script = PROJECT_ROOT / "scripts" / "monitor.py"
        monitor_log = logs_dir / "monitor_latest.log"
        try:
            mon = subprocess.Popen(
                [
                    sys.executable,
                    str(monitor_script),
                    "--train-log",
                    str(train_log_path),
                ],
                cwd=str(PROJECT_ROOT),
                stdout=monitor_log.open("w", encoding="utf-8"),
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            print(f"  Monitor:       started (pid={mon.pid}) → {monitor_log}")
            train_status["monitor_pid"] = mon.pid
            write_active_status(train=train_status)
        except Exception as e:
            print(C.warn(f"  Monitor auto-start failed: {e}"))

    class TrainArgs:
        output: str = ""
        resume: str = "latest"
    t_args = TrainArgs()
    t_args.output = str(output_dir)
    t_args.resume = getattr(args, "resume", "latest") or "latest"

    try:
        run_training(config, t_args)
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()


# ═══════════════════════════════════════════════════════════════════════════
# SERVE
# ═══════════════════════════════════════════════════════════════════════════

def cmd_serve(args):
    """Start the inference server with hot-reload."""
    print(f"\n{C.step(5, 'Starting inference server...')}\n")

    checkpoint = getattr(args, "checkpoint", None)
    if not checkpoint:
        # Prefer modern unified runs (semantic bootstrap) and prefer best.pt
        # over latest.pt to avoid serving a regressed in-progress snapshot.
        preferred_runs = [
            "unified_semantic_bootstrap",
            "unified",
        ]

        candidates = []
        for run_name in preferred_runs:
            cp_dir = CHECKPOINT_DIR / run_name
            if cp_dir.exists() and cp_dir.is_dir():
                candidates.append(cp_dir)

        # Also include any other unified* run directories, newest first.
        for cp_dir in sorted(
            [p for p in CHECKPOINT_DIR.glob("unified*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            if cp_dir not in candidates:
                candidates.append(cp_dir)

        for cp_dir in candidates:
            for fname in ["best.pt", "latest.pt", "final.pt"]:
                p = cp_dir / fname
                if p.exists():
                    checkpoint = str(p)
                    break
            if checkpoint:
                break

    if not checkpoint:
        print(C.err("No checkpoint found — run: python run.py train"))
        return

    # Resolve/heal `_active` symlink arguments.
    checkpoint, ckpt_note = _resolve_checkpoint_arg(str(checkpoint))
    if ckpt_note:
        print(C.warn(f"Checkpoint path healed: {ckpt_note}"))
    if not Path(checkpoint).exists():
        print(C.err(f"Checkpoint not found: {checkpoint}"))
        print(C.info("Tip: run `python run.py train` once to write checkpoints/_active pointers"))
        return

    config = load_config(args.config)
    port = getattr(args, "port", 8420) or 8420
    watch = getattr(args, "watch_interval", 30.0) or 30.0

    print(f"  Checkpoint:  {checkpoint}")
    print(f"  Port:        {port}")
    print(f"  Hot-reload:  every {watch}s")
    print()
    print(f"  {C.BOLD}Blender connects to: http://127.0.0.1:{port}{C.END}")
    print(f"  {C.info(f'Test: curl http://127.0.0.1:{port}/health')}")
    print()

    try:
        serve_cfg_rel = str(Path(args.config).resolve().relative_to(PROJECT_ROOT))
    except Exception:
        serve_cfg_rel = str(getattr(args, "config", "config.yaml"))
    try:
        ckpt_rel = str(Path(checkpoint).resolve().relative_to(PROJECT_ROOT)) if checkpoint else None
    except Exception:
        ckpt_rel = str(checkpoint) if checkpoint else None

    serve_status = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "config": serve_cfg_rel,
        "checkpoint_arg": str(getattr(args, "checkpoint", "")) or None,
        "checkpoint": ckpt_rel,
        "port": int(port),
        "device": getattr(args, "device", "auto") or "auto",
        "health_url": f"http://127.0.0.1:{port}/health",
    }
    write_active_status(serve=serve_status)

    from inference.server import _STATE, _watch_checkpoint, create_app
    import threading

    device_str = getattr(args, "device", "auto") or "auto"
    _STATE.load(checkpoint, config, device_str)

    watcher = threading.Thread(
        target=_watch_checkpoint, args=(_STATE, watch), daemon=True)
    watcher.start()

    app = create_app(_STATE)

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=port)


# ═══════════════════════════════════════════════════════════════════════════
# EVAL — geometric evaluation
# ═══════════════════════════════════════════════════════════════════════════

def cmd_eval(args):
    """Run geometric evaluation on a checkpoint."""
    import torch

    print(f"\n{C.BOLD}{'═' * 60}{C.END}")
    print(f"{C.BOLD}  Geometric Evaluation{C.END}")
    print(f"{C.BOLD}{'═' * 60}{C.END}\n")

    config = load_config(args.config)

    device_str = getattr(args, "device", "auto") or "auto"
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    checkpoint_path = args.checkpoint
    if not Path(checkpoint_path).exists():
        print(C.err(f"Checkpoint not found: {checkpoint_path}"))
        return

    print(f"  Checkpoint:  {checkpoint_path}")
    print(f"  Device:      {device}")
    print(f"  Max faces:   {args.max_faces}")
    print(f"  Temperature: {args.temperature}")
    print()

    print(f"  {C.info('Loading model...')}")
    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)

    from models.unified import UnifiedBlenderModel
    model = UnifiedBlenderModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    from processing.mesh_tokenizer import MeshTokenizer
    tok_config = config.get("tokenization", {})
    mesh_tokenizer = MeshTokenizer(
        vocab_size=tok_config.get("vocab_size", 8192),
        coord_range=tuple(tok_config.get("coordinate_range", [-1.0, 1.0])),
        max_faces=tok_config.get("max_faces", 2048),
    )

    from processing.bpe_tokenizer import BPETokenizer
    data_cfg = config.get("data", {})
    geo_dir = Path(data_cfg.get("geometry_dir", "data/datasets/geometry"))
    bpe_dir = geo_dir / "bpe_tokenizer"
    text_tokenizer = BPETokenizer.load(bpe_dir)

    step = checkpoint.get("step", checkpoint.get("global_step", 0))

    from evaluation.harness import run_geometric_eval
    results = run_geometric_eval(
        model, mesh_tokenizer, text_tokenizer,
        device, step, config,
        max_faces=args.max_faces,
        temperature=args.temperature,
    )

    s = results["summary"]
    print(f"\n{C.BOLD}Results (step {step}):{C.END}")
    gen_rate_str = f"{s['generation_rate']:.0%}"
    exp_rate_str = f"{s['expectations_rate']:.0%}"
    print(f"  Generation rate:   {C.ok(gen_rate_str)}")
    print(f"  Expectations met:  {C.ok(exp_rate_str)}")
    if "validity_score_mean" in s:
        print(f"  Validity score:    {s['validity_score_mean']:.3f}")
    if "face_count_mean" in s:
        print(f"  Avg face count:    {s['face_count_mean']:.0f}")
    print(f"  Time:              {results['elapsed_seconds']:.1f}s")
    print()

    for cat, data in s.get("by_category", {}).items():
        gen_rate = data["generated"] / max(data["total"], 1)
        rate_str = f"{gen_rate:.0%}"
        status = C.ok(rate_str) if gen_rate > 0.5 else C.warn(rate_str)
        print(f"  [{cat:12s}] {data['generated']}/{data['total']} generated ({status})")

    print()


# ═══════════════════════════════════════════════════════════════════════════
# CLEAN — safe cleanup for restarting training
# ═══════════════════════════════════════════════════════════════════════════


def cmd_clean(args):
    """Clean training artifacts (dry-run by default)."""
    if getattr(args, "all", False):
        args.checkpoints = True
        args.cache = True
        args.wandb = True
        args.eval = True
        args.renders = True
        args.feedback = True
        args.pycache = True

    # Scope defaulting: if no scopes are selected, clean the training artifacts
    # that are most likely to cause “resume from old state”.
    any_scope = any([
        getattr(args, "checkpoints", False),
        getattr(args, "cache", False),
        getattr(args, "wandb", False),
        getattr(args, "eval", False),
        getattr(args, "renders", False),
        getattr(args, "feedback", False),
        getattr(args, "pycache", False),
    ])
    if not any_scope:
        args.checkpoints = True
        args.wandb = True
        args.eval = True

    apply_changes = getattr(args, "apply", False)
    delete_instead = getattr(args, "rm", False)
    run_name = getattr(args, "run", None)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = Path(getattr(args, "archive_dir", "") or (PROJECT_ROOT / "_trash" / f"clean_{ts}")).resolve()

    # Collect explicit targets (dirs/files). For pycache we handle separately.
    targets: list[Path] = []

    if getattr(args, "checkpoints", False):
        if run_name and run_name != "all":
            targets.append(CHECKPOINT_DIR / run_name)
        else:
            targets.append(CHECKPOINT_DIR)

    if getattr(args, "cache", False):
        targets.extend([
            PROCESSED_DIR / ".mesh_cache",
            PROCESSED_DIR / ".mesh_cache_quarantine",
            PROCESSED_DIR / ".mesh_cache_backup",
        ])

    if getattr(args, "renders", False):
        targets.append(DATA_DIR / "renders")

    if getattr(args, "eval", False):
        eval_dir = DATA_DIR / "eval"
        targets.extend([
            eval_dir / "results.jsonl",
        ])
        if eval_dir.exists():
            targets.extend(sorted(eval_dir.glob("scaling_curve_*.json")))

    if getattr(args, "feedback", False):
        fb_dir = DATA_DIR / "feedback"
        targets.append(fb_dir / "feedback_buffer.jsonl")

    if getattr(args, "wandb", False):
        targets.extend([
            PROJECT_ROOT / "wandb",
            PROJECT_ROOT / ".wandb",
            PROJECT_ROOT / "runs",
        ])

    # Junk dirs (safe to delete outright).
    junk: list[Path] = []
    if getattr(args, "pycache", False):
        junk.extend([
            PROJECT_ROOT / ".pytest_cache",
            PROJECT_ROOT / ".coverage",
            PROJECT_ROOT / "coverage_html",
        ])
        junk.extend(_iter_pycache_dirs(PROJECT_ROOT))

    # Dedupe while preserving order
    seen: set[Path] = set()
    deduped_targets: list[Path] = []
    for t in targets:
        tr = t.resolve() if t.exists() else t
        if tr in seen:
            continue
        seen.add(tr)
        deduped_targets.append(t)

    print(f"\n{C.step('🧹', 'Clean training artifacts (safe)')}\n")
    print(f"  Mode:        {'APPLY' if apply_changes else 'DRY-RUN'}")
    if apply_changes:
        if delete_instead:
            action_str = "DELETE (permanent)"
        else:
            try:
                action_str = f"ARCHIVE → {archive_root.relative_to(PROJECT_ROOT)}"
            except Exception:
                action_str = f"ARCHIVE → {archive_root}"
        print(f"  Action:      {action_str}")
    print(f"  Scopes:      "
          f"checkpoints={getattr(args, 'checkpoints', False)}, "
          f"cache={getattr(args, 'cache', False)}, "
          f"wandb={getattr(args, 'wandb', False)}, "
          f"eval={getattr(args, 'eval', False)}, "
          f"renders={getattr(args, 'renders', False)}, "
          f"feedback={getattr(args, 'feedback', False)}, "
          f"pycache={getattr(args, 'pycache', False)}")
    if getattr(args, "checkpoints", False) and run_name and run_name != "all":
        print(f"  Run filter:  {run_name}")
    print()

    if not deduped_targets and not junk:
        print(C.warn("Nothing selected."))
        return

    # Preview
    def _fmt(p: Path) -> str:
        rel = _rel_to_root(p)
        return str(rel)

    if deduped_targets:
        print(f"  {C.BOLD}Targets{C.END}")
        for p in deduped_targets:
            exists = p.exists()
            print(f"    - {_fmt(p)}{' (missing)' if not exists else ''}")
        print()

    if junk:
        print(f"  {C.BOLD}Junk (always safe to delete){C.END}")
        for p in junk:
            print(f"    - {_fmt(p)}{' (missing)' if not p.exists() else ''}")
        print()

    # Apply
    if not apply_changes:
        print(C.info("Dry-run only. Re-run with --apply to perform changes."))
        print(C.info("Tip: add --rm to delete instead of archiving."))
        return

    if not delete_instead:
        archive_root.mkdir(parents=True, exist_ok=True)

    changed = 0
    for p in deduped_targets:
        try:
            status = _archive_or_delete(
                p,
                archive_root=archive_root,
                delete_instead=delete_instead,
            )
            if not status.startswith("skip"):
                changed += 1
            print(f"  {C.ok(_fmt(p))} — {status}")
        except Exception as e:
            print(f"  {C.err(_fmt(p))} — {e}")

    # Junk: delete outright (don’t archive, it’s noise)
    for p in junk:
        try:
            status = _archive_or_delete(
                p,
                archive_root=archive_root,
                delete_instead=True,
                allow_delete_without_archive=True,
            )
            if not status.startswith("skip"):
                changed += 1
            print(f"  {C.ok(_fmt(p))} — {status}")
        except Exception as e:
            print(f"  {C.err(_fmt(p))} — {e}")

    print()
    if changed == 0:
        print(C.warn("No changes made."))
    else:
        print(C.ok(f"Cleanup complete ({changed} item(s))."))
        if not delete_instead:
            print(C.info(f"Archived under: {archive_root}"))



# ═══════════════════════════════════════════════════════════════════════════
# DATA-AUDIT — training data diversity and quality
# ═══════════════════════════════════════════════════════════════════════════

def cmd_data_audit(args):
    """Audit training data diversity and quality."""
    import torch
    from collections import Counter

    print(f"\n{C.BOLD}{'═' * 60}{C.END}")
    print(f"{C.BOLD}  Training Data Audit{C.END}")
    print(f"{C.BOLD}{'═' * 60}{C.END}\n")

    cache_dir = PROCESSED_DIR / ".mesh_cache"
    if not cache_dir.exists():
        print(C.err("No cache directory found. Run: python run.py build"))
        return

    pt_files = list(cache_dir.glob("*.pt"))
    print(f"  {C.info(f'Found {len(pt_files):,} cached samples')}")
    print()

    labels = []
    face_counts = []
    token_lengths = []
    sources = Counter()
    label_counter = Counter()
    token_hashes = set()
    duplicates = 0

    for pf in pt_files:
        try:
            raw = torch.load(pf, map_location="cpu", weights_only=True)
            # .pt files may be a list of items or a single dict
            items = raw if isinstance(raw, list) else [raw]
            for data in items:
                if not isinstance(data, dict):
                    continue
                label = data.get("label", data.get("text", "unknown"))
                labels.append(label)
                label_counter[label] += 1

                mesh_tokens = data.get("mesh_tokens")
                if mesh_tokens is not None:
                    tlen = len(mesh_tokens) if hasattr(mesh_tokens, '__len__') else mesh_tokens.numel()
                    token_lengths.append(tlen)
                    n_faces = (tlen - 2) // 9
                    face_counts.append(n_faces)

                    tok_hash = hash(tuple(mesh_tokens.tolist()
                                          if hasattr(mesh_tokens, 'tolist')
                                          else mesh_tokens))
                    if tok_hash in token_hashes:
                        duplicates += 1
                    token_hashes.add(tok_hash)

                src = data.get("source", pf.stem.split("_")[0] if "_" in pf.stem else "unknown")
                sources[src] += 1
        except Exception:
            pass

    if not labels:
        print(C.warn("  No valid samples found"))
        return

    import numpy as np

    print(f"  {C.BOLD}Dataset Size{C.END}")
    print(f"     Total samples:     {len(labels):,}")
    print(f"     Unique labels:     {len(label_counter):,}")
    print(f"     Duplicate meshes:  {duplicates:,}")
    effective = len(labels) - duplicates
    print(f"     Effective samples: {effective:,}")
    print()

    if face_counts:
        fc = np.array(face_counts)
        print(f"  {C.BOLD}Face Count Distribution{C.END}")
        print(f"     Min:    {fc.min():,}")
        print(f"     Max:    {fc.max():,}")
        print(f"     Mean:   {fc.mean():.0f}")
        print(f"     Median: {np.median(fc):.0f}")
        print(f"     Std:    {fc.std():.0f}")

        bins = [0, 50, 100, 200, 500, 1000, 2000, 4000, 8000]
        print("     Histogram:")
        for i in range(len(bins) - 1):
            count = int(((fc >= bins[i]) & (fc < bins[i+1])).sum())
            bar = '█' * min(count * 40 // len(fc), 40) if len(fc) > 0 else ''
            print(f"       {bins[i]:>5d}-{bins[i+1]:<5d}  {count:>5d}  {bar}")
        over = int((fc >= bins[-1]).sum())
        if over > 0:
            print(f"       {bins[-1]:>5d}+       {over:>5d}")
        print()

    if sources:
        print(f"  {C.BOLD}Data Sources{C.END}")
        for src, count in sources.most_common():
            pct = count / len(labels) * 100
            print(f"     {src:.<25s} {count:>5d} ({pct:.1f}%)")
        print()

    unique_labels = len(label_counter)
    entropy = 0.0
    max_entropy = 0.0
    if unique_labels > 0:
        freqs = np.array(list(label_counter.values()), dtype=float)
        probs = freqs / freqs.sum()
        entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
        max_entropy = float(np.log2(unique_labels))

        print(f"  {C.BOLD}Label Diversity{C.END}")
        print(f"     Label entropy:     {entropy:.2f} bits "
              f"(max: {max_entropy:.2f})")
        print(f"     Normalized:        {entropy/max_entropy:.2%}")
        print(f"     Top-1 label freq:  {label_counter.most_common(1)[0][1]:,} "
              f"({label_counter.most_common(1)[0][0]!r})")
        print(f"     Singleton labels:  "
              f"{sum(1 for v in label_counter.values() if v == 1):,}")
        print()

    if getattr(args, "detailed", False) and label_counter:
        print(f"  {C.BOLD}Top 30 Labels{C.END}")
        for label, count in label_counter.most_common(30):
            pct = count / len(labels) * 100
            print(f"     {label[:40]:.<42s} {count:>4d} ({pct:.1f}%)")
        print()

    print(f"  {C.BOLD}Quality Summary{C.END}")
    dup_ratio = duplicates / max(len(labels), 1) * 100
    quality = "Good" if dup_ratio < 5 else "Fair" if dup_ratio < 15 else "Poor"
    print(f"     Dedup quality:  {quality} ({dup_ratio:.1f}% duplicates)")
    if unique_labels > 0:
        div_ratio = entropy / max_entropy if max_entropy > 0 else 0
        div_quality = ("Good" if div_ratio > 0.8 else
                       "Fair" if div_ratio > 0.6 else "Poor")
        print(f"     Label diversity: {div_quality} "
              f"({div_ratio:.0%} of max entropy)")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# DATA-QUALITY — comprehensive quality report (source, entropy, cap, etc.)
# ═══════════════════════════════════════════════════════════════════════════

def cmd_data_quality(args):
    """Run the comprehensive data quality report."""
    import subprocess, sys
    script = PROJECT_ROOT / "scripts" / "data_quality_report.py"
    cmd = [sys.executable, str(script), f"--top={args.top}", f"--cap={args.cap}"]
    if getattr(args, "fix_in_place", False):
        cmd.append("--fix-in-place")
    subprocess.run(cmd, check=False)


# ═══════════════════════════════════════════════════════════════════════════
# VL-RELABEL — relabel cache items using Qwen VL vision model
# ═══════════════════════════════════════════════════════════════════════════

def cmd_vl_relabel(args):
    """Relabel .pt cache files using rendered images + Qwen VL."""
    import subprocess as _sp, sys as _sys
    script = PROJECT_ROOT / "scripts" / "vl_relabel_cache.py"
    cmd = [_sys.executable, str(script)]
    if getattr(args, "max", None):
        cmd += ["--max", str(args.max)]
    if getattr(args, "skip_existing", False):
        cmd.append("--skip-existing")
    if getattr(args, "dry_run", False):
        cmd.append("--dry-run")
    cmd += ["--model", getattr(args, "model", "qwen2.5vl:32b")]
    if getattr(args, "timeout", None):
        cmd += ["--timeout", str(args.timeout)]
    _sp.run(cmd, check=False)


# ═══════════════════════════════════════════════════════════════════════════
# DATA — unified download→extract→render→VL-label→cache pipeline
# ═══════════════════════════════════════════════════════════════════════════

def cmd_data(args):
    """Full data pipeline: download → extract → render → VL label → cache.

    Live scraping is ON by default (pulls fresh models from open3dlab, blendswap,
    smutbase, objaverse, etc.).  Use --local to skip scraping and only reprocess
    .blend files already on disk.
    """
    import subprocess as _sp, sys as _sys
    script = PROJECT_ROOT / "scripts" / "data_pipeline.py"
    cmd = [_sys.executable, str(script)]
    sources = getattr(args, "sources", None) or []
    if sources:
        cmd += ["--sources"] + sources
    if getattr(args, "pull_max", None):
        cmd += ["--pull-max", str(args.pull_max)]
    if getattr(args, "pull_behave", None):
        cmd += ["--pull-behave", args.pull_behave]
    if getattr(args, "workers", None):
        cmd += ["--workers", str(args.workers)]
    if getattr(args, "keep_raw", False):
        cmd.append("--keep-raw")
    if getattr(args, "test", False):
        cmd.append("--test")
    if getattr(args, "local", False):
        cmd.append("--local")
    if getattr(args, "ignore_parse_markers", False):
        cmd.append("--ignore-parse-markers")
    _sp.run(cmd, check=False)


# ═══════════════════════════════════════════════════════════════════════════
# RENDER — batch render cached meshes → contrastive training images
# ═══════════════════════════════════════════════════════════════════════════

def cmd_render(args):
    """Batch render .pt cache files to multi-view PNGs for image-to-mesh training.

    Calls scripts/render_cache.py which:
      1. Reads .pt files from data/processed/.mesh_cache/
      2. Decodes mesh tokens → vertices + faces
      3. Launches Blender headless to render multi-view PNGs
      4. Saves renders to data/renders/ with manifest JSONs
      5. Optionally writes rendered image tensors back into .pt cache files
         so ContrastiveStream can load them during training.
    """
    print(f"\n{C.step(3, 'Rendering cached meshes...')}\n")

    renders_dir = DATA_DIR / "renders"
    cache_dir = PROCESSED_DIR / ".mesh_cache"

    if not cache_dir.exists() or count_files(cache_dir, "*.pt") == 0:
        print(C.warn("No .pt cache files found. Run 'python run.py build' first."))
        return

    n_cached = count_files(cache_dir, "*.pt")
    print(f"  Cache:    {n_cached:,} samples in {cache_dir}")
    print(f"  Output:   {renders_dir}")

    if not RENDER_SCRIPT.exists():
        print(C.err(f"Render script not found: {RENDER_SCRIPT}"))
        return

    cmd = [sys.executable, str(RENDER_SCRIPT)]

    max_samples = getattr(args, "max_samples", None)
    if max_samples:
        cmd += ["--max-samples", str(max_samples)]

    workers = getattr(args, "workers", 1)
    cmd += ["--workers", str(workers)]

    size = getattr(args, "size", 256)
    cmd += ["--size", str(size)]

    views = getattr(args, "views", 4)
    cmd += ["--views", str(views)]

    if getattr(args, "skip_existing", True):
        cmd.append("--skip-existing")

    if getattr(args, "embed", False):
        cmd.append("--embed-in-cache")

    subprocess.run(cmd)

    n_renders = count_files(renders_dir, "*.png") if renders_dir.exists() else 0
    print(f"\n{C.ok(f'Render complete: {n_renders:,} PNG images in {renders_dir}')}")
    print(C.info("These renders are loaded by ContrastiveStream during training."))
    print()


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE — full end-to-end
# ═══════════════════════════════════════════════════════════════════════════

def cmd_pipeline(args):
    """Run full pipeline: scrape → extract → build → render → train."""
    print(f"\n{C.BOLD}{'═' * 60}{C.END}")
    print(f"{C.BOLD}  blender-copilot — Full Pipeline{C.END}")
    print(f"{C.BOLD}{'═' * 60}{C.END}\n")

    if not getattr(args, "skip_scrape", False):
        args.sources = getattr(args, "sources", None) or ["blender", "blendswap"]
        args.output = None
        cmd_scrape(args)

    cmd_extract(args)
    cmd_build(args)

    if not getattr(args, "skip_render", False):
        args.max_samples = None
        args.workers = 1
        args.size = 256
        args.views = 4
        args.skip_existing = True
        args.embed = True
        cmd_render(args)

    if not getattr(args, "skip_train", False):
        cmd_train(args)

    print(f"\n{C.ok('Pipeline complete!')}")
    print("  Next: python run.py serve")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# POLICY MILESTONE 1 — trace generation, policy train, closed-loop rollout
# ═══════════════════════════════════════════════════════════════════════════

def cmd_trace_cache(args):
    """Generate collapse traces from real cache meshes (Milestone 1)."""
    print(f"\n{C.step('M1', 'Generating collapse traces from cache...')}\n")

    if not TRACE_CACHE_SCRIPT.exists():
        print(C.err(f"Missing script: {TRACE_CACHE_SCRIPT}"))
        return

    cmd = [
        sys.executable,
        str(TRACE_CACHE_SCRIPT),
        "--cache-dir",
        str(args.cache_dir),
        "--out-dir",
        str(args.out_dir),
        "--blender",
        str(args.blender),
        "--max-files",
        str(int(args.max_files)),
        "--max-steps",
        str(int(args.max_steps)),
        "--target-verts",
        str(int(args.target_verts)),
        "--timeout-s",
        str(int(args.timeout_s)),
        "--prompts-per-mesh",
        str(int(args.prompts_per_mesh)),
    ]
    if bool(args.skip_existing):
        cmd.append("--skip-existing")

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        print(C.err("Collapse trace generation failed."))
        return

    traces = count_files(Path(args.out_dir), "*/trace.jsonl")
    print(f"\n{C.ok(f'Collapse traces ready: {traces:,} trace files under {args.out_dir}')}")
    print()


def cmd_trace_terrain(args):
    """Generate collapse traces from synthetic terrain meshes (Milestone 1)."""
    print(f"\n{C.step('M1', 'Generating synthetic terrain collapse traces...')}\n")

    if not TRACE_TERRAIN_SCRIPT.exists():
        print(C.err(f"Missing script: {TRACE_TERRAIN_SCRIPT}"))
        return

    cmd = [
        sys.executable,
        str(TRACE_TERRAIN_SCRIPT),
        "--out-dir",
        str(args.out_dir),
        "--blender",
        str(args.blender),
        "--n",
        str(int(args.n)),
        "--seed",
        str(int(args.seed)),
        "--max-steps",
        str(int(args.max_steps)),
        "--target-verts",
        str(int(args.target_verts)),
    ]
    if bool(args.skip_existing):
        cmd.append("--skip-existing")

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        print(C.err("Synthetic terrain trace generation failed."))
        return

    traces = count_files(Path(args.out_dir), "*/trace.jsonl")
    print(f"\n{C.ok(f'Synthetic terrain traces complete: {traces:,} total trace files under {args.out_dir}')}")
    print()


def cmd_train_policy(args):
    """Train compact policy transformer (Milestone 1)."""
    print(f"\n{C.step('M1', 'Training policy transformer...')}\n")

    if not TRAIN_POLICY_SCRIPT.exists():
        print(C.err(f"Missing script: {TRAIN_POLICY_SCRIPT}"))
        return

    cmd = [
        sys.executable,
        str(TRAIN_POLICY_SCRIPT),
        "--config",
        str(args.policy_config),
        "--device",
        str(args.device),
        "--max-steps",
        str(int(args.max_steps)),
    ]
    if bool(args.resume):
        cmd.append("--resume")
    if bool(args.compile):
        cmd.append("--compile")

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        print(C.err("Policy training failed."))
        return

    print(f"\n{C.ok('Policy training finished.')}")
    print()


def cmd_rollout_policy(args):
    """Run closed-loop policy rollout in real Blender (Milestone 1)."""
    print(f"\n{C.step('M1', 'Running closed-loop policy rollout...')}\n")

    if not ROLLOUT_POLICY_SCRIPT.exists():
        print(C.err(f"Missing script: {ROLLOUT_POLICY_SCRIPT}"))
        return

    cmd = [
        sys.executable,
        str(ROLLOUT_POLICY_SCRIPT),
        "--ckpt",
        str(args.ckpt),
        "--out-dir",
        str(args.out_dir),
        "--steps",
        str(int(args.steps)),
        "--seed",
        str(int(args.seed)),
        "--device",
        str(args.device),
        "--blender",
        str(args.blender),
        "--temperature",
        str(float(args.temperature)),
        "--top-k",
        str(int(args.top_k)),
    ]

    if args.prompt:
        cmd.extend(["--prompt", str(args.prompt)])
    if args.goal_vertices is not None:
        cmd.extend(["--goal-vertices", str(int(args.goal_vertices))])
    if args.goal_symmetry is not None:
        cmd.extend(["--goal-symmetry", str(float(args.goal_symmetry))])
    if bool(args.deterministic):
        cmd.append("--deterministic")
    if bool(args.low_poly_bias):
        cmd.append("--low-poly-bias")

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        print(C.err("Closed-loop rollout failed."))
        return

    print(f"\n{C.ok(f'Rollout complete. Artifacts written to {args.out_dir}')}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# LAMBDA / CLOUD GPU COMMANDS
# ═══════════════════════════════════════════════════════════════════════════

def _require_host(args) -> str:
    host = getattr(args, "host", None)
    if not host:
        print(C.err("Cloud host required. Usage: python run.py lambda-setup user@host"))
        sys.exit(1)
    return host


def cmd_lambda_setup(args):
    """Upload code + data to cloud GPU and set up environment."""
    host = _require_host(args)
    print(f"\n{C.step(1, f'Setting up cloud instance: {host}')}\n")

    print(f"  {C.step('→', 'Uploading project code...')}")
    upload_script = PROJECT_ROOT / "cloud" / "upload_to_cloud.sh"
    subprocess.run(["bash", str(upload_script), host])

    print(f"\n  {C.step('→', 'Installing dependencies on remote...')}")
    subprocess.run([
        "ssh", host,
        f"cd {CLOUD_REMOTE_DIR} && bash cloud/setup_instance.sh"
    ])

    print(f"\n{C.ok('Cloud instance ready!')}")
    print(f"  Train: python run.py lambda-train {host}")
    print()


def cmd_lambda_train(args):
    """Start training on cloud GPU in a tmux session."""
    host = _require_host(args)
    print(f"\n{C.step('▶', f'Starting training on {host}...')}\n")

    subprocess.run([
        "ssh", host,
        f"cd {CLOUD_REMOTE_DIR} && "
        f"tmux kill-session -t train 2>/dev/null; "
        f"tmux new-session -d -s train 'bash cloud/train_cloud.sh'"
    ])

    print(C.ok("Training started in tmux session 'train'"))
    print(f"  Monitor: ssh {host} -t 'tmux attach -t train'")
    print(f"  Sync:    python run.py lambda-sync {host}")
    print(f"  Kill:    python run.py lambda-kill {host}")
    print()


def cmd_lambda_sync(args):
    """Sync checkpoints from cloud to local Mac."""
    host = _require_host(args)
    continuous = getattr(args, "continuous", False)

    print(f"\n{C.step('↓', f'Syncing checkpoints from {host}...')}\n")

    sync_script = PROJECT_ROOT / "cloud" / "sync_checkpoints.sh"
    cmd = ["bash", str(sync_script), host]
    if continuous:
        cmd.append("--continuous")
    subprocess.run(cmd)


def cmd_lambda_kill(args):
    """Kill training on cloud GPU."""
    host = _require_host(args)
    print(f"\n{C.step('■', f'Killing training on {host}...')}\n")

    subprocess.run([
        "ssh", host,
        "tmux kill-session -t train 2>/dev/null; "
        "pkill -f train_unified 2>/dev/null; "
        "pkill -f train_cloud 2>/dev/null; "
        "echo 'Training stopped.'"
    ])

    print(C.ok("Training stopped on remote."))
    print(f"  Sync final checkpoint: python run.py lambda-sync {host}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="blender-copilot — unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py status                         Show progress dashboard
  python run.py scrape --sources blendswap     Download BlendSwap models
  python run.py scrape --sources all           Download from ALL sources
  python run.py extract                        Extract .blend → JSON
  python run.py build                          Build .pt training cache
  python run.py render                         Render cached meshes → PNGs
  python run.py render --embed                 Render + embed tensors in .pt cache
  python run.py train                          Train the model
    python run.py trace-cache                    Generate real collapse traces
    python run.py trace-terrain                  Generate synthetic terrain traces
    python run.py train-policy                   Train compact policy transformer
    python run.py rollout-policy --ckpt ...      Closed-loop policy rollout
    python run.py clean --checkpoints --apply    Archive old checkpoints (safe)
  python run.py serve                          Start inference server
  python run.py pipeline                       Run everything end-to-end

  python run.py lambda-setup user@gpu-ip       Set up cloud GPU
  python run.py lambda-train user@gpu-ip       Start cloud training
  python run.py lambda-sync user@gpu-ip        Pull checkpoints
  python run.py lambda-kill user@gpu-ip        Stop cloud training
        """,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH,
                        help="Config file (default: config.yaml)")

    subs = parser.add_subparsers(dest="command", help="Command to run")

    # status
    subs.add_parser("status", help="Show progress dashboard")

    # data — unified pipeline
    sub = subs.add_parser(
        "data",
        help="Full pipeline: download → extract → render → VL label → cache (scraping ON by default)",
        description=(
            "Pulls fresh models from all sources (live scraping ON by default), processes each "
            "item through Blender extraction, quick multi-view render, and "
            "qwen2.5vl:32b labeling, then writes .pt cache entries. "
            "Use --local to skip scraping and only reprocess files already on disk."
        ),
    )
    sub.add_argument("--sources", nargs="*",
                     choices=["blender_official", "blendswap", "github",
                              "objaverse", "objaverse_xl", "open3dlab",
                              "smutbase", "youtube", "all"],
                     help="Which sources to pull from (default: all)")
    sub.add_argument("--pull-max", type=int, default=None,
                     help="Stop after N items total (default: run forever)")
    sub.add_argument("--pull-behave", default="concurrent",
                     choices=["concurrent", "batch"],
                     help=(
                         "concurrent: pull all sources simultaneously to "
                         "saturate bandwidth (default). "
                         "batch: one source at a time."
                     ))
    sub.add_argument("--workers", type=int, default=None,
                     help="Download/process worker threads (default: auto-detected)")
    sub.add_argument("--keep-raw", action="store_true",
                     help="Keep raw downloaded files after processing")
    sub.add_argument("--test", action="store_true",
                     help="Process exactly 1 item per source then exit")
    sub.add_argument("--local", action="store_true",
                     help=(
                         "Disk-only mode: skip all live scrapers. "
                         "Only reprocess .blend files already on disk. "
                         "Default: scraping is enabled."
                     ))
    sub.add_argument("--ignore-parse-markers", action="store_true",
                     help=(
                         "Ignore data/raw/<source>/.parsed_complete.json markers and "
                         "allow live re-downloads for marked sources."
                     ))

    # scrape
    sub = subs.add_parser("scrape", help="Download training data")
    sub.add_argument("--sources", nargs="*",
                     choices=["blender", "blendswap", "smutbase",
                              "open3dlab", "github", "objaverse",
                              "thingiverse", "sketchfab",
                              "wikimedia", "terminology", "all"],
                     help="Data sources (default: blender blendswap)")
    sub.add_argument("--output", help="Output directory")
    sub.add_argument("--max-size", type=float, default=500)
    sub.add_argument("--max-models", type=int, default=None)
    sub.add_argument("--max-pages", type=int, default=None)
    sub.add_argument("--no-crawl", action="store_true")

    # extract
    sub = subs.add_parser("extract", help="Extract .blend → JSON")
    sub.add_argument("--timeout", type=int, default=420,
                     help="Primary timeout per file in seconds (default: 420)")
    sub.add_argument("--retry-timeout", type=int, default=None,
                     help="One-time retry timeout after primary timeout")
    sub.add_argument("--timeout-grace", type=int, default=180,
                     help="Extra seconds for one retry if --retry-timeout is unset (default: 180)")
    sub.add_argument("--workers", type=int, default=2,
                     help="Parallel Blender extraction workers per source (default: 2)")
    sub.add_argument("--retry-immediate", action="store_true",
                     help="Retry timed out files immediately instead of deferring to a second pass")
    sub.add_argument("--mark-timeout-invalid", action="store_true",
                     help="Mark timed-out files as invalid (default: off; timed-out files stay retriable)")

    # build
    sub = subs.add_parser("build", help="Build .pt training cache")
    sub.add_argument("--dry-run", action="store_true",
                     help="Validate without writing files")

    # render
    sub = subs.add_parser("render", help="Batch render cached meshes → PNGs for image-to-mesh")
    sub.add_argument("--max-samples", type=int, default=None,
                     help="Max number of meshes to render (default: all)")
    sub.add_argument("--workers", type=int, default=1,
                     help="Parallel Blender processes (default: 1)")
    sub.add_argument("--size", type=int, default=256,
                     help="Render resolution in pixels (default: 256)")
    sub.add_argument("--views", type=int, default=4,
                     help="Camera viewpoints per mesh (default: 4)")
    sub.add_argument("--no-skip-existing", dest="skip_existing",
                     action="store_false", default=True,
                     help="Re-render even if PNGs already exist")
    sub.add_argument("--embed", action="store_true", default=False,
                     help="Write rendered image tensors back into .pt cache files")

    # train
    sub = subs.add_parser("train", help="Train unified model (Ctrl+C to stop)")
    sub.add_argument("--batch-size", type=int)
    sub.add_argument("--name", help="Run name for checkpoints")
    sub.add_argument("--resume", default="latest",
                     help="'latest' (default), checkpoint path, or 'none'")
    sub.add_argument("--monitor", action="store_true", default=True,
                     help="Auto-start scripts/monitor.py (writes to logs/monitor_latest.log)")
    sub.add_argument("--no-monitor", dest="monitor", action="store_false",
                     help="Do not auto-start monitor")

    # trace-cache (Milestone 1)
    sub = subs.add_parser(
        "trace-cache",
        help="Generate collapse traces from cache meshes (Milestone 1)",
    )
    sub.add_argument("--cache-dir", type=Path, default=Path("data/processed/.mesh_cache"))
    sub.add_argument("--out-dir", type=Path, default=Path("data/datasets/collapse_traces"))
    sub.add_argument("--blender", type=str, default=BLENDER_EXE)
    sub.add_argument("--max-files", type=int, default=1000)
    sub.add_argument("--max-steps", type=int, default=64)
    sub.add_argument("--target-verts", type=int, default=8)
    sub.add_argument("--timeout-s", type=int, default=180)
    sub.add_argument("--prompts-per-mesh", type=int, default=12)
    sub.add_argument("--skip-existing", action="store_true")

    # trace-terrain (Milestone 1)
    sub = subs.add_parser(
        "trace-terrain",
        help="Generate synthetic terrain collapse traces (Milestone 1)",
    )
    sub.add_argument("--out-dir", type=Path, default=Path("data/datasets/collapse_traces"))
    sub.add_argument("--blender", type=str, default=BLENDER_EXE)
    sub.add_argument("--n", type=int, default=200)
    sub.add_argument("--seed", type=int, default=0)
    sub.add_argument("--max-steps", type=int, default=64)
    sub.add_argument("--target-verts", type=int, default=16)
    sub.add_argument("--skip-existing", action="store_true")

    # train-policy (Milestone 1)
    sub = subs.add_parser(
        "train-policy",
        help="Train compact policy transformer (Milestone 1)",
    )
    sub.add_argument("--policy-config", type=Path, default=Path("config.policy_m3_quick_traces.yaml"))
    sub.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    sub.add_argument("--max-steps", type=int, default=2000)
    sub.add_argument("--resume", action="store_true")
    sub.add_argument("--compile", action="store_true")

    # rollout-policy (Milestone 1)
    sub = subs.add_parser(
        "rollout-policy",
        help="Run closed-loop policy rollout in Blender (Milestone 1)",
    )
    sub.add_argument("--ckpt", type=Path, required=True)
    sub.add_argument("--out-dir", type=Path, required=True)
    sub.add_argument("--steps", type=int, default=64)
    sub.add_argument("--seed", type=int, default=0)
    sub.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    sub.add_argument("--blender", type=str, default=BLENDER_EXE)
    sub.add_argument("--prompt", type=str, default=None)
    sub.add_argument("--goal-vertices", type=int, default=None)
    sub.add_argument("--goal-symmetry", type=float, default=None)
    sub.add_argument("--deterministic", action="store_true")
    sub.add_argument("--temperature", type=float, default=1.0)
    sub.add_argument("--top-k", type=int, default=0)
    sub.add_argument("--low-poly-bias", action="store_true")

    # eval
    sub = subs.add_parser("eval", help="Run geometric evaluation on checkpoint")
    sub.add_argument("--checkpoint",
                     default="checkpoints/unified/latest.pt",
                     help="Path to model checkpoint")
    sub.add_argument("--max-faces", type=int, default=512,
                     help="Max faces per generation")
    sub.add_argument("--temperature", type=float, default=0.7)
    sub.add_argument("--device", default="auto",
                     choices=["auto", "cuda", "mps", "cpu"])

    # clean
    sub = subs.add_parser(
        "clean",
        help="Clean training artifacts (dry-run by default)",
        description=(
            "Safely clean artifacts so a new training run starts fresh. "
            "Default is DRY-RUN; use --apply to perform changes. "
            "By default it archives into _trash/ instead of deleting; add --rm for permanent deletion."
        ),
    )
    sub.add_argument("--apply", action="store_true",
                     help="Perform cleanup actions (default: dry-run)")
    sub.add_argument("--rm", action="store_true",
                     help="Permanently delete instead of archiving")
    sub.add_argument("--archive-dir", default=None,
                     help="Archive directory (default: _trash/clean_<timestamp>)")
    sub.add_argument("--all", action="store_true",
                     help="Clean all supported artifact types")
    sub.add_argument("--checkpoints", action="store_true",
                     help="Clean checkpoints/ (or a specific run via --run)")
    sub.add_argument("--run", default=None,
                     help="Checkpoint run name under checkpoints/ (e.g. unified). Use 'all' to target entire checkpoints/")
    sub.add_argument("--cache", action="store_true",
                     help="Clean data/processed/.mesh_cache* training cache")
    sub.add_argument("--wandb", action="store_true",
                     help="Clean local W&B logs (wandb/, .wandb/, runs/)")
    sub.add_argument("--eval", action="store_true",
                     help="Clean eval outputs (data/eval/results.jsonl, scaling_curve_*.json)")
    sub.add_argument("--renders", action="store_true",
                     help="Clean rendered images (data/renders)")
    sub.add_argument("--feedback", action="store_true",
                     help="Clean RLHF feedback buffer (data/feedback/feedback_buffer.jsonl)")
    sub.add_argument("--pycache", action="store_true",
                     help="Delete Python caches (__pycache__/, .pytest_cache/, coverage)")

    # data-audit
    sub = subs.add_parser("data-audit",
                          help="Audit training data diversity and quality")
    sub.add_argument("--detailed", action="store_true",
                     help="Show per-label breakdown")

    # data-quality (comprehensive quality report)
    sub = subs.add_parser("data-quality",
                          help="Full data quality report: source attribution, entropy, label cap, etc.")
    sub.add_argument("--top", type=int, default=15, help="Top N frequent labels to show")
    sub.add_argument("--cap", type=int, default=100, help="Label frequency cap threshold")
    sub.add_argument("--fix-in-place", action="store_true",
                     help="Apply cap and remove empty labels directly")

    # vl-relabel (Qwen VL relabeling)
    sub = subs.add_parser("vl-relabel",
                          help="Relabel cache items using rendered images + Qwen VL")
    sub.add_argument("--max", type=int, default=None,
                     help="Maximum number of items to relabel")
    sub.add_argument("--model", default="qwen2.5vl:32b",
                     help="Ollama model name (default: qwen2.5vl:32b)")
    sub.add_argument("--skip-existing", action="store_true",
                     help="Skip items that already have a vl_label in manifest")
    sub.add_argument("--dry-run", action="store_true",
                     help="Print what would be relabeled without writing changes")
    sub.add_argument("--timeout", type=int, default=60,
                     help="Per-item Ollama timeout in seconds (default: 60)")

    # serve
    sub = subs.add_parser("serve", help="Start inference server")
    sub.add_argument("--checkpoint", help="Path to model checkpoint")
    sub.add_argument("--port", type=int, default=8420)
    sub.add_argument("--device", default="auto",
                     choices=["auto", "cuda", "mps", "cpu"])
    sub.add_argument("--watch-interval", type=float, default=30.0)

    # pipeline
    sub = subs.add_parser("pipeline", help="Full pipeline: scrape→train")
    sub.add_argument("--sources", nargs="*",
                     choices=["blender", "blendswap", "smutbase",
                              "open3dlab", "github", "objaverse",
                              "thingiverse", "sketchfab",
                              "wikimedia", "terminology", "all"])
    sub.add_argument("--batch-size", type=int)
    sub.add_argument("--name", help="Run name")
    sub.add_argument("--skip-scrape", action="store_true")
    sub.add_argument("--skip-render", action="store_true",
                     help="Skip the render step (use synthetic renders only)")
    sub.add_argument("--skip-train", action="store_true")

    # lambda commands
    for cmd_name, help_text in [
        ("lambda-setup", "Set up cloud GPU instance"),
        ("lambda-train", "Start training on cloud GPU"),
        ("lambda-sync",  "Sync checkpoints from cloud"),
        ("lambda-kill",  "Kill training on cloud GPU"),
    ]:
        sub = subs.add_parser(cmd_name, help=help_text)
        sub.add_argument("host", help="SSH target (e.g. user@gpu-ip)")
        if cmd_name == "lambda-sync":
            sub.add_argument("--continuous", action="store_true",
                             help="Keep syncing in a loop")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        print(f"\n{C.info('Try: python run.py status')}")
        return

    commands = {
        "status":       cmd_status,
        "data":         cmd_data,
        "scrape":       cmd_scrape,
        "extract":      cmd_extract,
        "build":        cmd_build,
        "render":       cmd_render,
        "train":        cmd_train,
        "trace-cache":  cmd_trace_cache,
        "trace-terrain": cmd_trace_terrain,
        "train-policy": cmd_train_policy,
        "rollout-policy": cmd_rollout_policy,
        "clean":        cmd_clean,
        "eval":         cmd_eval,
        "data-audit":   cmd_data_audit,
        "data-quality": cmd_data_quality,
        "vl-relabel":   cmd_vl_relabel,
        "serve":        cmd_serve,
        "pipeline":     cmd_pipeline,
        "lambda-setup": cmd_lambda_setup,
        "lambda-train": cmd_lambda_train,
        "lambda-sync":  cmd_lambda_sync,
        "lambda-kill":  cmd_lambda_kill,
    }

    cmd = commands.get(args.command)
    if cmd:
        cmd(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
