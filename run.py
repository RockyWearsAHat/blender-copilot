#!/usr/bin/env python3
"""BlenderModelTraining — unified CLI.

One command to rule them all. Instead of remembering 6 different module
paths and a dozen flags, just run:

    python run.py scrape          # Download training data
    python run.py extract         # Extract mesh data from downloads
    python run.py build           # Build training datasets
    python run.py train           # Train the model
    python run.py serve           # Start inference server
    python run.py pipeline        # Run scrape → extract → build → train
    python run.py status          # Show current progress

Each subcommand auto-detects device, paths, and config — zero manual input.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DATASET_DIR = DATA_DIR / "datasets"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CONFIG_PATH = PROJECT_ROOT / "config_test.yaml"  # Use test config by default
FULL_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Colors for terminal output
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
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(config_path: Path | None = None) -> dict:
    import yaml
    path = config_path or CONFIG_PATH
    if not path.exists():
        path = FULL_CONFIG_PATH
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
    """Detect the best available compute device."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return f"cuda ({name})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps (Apple Silicon)"
        return "cpu"
    except ImportError:
        return "cpu (torch not installed)"


# ── Status Command ────────────────────────────────────────────────────────

def cmd_status(args):
    """Show current progress of data collection and training."""
    print(f"\n{C.BOLD}{'═' * 60}{C.END}")
    print(f"{C.BOLD}  BlenderModelTraining — Status{C.END}")
    print(f"{C.BOLD}{'═' * 60}{C.END}\n")

    # Device
    device = detect_device()
    print(f"  {C.info(f'Device: {device}')}")
    print()

    # ── Raw Data ──
    print(f"  {C.BOLD}📁 Raw Data{C.END}")
    sources = [
        ("Objaverse (GLB)", RAW_DIR / "objaverse" / "models", "*.glb"),
        ("Blender Official", RAW_DIR / "blender_official" / "models", "*.blend"),
        ("YouTube Transcripts", RAW_DIR / "youtube", "*.json"),
    ]
    total_raw = 0
    for name, path, pattern in sources:
        n = count_files(path, pattern)
        total_raw += n
        sz = dir_size_mb(path)
        status = C.ok(f"{n:,} files ({sz:.0f} MB)") if n > 0 else C.warn("0 files")
        print(f"     {name:.<30s} {status}")
    print(f"     {'Total':.<30s} {total_raw:,} files")
    print()

    # ── Processed Data ──
    print(f"  {C.BOLD}⚙️  Processed Data{C.END}")
    proc_dirs = [
        ("Objaverse meshes", PROCESSED_DIR / "objaverse", "*.json"),
        ("Blender Official meshes", PROCESSED_DIR / "blender_official", "*.json"),
    ]
    total_proc = 0
    for name, path, pattern in proc_dirs:
        n = count_files(path, pattern)
        total_proc += n
        status = C.ok(f"{n:,} objects") if n > 0 else C.warn("not yet extracted")
        print(f"     {name:.<30s} {status}")
    print()

    # ── Datasets ──
    print(f"  {C.BOLD}📊 Training Datasets{C.END}")
    for split in ["train", "val", "test"]:
        path = DATASET_DIR / "geometry" / f"{split}.jsonl"
        if path.exists():
            n = sum(1 for _ in open(path))
            print(f"     {split:.<30s} {C.ok(f'{n:,} examples')}")
        else:
            print(f"     {split:.<30s} {C.warn('not built yet')}")
    print()

    # ── Checkpoints ──
    print(f"  {C.BOLD}🏋️  Checkpoints{C.END}")
    for cp_dir in sorted(CHECKPOINT_DIR.glob("*")) if CHECKPOINT_DIR.exists() else []:
        if cp_dir.is_dir():
            best = cp_dir / "best.pt"
            final = cp_dir / "final.pt"
            steps = list(cp_dir.glob("step_*.pt"))
            if best.exists() or final.exists():
                parts = []
                if best.exists():
                    sz = best.stat().st_size / (1024 * 1024)
                    parts.append(f"best.pt ({sz:.0f}MB)")
                if final.exists():
                    parts.append("final.pt")
                if steps:
                    parts.append(f"{len(steps)} step checkpoints")
                print(f"     {cp_dir.name:.<30s} {C.ok(', '.join(parts))}")
            else:
                print(f"     {cp_dir.name:.<30s} {C.warn('empty')}")
    if not CHECKPOINT_DIR.exists() or not list(CHECKPOINT_DIR.glob("*")):
        print(f"     {'(none)':.<30s} {C.warn('no training done yet')}")
    print()

    # ── Config ──
    print(f"  {C.BOLD}⚙️  Config{C.END}")
    if CONFIG_PATH.exists():
        cfg = load_config(CONFIG_PATH)
        geo = cfg.get("models", {}).get("geometry", {})
        tr = cfg.get("training", {})
        print(f"     Model: {geo.get('hidden_size', '?')}d, "
              f"{geo.get('num_layers', '?')} layers, "
              f"{geo.get('num_heads', '?')} heads")
        print(f"     Training: {tr.get('max_steps', '?')} steps, "
              f"batch={tr.get('batch_size', '?')}, "
              f"lr={tr.get('learning_rate', '?')}")

    print(f"\n{C.BOLD}{'═' * 60}{C.END}\n")


# ── Scrape Command ────────────────────────────────────────────────────────

def cmd_scrape(args):
    """Download training data from all sources."""
    print(f"\n{C.step(1, 'Scraping training data...')}\n")

    sources = args.sources or ["blender", "objaverse"]
    config = load_config(args.config)

    for source in sources:
        if source in ("blender", "blender_official"):
            print(f"\n{C.step('1a', 'Downloading Blender official demo files...')}")
            from scrapers.blender_official import download_blender_official
            output = Path(args.output or "data/raw/blender_official")
            output.mkdir(parents=True, exist_ok=True)
            download_blender_official(
                output,
                max_size_mb=args.max_size or 500,
                crawl=not args.no_crawl,
            )

        elif source == "objaverse":
            print(f"\n{C.step('1b', 'Downloading Objaverse models...')}")
            from scrapers.objaverse_scraper import download_objaverse_models
            output = Path(args.output or "data/raw/objaverse")
            output.mkdir(parents=True, exist_ok=True)
            max_models = args.max_models or config.get("scraping", {}).get(
                "objaverse", {}).get("max_models", 1000)
            download_objaverse_models(output, config, max_models=max_models)

        elif source == "youtube":
            print(f"\n{C.step('1c', 'Scraping YouTube transcripts...')}")
            try:
                from scrapers.youtube_scraper import main as yt_main
                sys.argv = ["", "--output", args.output or "data/raw/youtube"]
                yt_main()
            except Exception as e:
                print(C.err(f"YouTube scraper failed: {e}"))
                print(C.info("Install: pip install yt-dlp youtube-transcript-api"))

        else:
            print(C.warn(f"Unknown source: {source}"))

    print(f"\n{C.ok('Scraping complete!')}\n")


# ── Extract Command ───────────────────────────────────────────────────────

def cmd_extract(args):
    """Extract mesh data from downloaded files."""
    print(f"\n{C.step(2, 'Extracting mesh data...')}\n")

    config = load_config(args.config)

    # Process Objaverse GLB files with mesh_extractor (no Blender needed)
    objaverse_raw = RAW_DIR / "objaverse" / "models"
    objaverse_out = PROCESSED_DIR / "objaverse"
    if objaverse_raw.exists() and count_files(objaverse_raw, "*.glb") > 0:
        n = count_files(objaverse_raw, "*.glb")
        print(f"  {C.info(f'Processing {n} Objaverse GLB files...')}")
        from processing.mesh_extractor import process_directory
        process_directory(
            objaverse_raw, objaverse_out,
            metadata_dir=RAW_DIR / "objaverse" / "metadata",
            config=config,
        )
    else:
        print(f"  {C.warn('No Objaverse data found — run: python run.py scrape')}")

    # Process Blender Official .blend files with blend_extractor (needs Blender)
    blend_raw = RAW_DIR / "blender_official" / "models"
    blend_out = PROCESSED_DIR / "blender_official"
    if blend_raw.exists() and count_files(blend_raw, "*.blend") > 0:
        n = count_files(blend_raw, "*.blend")
        blender_exe = config.get("processing", {}).get("blender_executable", "blender")

        # Check if Blender is available
        blender_path = shutil.which(blender_exe)
        if blender_path:
            print(f"  {C.info(f'Processing {n} .blend files with Blender headless...')}")
            blend_out.mkdir(parents=True, exist_ok=True)

            for blend_file in sorted(blend_raw.glob("*.blend")):
                out_file = blend_out / f"{blend_file.stem}.json"
                if out_file.exists():
                    continue
                print(f"    Processing: {blend_file.name}")
                try:
                    subprocess.run([
                        blender_path, "--background", "--python",
                        str(PROJECT_ROOT / "processing" / "blend_extractor.py"),
                        "--", "--input", str(blend_file), "--output", str(blend_out),
                    ], capture_output=True, timeout=120)
                except subprocess.TimeoutExpired:
                    print(f"    {C.warn(f'Timeout: {blend_file.name}')}")
                except FileNotFoundError:
                    print(f"    {C.err('Blender not found')}")
                    break
        else:
            print(f"  {C.warn(f'Blender not found at {blender_exe!r} — skipping .blend extraction')}")
            print(f"  {C.info('Set processing.blender_executable in config.yaml to Blender path')}")
            print(f"  {C.info('Or install Blender: brew install --cask blender')}")
            # Fall back to mesh_extractor for any exported formats
            # Check if there are any glb/obj files alongside .blend files
            alt_files = list(blend_raw.glob("*.glb")) + list(blend_raw.glob("*.obj"))
            if alt_files:
                print(f"  {C.info(f'Found {len(alt_files)} exportable format files, processing those...')}")
                from processing.mesh_extractor import process_directory
                process_directory(blend_raw, blend_out, None, config)

    total = count_files(objaverse_out, "*.json") + count_files(blend_out, "*.json")
    print(f"\n{C.ok(f'Extraction complete! {total} processed files.')}\n")


# ── Build Command ─────────────────────────────────────────────────────────

def cmd_build(args):
    """Build training datasets from processed data."""
    print(f"\n{C.step(3, 'Building training datasets...')}\n")

    config = load_config(args.config)

    # Collect all processed directories
    input_dirs = []
    for d in PROCESSED_DIR.iterdir() if PROCESSED_DIR.exists() else []:
        if d.is_dir() and count_files(d, "*.json") > 0:
            input_dirs.append(d)
            print(f"  {C.info(f'Found {count_files(d, '*.json')} files in {d.name}/')}")

    if not input_dirs:
        print(C.err("No processed data found — run: python run.py extract"))
        return

    # Use the dataset builder
    from processing.dataset_builder import build_geometry_dataset
    from processing.mesh_tokenizer import MeshTokenizer

    tok_config = config.get("tokenization", {})
    tokenizer = MeshTokenizer(
        vocab_size=tok_config.get("vocab_size", 8192),
        coord_range=tuple(tok_config.get("coordinate_range", [-1.0, 1.0])),
        max_faces=tok_config.get("max_faces", 2048),
    )

    geo_dir = DATASET_DIR / "geometry"
    geo_dir.mkdir(parents=True, exist_ok=True)

    all_examples = []
    for input_dir in input_dirs:
        for json_file in sorted(input_dir.glob("*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                for obj in data.get("objects", []):
                    mesh = obj.get("mesh", {})
                    verts = mesh.get("vertices", [])
                    faces = mesh.get("faces", [])
                    if len(faces) < 12:
                        continue

                    tokens = tokenizer.encode_mesh(verts, faces)
                    if not tokens:
                        continue

                    # Generate text label
                    text = _generate_text_label(obj, data.get("metadata", {}))
                    all_examples.append({"text": text, "tokens": tokens})
            except Exception:
                continue

    if not all_examples:
        print(C.err("No valid examples found after tokenization"))
        return

    # Deduplicate
    seen = set()
    unique = []
    for ex in all_examples:
        key = tuple(ex["tokens"][:50])
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    print(f"  {C.info(f'{len(all_examples)} total → {len(unique)} unique examples')}")

    # Split 90/5/5
    import random
    random.seed(42)
    random.shuffle(unique)
    n = len(unique)
    n_val = max(1, int(n * 0.05))
    n_test = max(1, int(n * 0.05))
    n_train = n - n_val - n_test

    splits = {
        "train": unique[:n_train],
        "val": unique[n_train:n_train + n_val],
        "test": unique[n_train + n_val:],
    }

    for split_name, examples in splits.items():
        path = geo_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"  {C.ok(f'{split_name}: {len(examples)} examples → {path.name}')}")

    # Save tokenizer config
    tok_path = geo_dir / "tokenizer.json"
    with open(tok_path, "w") as f:
        json.dump(tok_config, f, indent=2)

    print(f"\n{C.ok(f'Dataset built! {len(unique)} total examples.')}\n")


def _generate_text_label(obj_data: dict, metadata: dict) -> str:
    """Generate descriptive text label for a mesh object."""
    parts = []

    # From metadata
    name = metadata.get("name", "") or obj_data.get("name", "")
    if name and name != "Object":
        parts.append(name)

    desc = metadata.get("description", "")
    if desc:
        parts.append(desc[:200])

    tags = metadata.get("tags", [])
    if tags:
        parts.append(", ".join(tags[:5]))

    category = metadata.get("category", "")
    if category:
        parts.append(f"category: {category}")

    # From mesh stats
    mesh = obj_data.get("mesh", {})
    nv = mesh.get("num_vertices", 0)
    nf = mesh.get("num_faces", 0)
    if nv and nf:
        parts.append(f"{nv} vertices, {nf} faces")

    if not parts:
        parts.append("3D object")

    return " | ".join(parts)


# ── Train Command ─────────────────────────────────────────────────────────

def cmd_train(args):
    """Train the geometry model."""
    import torch

    print(f"\n{C.step(4, 'Training geometry model...')}\n")

    config = load_config(args.config)
    geo_config = config.get("models", {}).get("geometry", {})
    train_config = config.get("training", {})

    # Check dataset exists
    train_file = DATASET_DIR / "geometry" / "train.jsonl"
    if not train_file.exists():
        print(C.err("No training data — run: python run.py build"))
        return

    n_train = sum(1 for _ in open(train_file))
    val_file = DATASET_DIR / "geometry" / "val.jsonl"
    n_val = sum(1 for _ in open(val_file)) if val_file.exists() else 0

    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        device_name = "Apple Silicon GPU"
    else:
        device = "cpu"
        device_name = "CPU"

    # Output directory
    run_name = args.name or f"geometry_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = CHECKPOINT_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Display training info
    max_steps = args.steps or train_config.get("max_steps", 2000)
    batch_size = args.batch_size or train_config.get("batch_size", 2)
    lr = train_config.get("learning_rate", 3e-4)

    print(f"  {C.BOLD}Training Configuration{C.END}")
    print(f"  {'─' * 40}")
    print(f"  Device:        {device_name}")
    print(f"  Model:         {geo_config.get('hidden_size', 256)}d, "
          f"{geo_config.get('num_layers', 6)} layers, "
          f"{geo_config.get('num_heads', 4)} heads")
    print(f"  Train data:    {n_train} examples")
    print(f"  Val data:      {n_val} examples")
    print(f"  Batch size:    {batch_size}")
    print(f"  Max steps:     {max_steps}")
    print(f"  Learning rate: {lr}")
    print(f"  Output:        {output_dir}")
    print(f"  {'─' * 40}")
    print()

    # Override config with CLI args
    if args.steps:
        train_config["max_steps"] = args.steps
    if args.batch_size:
        train_config["batch_size"] = args.batch_size

    # Import and run training
    from training.train_geometry import train as run_training

    class TrainArgs:
        pass
    t_args = TrainArgs()
    t_args.dataset = str(DATASET_DIR / "geometry")
    t_args.output = str(output_dir)
    t_args.config = str(args.config or CONFIG_PATH)
    t_args.resume = args.resume

    run_training(t_args)


# ── Serve Command ─────────────────────────────────────────────────────────

def cmd_serve(args):
    """Start the inference server for use in Blender."""
    print(f"\n{C.step(5, 'Starting inference server...')}\n")

    # Find best checkpoint
    checkpoint = args.checkpoint
    if not checkpoint:
        # Auto-find the best checkpoint
        for cp_dir in sorted(CHECKPOINT_DIR.glob("*"), reverse=True):
            best = cp_dir / "best.pt"
            if best.exists():
                checkpoint = str(best)
                break

    if not checkpoint:
        print(C.err("No checkpoint found — run: python run.py train"))
        return

    config_path = str(args.config or CONFIG_PATH)
    port = args.port or 8420

    print(f"  Checkpoint: {checkpoint}")
    print(f"  Config:     {config_path}")
    print(f"  Port:       {port}")
    print()
    print(f"  {C.BOLD}In Blender, connect to: http://127.0.0.1:{port}{C.END}")
    print(f"  {C.info('Test with: curl http://127.0.0.1:{port}/health')}")
    print()

    from inference.server import load_model, create_app
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model, tokenizer, device, text_tokenizer = load_model(checkpoint, config)
    app = create_app(model, tokenizer, device, config, text_tokenizer=text_tokenizer)

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=port)


# ── Pipeline Command ──────────────────────────────────────────────────────

def cmd_pipeline(args):
    """Run the full pipeline: scrape → extract → build → train."""
    print(f"\n{C.BOLD}{'═' * 60}{C.END}")
    print(f"{C.BOLD}  BlenderModelTraining — Full Pipeline{C.END}")
    print(f"{C.BOLD}{'═' * 60}{C.END}\n")

    # Step 1: Scrape
    if not args.skip_scrape:
        args.sources = args.sources or ["blender", "objaverse"]
        args.output = None
        args.max_size = 500
        args.max_models = args.max_models or 1000
        args.no_crawl = False
        cmd_scrape(args)

    # Step 2: Extract
    cmd_extract(args)

    # Step 3: Build dataset
    cmd_build(args)

    # Step 4: Train
    if not args.skip_train:
        args.steps = args.steps or 2000
        args.batch_size = args.batch_size or 2
        args.name = args.name or None
        args.resume = None
        cmd_train(args)

    print(f"\n{C.BOLD}{'═' * 60}{C.END}")
    print(f"{C.ok('Pipeline complete!')}")
    print(f"\n  Next: Start serving with:  python run.py serve")
    print(f"  Then in Blender, connect to http://127.0.0.1:8420")
    print(f"{C.BOLD}{'═' * 60}{C.END}\n")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BlenderModelTraining — train 3D models from Blender data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py status                    Show current progress
  python run.py scrape                    Download data (Blender + Objaverse)
  python run.py scrape --sources youtube  Scrape YouTube transcripts
  python run.py extract                   Extract mesh data
  python run.py build                     Build training dataset
  python run.py train                     Train the model
  python run.py train --steps 5000        Train for more steps
  python run.py serve                     Start inference server
  python run.py pipeline                  Run everything start to finish
  python run.py pipeline --skip-scrape    Skip download, re-extract + retrain
        """,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--config", type=Path, default=None,
                        help=f"Config file (default: {CONFIG_PATH.name})")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # status
    sub = subparsers.add_parser("status", help="Show progress dashboard")

    # scrape
    sub = subparsers.add_parser("scrape", help="Download training data")
    sub.add_argument("--sources", nargs="*",
                     choices=["blender", "objaverse", "youtube"],
                     help="Data sources (default: blender objaverse)")
    sub.add_argument("--output", help="Output directory")
    sub.add_argument("--max-size", type=float, default=500,
                     help="Max file size MB (default: 500)")
    sub.add_argument("--max-models", type=int, default=1000,
                     help="Max Objaverse models (default: 1000)")
    sub.add_argument("--no-crawl", action="store_true",
                     help="Don't crawl directory listings")

    # extract
    sub = subparsers.add_parser("extract", help="Extract mesh data from downloads")

    # build
    sub = subparsers.add_parser("build", help="Build training datasets")

    # train
    sub = subparsers.add_parser("train", help="Train the geometry model")
    sub.add_argument("--steps", type=int, help="Override max training steps")
    sub.add_argument("--batch-size", type=int, help="Override batch size")
    sub.add_argument("--name", help="Run name for checkpoints")
    sub.add_argument("--resume", help="Resume from checkpoint path")

    # serve
    sub = subparsers.add_parser("serve", help="Start inference server")
    sub.add_argument("--checkpoint", help="Path to model checkpoint")
    sub.add_argument("--port", type=int, default=8420)

    # pipeline
    sub = subparsers.add_parser("pipeline", help="Run full pipeline (scrape→train)")
    sub.add_argument("--sources", nargs="*",
                     choices=["blender", "objaverse", "youtube"])
    sub.add_argument("--max-models", type=int, default=1000)
    sub.add_argument("--steps", type=int, help="Training steps")
    sub.add_argument("--batch-size", type=int, help="Training batch size")
    sub.add_argument("--name", help="Run name")
    sub.add_argument("--skip-scrape", action="store_true",
                     help="Skip download step")
    sub.add_argument("--skip-train", action="store_true",
                     help="Skip training step")

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Default config
    if args.config is None:
        args.config = CONFIG_PATH if CONFIG_PATH.exists() else FULL_CONFIG_PATH

    if not args.command:
        parser.print_help()
        print(f"\n{C.info('Try: python run.py status')}")
        return

    commands = {
        "status": cmd_status,
        "scrape": cmd_scrape,
        "extract": cmd_extract,
        "build": cmd_build,
        "train": cmd_train,
        "serve": cmd_serve,
        "pipeline": cmd_pipeline,
    }

    cmd = commands.get(args.command)
    if cmd:
        cmd(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
