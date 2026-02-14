# BlenderModelTraining

Fine-tuned local models for Blender 3D generation — trained on real community
.blend files, not general-purpose LLMs.

## The Problem

General-purpose LLMs (GPT-4, Claude, etc.) cannot do 3D modeling:

- They can't spatially reason about vertex coordinates
- They generate Python scripts, not actual geometry
- They're expensive (API costs per generation)
- They're slow (10+ round-trips per model)

## The Solution

Train **specialized small models** on thousands of real Blender projects so they
learn actual 3D spatial relationships — the same way Stable Diffusion learned
pixels from images.

### Key Differences from the LLM Approach

| Aspect            | LLM Approach (old)         | Trained Models (new)        |
| ----------------- | -------------------------- | --------------------------- |
| Output            | Python scripts             | Direct mesh/material data   |
| Spatial reasoning | None (hallucinated coords) | Learned from real meshes    |
| Cost              | $0.05-0.50 per generation  | Free (local inference)      |
| Speed             | 10-30s (API round-trips)   | 1-5s (local GPU)            |
| Quality           | Blobby, unrecognizable     | Trained on real artist work |

## Architecture

### Specialized Model Pipeline

```
User Prompt
    │
    ├─► [Geometry Model]  → mesh vertices, faces, edges
    │       Trained on: .blend mesh data + text descriptions
    │       Output: structured mesh representation
    │
    ├─► [Material Model]  → shader node trees, textures
    │       Trained on: Blender material node graphs
    │       Output: node tree definition
    │
    ├─► [Modifier Model]  → modifier stacks, weights
    │       Trained on: modifier configurations from real projects
    │       Output: ordered modifier list with parameters
    │
    ├─► [Animation Model] → keyframes, drivers, constraints
    │       Trained on: animation data from real projects
    │       Output: keyframe sequences
    │
    └─► [Blender Injector] → creates objects directly in scene
            No scripts — direct bpy.data manipulation
```

### Data Pipeline

```
Scrape Sources              Process .blend Files         Train Models
─────────────             ─────────────────────       ──────────────
BlendSwap          ─►     Blender headless     ─►     Geometry Model
GitHub repos       ─►     Extract meshes       ─►     Material Model
Blender Artists    ─►     Extract materials    ─►     Modifier Model
YouTube tutorials  ─►     Extract modifiers    ─►     Animation Model
Documentation      ─►     Generate text labels
Free 3D sites      ─►     Quality filtering
```

### Output Format (not Python scripts)

The models output a structured representation that gets directly injected:

```json
{
  "objects": [
    {
      "name": "Car Body",
      "mesh": {
        "vertices": [[x,y,z], ...],
        "faces": [[v1,v2,v3,v4], ...],
        "normals": [[nx,ny,nz], ...]
      },
      "modifiers": [
        {"type": "MIRROR", "axis": "Y", "clipping": true},
        {"type": "SUBSURF", "levels": 2, "render_levels": 3}
      ],
      "materials": [
        {
          "name": "Matte Black",
          "type": "PBR",
          "base_color": [0.02, 0.02, 0.02],
          "roughness": 0.4,
          "metallic": 0.3
        }
      ],
      "transforms": {
        "location": [0, 0, 0.5],
        "rotation": [0, 0, 0],
        "scale": [1, 1, 1]
      }
    }
  ]
}
```

The Blender addon's injector creates real objects from this data — no scripts
involved, just direct data population via `bpy.data`.

## Project Structure

```
BlenderModelTraining/
├── README.md
├── requirements.txt
├── config.yaml                  # Global settings
│
├── scrapers/                    # Data collection
│   ├── blendswap_scraper.py     # Scrape free .blend files
│   ├── github_scraper.py        # Find .blend files on GitHub
│   ├── youtube_scraper.py       # Tutorial transcripts + metadata
│   ├── blender_artists_scraper.py
│   └── utils.py                 # Common scraping utilities
│
├── processing/                  # .blend → training data
│   ├── blend_extractor.py       # Headless Blender mesh/material extraction
│   ├── mesh_tokenizer.py        # Convert meshes to model-friendly tokens
│   ├── material_extractor.py    # Extract node trees as structured data
│   ├── labeler.py               # Auto-label extracted data with descriptions
│   ├── quality_filter.py        # Filter low-quality / broken files
│   └── dataset_builder.py       # Assemble final training datasets
│
├── models/                      # Model definitions
│   ├── geometry/                # Mesh generation model
│   │   ├── model.py
│   │   ├── tokenizer.py         # Mesh face tokenization (MeshGPT-style)
│   │   └── config.py
│   ├── materials/               # Material/shader model
│   │   ├── model.py
│   │   └── config.py
│   ├── modifiers/               # Modifier stack model
│   │   ├── model.py
│   │   └── config.py
│   └── animation/               # Animation model
│       ├── model.py
│       └── config.py
│
├── training/                    # Training scripts
│   ├── train_geometry.py
│   ├── train_materials.py
│   ├── train_modifiers.py
│   ├── train_animation.py
│   └── evaluate.py              # Evaluation metrics
│
├── inference/                   # Local inference
│   ├── server.py                # Local inference server (FastAPI)
│   ├── geometry_inference.py
│   ├── material_inference.py
│   └── blender_injector.py      # Inject model output into Blender scene
│
├── data/                        # Downloaded/processed data (gitignored)
│   ├── raw/                     # Downloaded .blend files
│   ├── processed/               # Extracted training data
│   └── datasets/                # Final training datasets
│
└── scripts/                     # Utility scripts
    ├── download_all.sh          # Run all scrapers
    ├── process_all.sh           # Run all processing
    └── setup_env.sh             # Environment setup
```

## Quickstart

```bash
# 1. Setup
cd BlenderModelTraining
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Scrape training data
python -m scrapers.blendswap_scraper --output data/raw/blendswap
python -m scrapers.github_scraper --output data/raw/github

# 3. Process .blend files (requires Blender installed)
blender --background --python processing/blend_extractor.py -- --input data/raw --output data/processed

# 4. Build datasets
python -m processing.dataset_builder --input data/processed --output data/datasets

# 5. Train (start with geometry model)
python -m training.train_geometry --dataset data/datasets/geometry --output models/geometry/checkpoints

# 6. Run local inference server
python -m inference.server --model models/geometry/checkpoints/best.pt --port 8420
```

## Training Data Sources

| Source             | Type                 | Est. Files | License                     |
| ------------------ | -------------------- | ---------- | --------------------------- |
| BlendSwap          | .blend files         | 10,000+    | CC-BY, CC-0                 |
| GitHub             | .blend in repos      | 50,000+    | Various (filter by license) |
| Blender Artists    | Forum attachments    | 5,000+     | CC-BY                       |
| Free3D/Sketchfab   | Downloadable models  | 20,000+    | CC-0, CC-BY                 |
| Blender Demo Files | Official demos       | 50+        | CC-0                        |
| YouTube            | Tutorial transcripts | 100,000+   | Fair use (text only)        |

## Model Details

### Geometry Model

- **Architecture**: Transformer-based autoregressive mesh generation (MeshGPT-inspired)
- **Input**: Text description + optional reference image
- **Output**: Tokenized mesh faces (vertex coordinates quantized to vocabulary)
- **Training**: ~10K mesh examples with text labels
- **Target size**: 1-3B parameters (runs on consumer GPU)

### Material Model

- **Architecture**: Small transformer / decision tree
- **Input**: Text description + mesh context
- **Output**: Blender shader node graph as structured data
- **Training**: Extracted node trees from .blend files
- **Target size**: 100M-500M parameters

### Modifier Model

- **Architecture**: Sequence classifier
- **Input**: Text description + mesh properties
- **Output**: Ordered modifier stack with parameters
- **Training**: Modifier configurations from real projects
- **Target size**: 50M-200M parameters

## Hardware Requirements

### Training

- 1× NVIDIA GPU with 24GB+ VRAM (RTX 4090, A6000, etc.)
- 64GB+ RAM
- 500GB+ storage for training data
- Training time: 2-7 days per model

### Inference (end user)

- Any GPU with 8GB+ VRAM (RTX 3060+, M1/M2 Mac)
- OR CPU-only mode (slower, ~10-30s per generation)
- ~2-4GB model files per specialist model

## Roadmap

### Phase 1: Data Collection (Weeks 1-3)

- [ ] Build and test all scrapers
- [ ] Download 10,000+ .blend files
- [ ] Set up quality filtering pipeline

### Phase 2: Data Processing (Weeks 3-5)

- [ ] Extract meshes, materials, modifiers from .blend files
- [ ] Build mesh tokenizer
- [ ] Auto-label dataset with text descriptions
- [ ] Quality filter and deduplicate

### Phase 3: Geometry Model (Weeks 5-10)

- [ ] Implement mesh tokenizer (MeshGPT-style)
- [ ] Train geometry model on extracted meshes
- [ ] Evaluate: IoU, Chamfer distance, visual quality
- [ ] Optimize for consumer GPU inference

### Phase 4: Material Model (Weeks 8-12)

- [ ] Extract and serialize node trees
- [ ] Train material prediction model
- [ ] Integrate with geometry model output

### Phase 5: Addon Integration (Weeks 10-14)

- [ ] Build local inference server
- [ ] Build Blender injector (model output → scene objects)
- [ ] Replace API calls with local model calls
- [ ] Test end-to-end pipeline

### Phase 6: Additional Models (Weeks 12+)

- [ ] Modifier stack prediction
- [ ] Animation/keyframe generation
- [ ] UV unwrapping optimization
