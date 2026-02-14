"""Smoke test — verify everything works end-to-end."""
import torch
import yaml
from models.geometry.model import GeometryModel
from models.materials.model import MaterialModel
from models.modifiers.model import ModifierModel
from processing.mesh_tokenizer import MeshTokenizer

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# 1. Tokenizer roundtrip
print("=" * 50)
print("1. Tokenizer roundtrip test")
print("=" * 50)

tokenizer = MeshTokenizer(vocab_size=8192, coord_range=(-1.0, 1.0), max_faces=2048)

cube_verts = [
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
]
cube_faces = [
    [0, 1, 2], [0, 2, 3],
    [4, 6, 5], [4, 7, 6],
    [0, 4, 5], [0, 5, 1],
    [2, 6, 7], [2, 7, 3],
    [0, 3, 7], [0, 7, 4],
    [1, 5, 6], [1, 6, 2],
]

tokens = tokenizer.encode_mesh(cube_verts, cube_faces)
decoded_verts, decoded_faces = tokenizer.decode_tokens(tokens)
print(f"  Cube: {len(cube_faces)} faces -> {len(tokens)} tokens -> decoded {len(decoded_faces)} faces, {len(decoded_verts)} verts")
print("  ✅ Tokenizer OK")

# 2. Device detection
print()
print("=" * 50)
print("2. Device detection")
print("=" * 50)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"  Using: {device}")

# 3. Geometry model forward pass
print()
print("=" * 50)
print("3. Geometry model forward pass")
print("=" * 50)

model = GeometryModel(config).to(device)
print(f"  Parameters: {model.count_parameters():,}")

text_ids = torch.randint(0, 1000, (1, 64)).to(device)
text_mask = torch.ones(1, 64).to(device)
mesh_tokens = torch.randint(0, 8192, (1, 32)).to(device)

with torch.no_grad():
    logits = model(text_ids, text_mask, mesh_tokens)
    print(f"  Forward: (1, 32) -> logits {tuple(logits.shape)}")

# Quick generate test
with torch.no_grad():
    generated = model.generate(text_ids, text_mask, max_tokens=20, temperature=1.0)
    print(f"  Generate: produced {generated.shape[1]} tokens")
print("  ✅ Geometry model OK")

del model
torch.mps.empty_cache() if device.type == "mps" else None

# 4. Material model
print()
print("=" * 50)
print("4. Material model forward pass")
print("=" * 50)

mat_model = MaterialModel(config).to(device)
print(f"  Parameters: {mat_model.count_parameters():,}")

with torch.no_grad():
    mat_tokens = torch.randint(0, 4096, (1, 32)).to(device)
    mat_logits = mat_model(text_ids, text_mask, mat_tokens)
    print(f"  Forward: (1, 32) -> logits {tuple(mat_logits.shape)}")
print("  ✅ Material model OK")

del mat_model

# 5. Modifier model
print()
print("=" * 50)
print("5. Modifier model forward pass")
print("=" * 50)

mod_model = ModifierModel(config).to(device)
print(f"  Parameters: {mod_model.count_parameters():,}")

mesh_stats = torch.randn(1, 12).to(device)
with torch.no_grad():
    outputs = mod_model(text_ids, text_mask, mesh_stats)
    print(f"  Count logits: {tuple(outputs['count_logits'].shape)}")
    print(f"  Type logits:  {len(outputs['type_logits'])} heads")
print("  ✅ Modifier model OK")

print()
print("=" * 50)
print("ALL SMOKE TESTS PASSED ✅")
print("=" * 50)
