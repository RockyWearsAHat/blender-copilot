"""Quick test: compare tokenizer behavior with 8192 vs 1024 vocab."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.mesh_tokenizer import MeshTokenizer
import numpy as np
from collections import Counter

cube_v = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]
cube_f = [[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],[2,6,7],[2,7,3],[0,3,7],[0,7,4],[1,5,6],[1,6,2]]

for vs in [8192, 1024]:
    tok = MeshTokenizer(vocab_size=vs)
    tokens = tok.encode_mesh(cube_v, cube_f)
    coord_tokens = [t for t in tokens if t >= tok.SPECIAL_TOKENS]
    unique = set(coord_tokens)
    c = Counter(coord_tokens)
    print(f"\nvocab_size={vs}: {len(tokens)} tokens, {len(unique)} unique coord tokens")
    print(f"  coord_bins={tok.coord_bins}, range per bin={2.0/tok.coord_bins:.6f}")
    print(f"  Token distribution: {dict(c)}")

print("\n--- Rotated cube (45 deg around Z) ---")
angle = np.pi / 4
R = np.array([[np.cos(angle), -np.sin(angle), 0],
              [np.sin(angle), np.cos(angle), 0],
              [0, 0, 1]])
rotated_v = (np.array(cube_v) @ R.T)
mx = np.abs(rotated_v).max()
rotated_v = (rotated_v / mx).tolist()

for vs in [8192, 1024]:
    tok = MeshTokenizer(vocab_size=vs)
    tokens = tok.encode_mesh(rotated_v, cube_f)
    coord_tokens = [t for t in tokens if t >= tok.SPECIAL_TOKENS]
    unique = set(coord_tokens)
    print(f"  vocab_size={vs}: {len(unique)} unique coord tokens out of {len(coord_tokens)} total")

print("\n--- Random rotation (full 3D) ---")
from training.train_unified import augment_vertices
rng = np.random.RandomState(42)
aug_v = augment_vertices(np.array(cube_v), rotate=True, jitter_std=0.002, rng=rng)
mx = np.abs(aug_v).max()
aug_v = (aug_v / mx).tolist()

for vs in [8192, 1024]:
    tok = MeshTokenizer(vocab_size=vs)
    tokens = tok.encode_mesh(aug_v, cube_f)
    coord_tokens = [t for t in tokens if t >= tok.SPECIAL_TOKENS]
    unique = set(coord_tokens)
    print(f"  vocab_size={vs}: {len(unique)} unique coord tokens out of {len(coord_tokens)} total")
