"""Tokenizer roundtrip test + training data token distribution analysis."""
import json, sys, os
import numpy as np
from pathlib import Path
from collections import Counter

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
os.chdir(root)

from processing.mesh_tokenizer import MeshTokenizer

tok = MeshTokenizer(vocab_size=8192, coord_range=(-1.0, 1.0), max_faces=400)
print(f"BOS={tok.BOS}, EOS={tok.EOS}, PAD={tok.PAD}, SPECIAL_TOKENS={tok.SPECIAL_TOKENS}")

# 1. Test encode/decode roundtrip with known cube
print("\n=== CUBE ROUNDTRIP TEST ===")
cube_verts = [
    [-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
    [1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1],
]
# 12 triangular faces for a cube
cube_faces = [
    [0,2,6],[0,6,4],  # -X face
    [1,5,7],[1,7,3],  # +X face
    [0,1,3],[0,3,2],  # -Y face
    [4,6,7],[4,7,5],  # +Y face
    [0,4,5],[0,5,1],  # -Z face
    [2,3,7],[2,7,6],  # +Z face
]

tokens = tok.encode_mesh(cube_verts, cube_faces)
print(f"Cube encoded: {len(tokens)} tokens")
print(f"Token values: {tokens[:30]}")
print(f"Unique tokens: {len(set(tokens))}")
token_counter = Counter(tokens)
print(f"Token frequency: {token_counter.most_common(10)}")

# Check what values map to -1, 0, 1 in coordinate space
for val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    qval = tok.quantize_coord(val)
    dval = tok.dequantize_coord(qval)
    print(f"  coord {val:+.1f} -> token {qval:5d} -> decoded {dval:+.4f}")

# Decode back
verts_dec, faces_dec = tok.decode_tokens(tokens)
print(f"\nDecoded: {len(verts_dec)} verts, {len(faces_dec)} faces")
if verts_dec:
    v = np.array(verts_dec)
    print(f"Vert ranges: x=[{v[:,0].min():.3f},{v[:,0].max():.3f}] "
          f"y=[{v[:,1].min():.3f},{v[:,1].max():.3f}] "
          f"z=[{v[:,2].min():.3f},{v[:,2].max():.3f}]")

# 2. Analyze token distribution in actual training data
print("\n\n=== TRAINING DATA TOKEN DISTRIBUTION ===")
all_tokens = Counter()
sample_count = 0
total_token_count = 0

base = Path("data/processed/objaverse")
for f in sorted(base.glob("*.json"))[:200]:
    try:
        d = json.loads(f.read_text())
    except Exception:
        continue
    for obj in d.get("objects", []):
        mesh = obj.get("mesh", {})
        verts = mesh.get("vertices", [])
        faces = mesh.get("faces", [])
        if not verts or not faces or len(faces) < 2:
            continue
        if len(faces) > 400:
            continue  # skip objects that would need decimation
        try:
            from processing.mesh_ops import normalize_mesh
            verts_norm = normalize_mesh(verts, target_range=(-1.0, 1.0))
            tokens = tok.encode_mesh(verts_norm, faces)
        except Exception:
            continue
        # Count tokens (skip BOS/EOS)
        for t in tokens[1:-1]:
            all_tokens[t] += 1
            total_token_count += 1
        sample_count += 1
        if sample_count >= 100:
            break
    if sample_count >= 100:
        break

print(f"Analyzed {sample_count} meshes, {total_token_count} tokens total")
print(f"Unique token values used: {len(all_tokens)}")
print(f"\nTop 20 most common tokens:")
for token, count in all_tokens.most_common(20):
    pct = count / total_token_count * 100
    coord = tok.dequantize_coord(token) if token >= tok.SPECIAL_TOKENS else None
    label = f" (coord={coord:+.4f})" if coord is not None else " (SPECIAL)"
    print(f"  token {token:5d}: {count:6d} ({pct:5.2f}%){label}")

# Check the specific collapsed tokens the model produces
print(f"\nCollapsed tokens analysis:")
for t in [4, 4097, 8191]:
    count = all_tokens.get(t, 0)
    pct = count / total_token_count * 100 if total_token_count > 0 else 0
    coord = tok.dequantize_coord(t) if t >= tok.SPECIAL_TOKENS else None
    label_str = f" (coord={coord:+.4f})" if coord is not None else " (SPECIAL)"
    print(f"  token {t:5d}: {count:6d} hits ({pct:.2f}%){label_str}")

# Distribution statistics
if all_tokens:
    values = list(all_tokens.values())
    print(f"\nToken frequency stats:")
    print(f"  Min frequency: {min(values)}")
    print(f"  Max frequency: {max(values)}")
    print(f"  Mean frequency: {np.mean(values):.1f}")
    print(f"  Median frequency: {np.median(values):.1f}")

    # How many tokens cover 50% and 90% of the data
    sorted_counts = sorted(all_tokens.values(), reverse=True)
    cumsum = np.cumsum(sorted_counts)
    p50 = np.searchsorted(cumsum, total_token_count * 0.5) + 1
    p90 = np.searchsorted(cumsum, total_token_count * 0.9) + 1
    print(f"  Tokens covering 50% of data: {p50}")
    print(f"  Tokens covering 90% of data: {p90}")
