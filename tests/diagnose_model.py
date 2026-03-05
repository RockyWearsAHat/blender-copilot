"""Direct model diagnostic - bypass the server to inspect raw token probabilities."""
import torch
import yaml
import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(root)
sys.path.insert(0, root)

config = yaml.safe_load(open("config.unified_m3_semantic_bootstrap_stable.yaml"))
from models.unified import UnifiedBlenderModel

model = UnifiedBlenderModel(config)

ckpt = torch.load(
    "checkpoints/unified_semantic_bootstrap/best.pt",
    map_location="cpu",
    weights_only=False,
)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

print(f"Model loaded: step={ckpt.get('step')}, loss={ckpt.get('loss')}")

from processing.mesh_tokenizer import MeshTokenizer

tok_cfg = config.get("tokenization", {})
tok = MeshTokenizer(
    vocab_size=tok_cfg.get("vocab_size", 8192),
    coord_range=tuple(tok_cfg.get("coordinate_range", [-1.0, 1.0])),
    max_faces=tok_cfg.get("max_faces", 2048),
)
print(f"BOS={tok.BOS}, EOS={tok.EOS}, vocab={tok.vocab_size}")

# Try to find the actual text tokenizer - same logic as inference/server.py
text_tok = None
from pathlib import Path
cp_dir = Path("checkpoints/unified_semantic_bootstrap")

# Try BPE first
bpe_search = [
    cp_dir / "bpe_tokenizer",
    Path("data/datasets/geometry/bpe_tokenizer"),
]
for bp in bpe_search:
    if bp.is_dir() and (bp / "tokenizer.model").exists():
        from processing.bpe_tokenizer import BPETokenizer
        text_tok = BPETokenizer.load(bp)
        print(f"Using BPE tokenizer: {bp}")
        break

# Fall back to legacy
if text_tok is None:
    for sp in [
        cp_dir / "text_tokenizer.json",
        Path("data/datasets/geometry/text_tokenizer.json"),
    ]:
        if sp.exists():
            from processing.text_tokenizer import TextTokenizer
            text_tok = TextTokenizer.load(sp)
            print(f"Using legacy text tokenizer: {sp}")
            break

# fallback: char-level
use_char_level = text_tok is None
if use_char_level:
    print("No text tokenizer found - using char-level fallback")

prompts = ["a cube", "a sphere", "low poly car", "a donut"]

for prompt in prompts:
    print(f"\n{'='*60}")
    print(f"PROMPT: {prompt!r}")

    if text_tok is not None:
        ids, mask = text_tok.encode_padded(prompt, max_length=192)
    else:
        ids = [ord(c) % 8000 for c in prompt[:192]]
        mask = [1] * len(ids)
        ids += [0] * (192 - len(ids))
        mask += [0] * (192 - len(mask))
    text_ids = torch.tensor([ids], dtype=torch.long)
    text_mask = torch.tensor([mask], dtype=torch.float)

    with torch.no_grad():
        text_seq, _ = model.text_encoder(text_ids, text_mask)

        # Generate tokens
        tokens = model.geometry_decoder.generate(
            text_seq,
            text_mask,
            max_tokens=128 * 9 + 2,
            temperature=0.9,
            top_k=0,
            top_p=1.0,
            cfg_scale=0.0,
        )
        token_list = tokens[0].cpu().tolist()
        print(f"Generated {len(token_list)} tokens: {token_list[:30]}")

        # Examine probability distribution at step 0 (after BOS)
        bos = torch.tensor([[tok.BOS]], dtype=torch.long)
        x = model.geometry_decoder.mesh_embed(bos)
        text_cond = model.geometry_decoder.text_proj(text_seq)

        for layer in model.geometry_decoder.layers:
            x = layer(x, text_cond, rope=model.geometry_decoder.rope)
        x = model.geometry_decoder.norm(x)
        logits = model.geometry_decoder.output_proj(x[:, 0, :])

        probs = torch.softmax(logits, dim=-1)
        top_vals, top_ids = torch.topk(probs, k=10)
        print("Top-10 probs after BOS:")
        for i in range(10):
            tid = top_ids[0, i].item()
            p = top_vals[0, i].item()
            label = ""
            if tid == tok.BOS:
                label = " (BOS)"
            elif tid == tok.EOS:
                label = " (EOS)"
            elif tid == 0:
                label = " (PAD)"
            print(f"  token {tid:5d}: prob={p:.6f}{label}")

        eos_prob = probs[0, tok.EOS].item()
        print(f"EOS prob: {eos_prob:.6f}")

        # Check logit statistics
        print(f"Logit stats: mean={logits.mean():.3f}, std={logits.std():.3f}, "
              f"min={logits.min():.3f}, max={logits.max():.3f}")

        # Check how many tokens have > 1% probability
        high_prob_count = (probs[0] > 0.01).sum().item()
        print(f"Tokens with >1% prob: {high_prob_count}")

        # Also try decoding the generated tokens
        try:
            vertices, faces = tok.decode_tokens(token_list)
            print(f"Decoded: {len(vertices)} verts, {len(faces)} faces")
        except Exception as e:
            print(f"Decode failed: {e}")
