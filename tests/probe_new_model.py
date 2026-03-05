"""Quick probe: check token diversity of new 1024-vocab model."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import yaml

config_path = Path("config.retrain_v1024_focal.yaml")
ckpt_path = Path("checkpoints/retrain_v1024_focal/best.pt")

if not ckpt_path.exists():
    ckpt_path = Path("checkpoints/retrain_v1024_focal/latest.pt")
if not ckpt_path.exists():
    print("No checkpoint found yet!")
    sys.exit(1)

config = yaml.safe_load(config_path.read_text())

from models.unified import UnifiedBlenderModel
model = UnifiedBlenderModel(config)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(state, strict=False)
model.eval()

step = ckpt.get("step", "?")
loss = ckpt.get("loss", "?")
print(f"Checkpoint: {ckpt_path.name}, step={step}, loss={loss}")
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# Simple char-level text encoding (fallback when no tokenizer file)
text = "a cube"
max_text_len = config["unified"].get("text_max_length", 192)
ids = [ord(c) % 8000 for c in text[:max_text_len]]
mask = [1] * len(ids)
ids += [0] * (max_text_len - len(ids))
mask += [0] * (max_text_len - len(mask))

text_ids = torch.tensor([ids])
text_mask = torch.tensor([mask])

# Start with BOS token and generate a few tokens
from processing.mesh_tokenizer import MeshTokenizer
tok_cfg = config.get("tokenization", {})
mt = MeshTokenizer(vocab_size=tok_cfg.get("vocab_size", 1024))

input_tok = torch.tensor([[mt.BOS]])
with torch.no_grad():
    logits = model.forward_geometry(text_ids, text_mask, input_tok)

# Check distribution of first predicted token
probs = torch.softmax(logits[0, -1], dim=-1)
top_vals, top_ids = probs.topk(20)

print(f"\nTop 20 predicted tokens after BOS (prompt='a cube'):")
print(f"{'Token':>6} {'Prob':>8} {'Coord':>8}")
print("-" * 26)
for val, idx in zip(top_vals, top_ids):
    idx_int = idx.item()
    if idx_int >= mt.SPECIAL_TOKENS:
        coord = mt.dequantize_token(idx_int)
        coord_str = f"{coord:.4f}"
    else:
        names = {0: "PAD", 1: "BOS", 2: "EOS", 3: "SEP"}
        coord_str = names.get(idx_int, "?")
    print(f"{idx_int:>6} {val.item():>8.4f} {coord_str:>8}")

# Count how many tokens have >0.1% probability
above_01 = (probs > 0.001).sum().item()
above_1 = (probs > 0.01).sum().item()
print(f"\nTokens with >0.1% prob: {above_01} / {mt.vocab_size}")
print(f"Tokens with >1.0% prob: {above_1} / {mt.vocab_size}")

# Compare with old model stat: only 4 tokens had >1% prob
if above_1 > 10:
    print("\n** HEALTHY: model is spreading probability across many tokens **")
elif above_1 > 4:
    print("\n** IMPROVING: more diversity than the collapsed model (was 4) **")
else:
    print("\n** WARNING: still very concentrated, may need more training **")
