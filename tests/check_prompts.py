"""Check what prompts look like after enrichment."""
import sys, os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
os.chdir(root)

from processing.prompt_semantics import enrich_prompt_text

for p in ["a cube", "a sphere", "a donut", "low poly car"]:
    enriched = enrich_prompt_text(p)
    print(f"  {p!r:20s} -> {enriched!r}")
