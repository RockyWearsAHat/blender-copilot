#!/usr/bin/env python3
"""Decompress all BlendSwap files (ZIP/zstd) into a flat output directory."""
import os
import sys
import zipfile
import subprocess
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "blendswap"
OUT_DIR = Path("/tmp/blendswap_unzip")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_blends = sorted(RAW_DIR.rglob("*.blend"))
    print(f"Found {len(all_blends)} .blend files in {RAW_DIR}")
    
    extracted = 0
    skipped = 0
    failed = 0
    
    for bf in all_blends:
        bid = bf.stem
        subdir = bf.parent.name
        
        # Check file type
        result = subprocess.run(["file", "--brief", str(bf)], capture_output=True, text=True)
        ftype = result.stdout.strip().lower()
        
        if "zip archive" in ftype:
            try:
                with zipfile.ZipFile(bf, 'r') as zf:
                    blend_files = [n for n in zf.namelist() if n.endswith('.blend') and not n.startswith('__MACOSX')]
                    if not blend_files:
                        print(f"  SKIP {bid} ({subdir}): ZIP has no .blend files inside: {zf.namelist()[:5]}")
                        skipped += 1
                        continue
                    for blend_name in blend_files:
                        safe_name = f"{bid}_{subdir}_{Path(blend_name).stem}.blend"
                        out_path = OUT_DIR / safe_name
                        if out_path.exists():
                            skipped += 1
                            continue
                        with zf.open(blend_name) as src, open(out_path, 'wb') as dst:
                            dst.write(src.read())
                        extracted += 1
                        print(f"  ZIP {bid} ({subdir}): {blend_name} -> {safe_name}")
            except zipfile.BadZipFile:
                print(f"  FAIL {bid} ({subdir}): Bad ZIP file")
                failed += 1
            except Exception as e:
                print(f"  FAIL {bid} ({subdir}): {e}")
                failed += 1
                
        elif "zstandard" in ftype:
            safe_name = f"{bid}_{subdir}.blend"
            out_path = OUT_DIR / safe_name
            if out_path.exists():
                skipped += 1
                continue
            try:
                subprocess.run(["zstd", "-d", str(bf), "-o", str(out_path)], 
                             capture_output=True, check=True)
                extracted += 1
                print(f"  ZSTD {bid} ({subdir}): -> {safe_name}")
            except Exception as e:
                print(f"  FAIL {bid} ({subdir}): zstd decompress failed: {e}")
                failed += 1
                
        elif "blender" in ftype or "data" in ftype:
            # Already a raw .blend file
            safe_name = f"{bid}_{subdir}.blend"
            out_path = OUT_DIR / safe_name
            if out_path.exists():
                skipped += 1
                continue
            # Copy directly
            import shutil
            shutil.copy2(bf, out_path)
            extracted += 1
            print(f"  COPY {bid} ({subdir}): raw .blend -> {safe_name}")
        else:
            print(f"  UNKNOWN {bid} ({subdir}): {ftype}")
            failed += 1
    
    print(f"\nDone: {extracted} extracted, {skipped} skipped, {failed} failed")
    print(f"Output: {OUT_DIR}")
    out_count = len(list(OUT_DIR.glob("*.blend")))
    print(f"Total .blend files ready: {out_count}")

if __name__ == "__main__":
    main()
