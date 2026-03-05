"""Extract structured training data from .blend files using Blender headless.

This script runs INSIDE Blender's Python environment (via --background mode).
It reads a .blend file and extracts:
- Mesh data (vertices, faces, normals) for each object
- Material/shader node tree definitions
- Modifier stacks
- Object hierarchy and transforms
- Scene metadata

The output is a structured JSON that represents the 3D content without
any Python scripts — just pure geometry and material data.

Usage:
    blender --background --python processing/blend_extractor.py -- \
        --input data/raw/blendswap/vehicles \
        --output data/processed/vehicles
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# This script runs inside Blender — bpy is always available
try:
    import bpy
    import bmesh
    from mathutils import Vector
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
    print("ERROR: This script must run inside Blender.")
    print("Usage: blender --background --python processing/blend_extractor.py -- --input <dir> --output <dir>")
    sys.exit(1)

# Full-resolution cap for packed image extraction (PNG, lossless)
IMAGE_MAX_SIZE = 4096
# Small preview thumbnail (JPEG, for quick loading)
IMAGE_THUMB_SIZE = 256
IMAGE_JPEG_QUALITY = 80


def _try_load_image_pixels(img) -> bool:
    """Force Blender to load image pixels into memory.

    Handles packed, external, UDIM, and sequence images.
    Returns True if pixels are available after loading.
    """
    # Already loaded?
    if img.size[0] > 0 and img.size[1] > 0:
        try:
            _ = img.pixels[0]
            return True
        except (IndexError, RuntimeError):
            pass

    # Try to reload from packed data
    if img.packed_file:
        try:
            img.unpack(method="USE_ORIGINAL")
        except Exception:
            pass
        try:
            img.reload()
        except Exception:
            pass
        if img.size[0] > 0:
            return True

    # Try resolving external filepath
    if img.filepath:
        try:
            abs_path = bpy.path.abspath(img.filepath)
            if os.path.isfile(abs_path):
                img.filepath = abs_path
                img.reload()
                if img.size[0] > 0:
                    return True
        except Exception:
            pass

    # Fallback: search nearby directories by filename
    # Handles cases where textures are in a textures/ subfolder next to the .blend
    try:
        blend_dir = Path(bpy.data.filepath).parent if bpy.data.filepath else None
        if blend_dir and img.filepath:
            fname_lower = Path(bpy.path.abspath(img.filepath)).name.lower()
            if fname_lower:
                _TEXTURE_EXTS = {".png",".jpg",".jpeg",".tga",".tiff",".tif",
                                  ".exr",".hdr",".bmp",".dds",".webp",".psd"}
                search_dirs = [
                    blend_dir,
                    blend_dir / "textures",
                    blend_dir / "tex",
                    blend_dir / "Textures",
                    blend_dir / "maps",
                    blend_dir.parent,
                    blend_dir.parent / "textures",
                    blend_dir.parent / "tex",
                ]
                for sd in search_dirs:
                    if not sd.is_dir():
                        continue
                    try:
                        for candidate in sd.iterdir():
                            if (candidate.is_file()
                                    and candidate.suffix.lower() in _TEXTURE_EXTS
                                    and candidate.name.lower() == fname_lower):
                                img.filepath = str(candidate)
                                img.reload()
                                if img.size[0] > 0:
                                    return True
                                break
                    except OSError:
                        pass
    except Exception:
        pass

    # UDIM tiles — try to resolve the first tile
    if "<UDIM>" in (img.filepath or "") or img.source == "TILED":
        try:
            # Replace <UDIM> with 1001 to find first tile
            test_path = bpy.path.abspath(img.filepath.replace("<UDIM>", "1001"))
            if os.path.isfile(test_path):
                img.filepath = test_path
                img.source = "FILE"
                img.reload()
                if img.size[0] > 0:
                    return True
        except Exception:
            pass

    # Last resort: try packing then reloading
    try:
        img.pack()
        img.reload()
        if img.size[0] > 0:
            return True
    except Exception:
        pass

    return img.size[0] > 0 and img.size[1] > 0


def _remap_missing_image_paths(blend_dir: "Path") -> int:
    """Remap image filepaths whose stored path doesn't resolve on disk.

    Builds a filename → path index from directories adjacent to the .blend
    (same dir, textures/, tex/, Textures/, maps/, and one level up variants),
    then updates any image whose ``bpy.path.abspath`` path is missing.

    Returns the number of images successfully remapped.
    """
    _TEXTURE_EXTS = {".png", ".jpg", ".jpeg", ".tga", ".tiff", ".tif",
                     ".exr", ".hdr", ".bmp", ".dds", ".webp", ".psd"}

    search_roots = [
        blend_dir,
        blend_dir / "textures",
        blend_dir / "tex",
        blend_dir / "Textures",
        blend_dir / "maps",
        blend_dir.parent,
        blend_dir.parent / "textures",
        blend_dir.parent / "tex",
        blend_dir.parent / "Textures",
    ]

    # Build lowercase-filename → list-of-paths index
    file_index: dict = {}
    for root in search_roots:
        if not root.is_dir():
            continue
        try:
            # One level deep is enough for nearly all packing conventions
            for p in root.iterdir():
                if p.is_file() and p.suffix.lower() in _TEXTURE_EXTS:
                    file_index.setdefault(p.name.lower(), []).append(p)
                elif p.is_dir():
                    # One extra level for nested texture folders
                    try:
                        for pp in p.iterdir():
                            if pp.is_file() and pp.suffix.lower() in _TEXTURE_EXTS:
                                file_index.setdefault(pp.name.lower(), []).append(pp)
                    except OSError:
                        pass
        except OSError:
            pass

    if not file_index:
        return 0

    remapped = 0
    for img in bpy.data.images:
        if img.packed_file:
            continue  # already packed
        if not img.filepath:
            continue
        try:
            abs_path = bpy.path.abspath(img.filepath)
        except Exception:
            continue
        if os.path.isfile(abs_path):
            continue  # already resolves fine

        fname_lower = Path(abs_path).name.lower()
        candidates = file_index.get(fname_lower, [])
        if not candidates:
            continue

        # Prefer the candidate with the shortest path (closest to blend_dir)
        best = min(candidates, key=lambda p: len(p.parts))
        img.filepath = str(best)
        remapped += 1

    if remapped:
        print(f"  Remapped {remapped} external texture path(s) to nearby files")
    return remapped


def _is_broken_image(img) -> bool:
    """Detect broken/placeholder images that should NOT be included in training.

    Blender shows missing textures as solid magenta (1,0,1). Other broken
    indicators: 0×0 size, completely uniform color (failed load producing
    solid black/white), or virtual images (Render Result, Viewer Node).

    Returns True if the image is broken and should be skipped.
    """
    # Virtual images — never meaningful pixel data
    if img.source in ("VIEWER",):
        return True
    if img.name in ("Render Result", "Viewer Node"):
        return True
    if img.name.startswith("Viewer Node"):
        return True

    # Zero-size images are useless
    w, h = img.size[0], img.size[1]
    if w == 0 or h == 0:
        return True

    # Sample pixels to detect solid-color placeholders
    try:
        px = img.pixels
        total = len(px)
        if total < 4:
            return True  # No pixel data

        # Sample a few pixels spread across the image (RGBA = 4 floats each)
        stride = max(1, total // (4 * 16))  # ~16 sample points
        samples = []
        for i in range(0, min(total, stride * 16 * 4), stride * 4):
            r, g, b = px[i], px[i + 1], px[i + 2]
            samples.append((r, g, b))

        if not samples:
            return True

        # Check for solid magenta (Blender's missing-texture pink)
        # Blender uses (1.0, 0.0, 1.0) or close to it
        all_pink = all(
            r > 0.8 and g < 0.2 and b > 0.8
            for r, g, b in samples
        )
        if all_pink:
            print(f"    Skipping broken image {img.name}: solid magenta (missing texture)")
            return True

        # Check for completely uniform color (all samples identical)
        # This catches solid-black or solid-white failed loads
        first = samples[0]
        all_same = all(
            abs(r - first[0]) < 0.01 and
            abs(g - first[1]) < 0.01 and
            abs(b - first[2]) < 0.01
            for r, g, b in samples
        )
        if all_same and len(samples) >= 4:
            # Solid color — could be legit (a flat color texture for a material)
            # Only skip if it's a known bad color: pure black with no alpha variation
            # or if the image has a broken filepath
            r0, g0, b0 = first
            is_black = r0 < 0.01 and g0 < 0.01 and b0 < 0.01
            has_broken_path = (
                img.filepath and
                not img.packed_file and
                not os.path.isfile(bpy.path.abspath(img.filepath))
            )
            if is_black and has_broken_path:
                print(f"    Skipping broken image {img.name}: solid black with missing file")
                return True

    except (RuntimeError, IndexError, AttributeError):
        # Can't read pixels at all — broken
        return True

    return False


def extract_image_data(img, thumb_size: int = IMAGE_THUMB_SIZE,
                       full_size: int = IMAGE_MAX_SIZE) -> dict | None:
    """Extract image metadata + pixel data as base64 PNG (full res) + JPEG thumbnail.

    Handles ALL image types:
    - Packed images (embedded in .blend)
    - External file references (resolved via bpy.path.abspath)
    - UDIM tiles (resolves first tile)
    - Generated/procedural textures

    Stores two representations:
    - ``image_data``: lossless PNG at original resolution (capped at full_size px)
    - ``thumbnail``: lossy JPEG at thumb_size px for quick preview
    """
    import base64
    import tempfile

    if img is None:
        return None

    result = {
        "name": img.name,
        "original_size": [img.size[0], img.size[1]],
        "colorspace": img.colorspace_settings.name if img.colorspace_settings else "sRGB",
        "source": img.source,
        "alpha_mode": img.alpha_mode,
    }
    if img.filepath:
        result["filepath"] = img.filepath

    # Force-load pixels if not already in memory
    _try_load_image_pixels(img)

    # Skip broken/placeholder images (pink missing-texture, dead refs, etc.)
    if _is_broken_image(img):
        return None  # Trash it entirely — don't include in training data

    orig_w, orig_h = img.size[0], img.size[1]
    result["original_size"] = [orig_w, orig_h]

    if orig_w == 0 or orig_h == 0:
        return None  # Can't produce a thumbnail, skip

    scene = bpy.context.scene

    def _save_image_copy(source_img, target_w, target_h, file_format, quality,
                         tmp_name, color_mode="RGB"):
        """Duplicate source_img, optionally scale, save to temp file, return bytes."""
        dup = source_img.copy()
        dup.name = tmp_name
        try:
            if target_w != source_img.size[0] or target_h != source_img.size[1]:
                dup.scale(target_w, target_h)
            ext = ".png" if file_format == "PNG" else ".jpg"
            tmp_path = os.path.join(tempfile.gettempdir(), f"{tmp_name}{ext}")
            old_fmt = scene.render.image_settings.file_format
            old_q   = scene.render.image_settings.quality
            old_cm  = scene.render.image_settings.color_mode
            scene.render.image_settings.file_format = file_format
            scene.render.image_settings.quality = quality
            # PNG supports RGBA; JPEG only RGB
            scene.render.image_settings.color_mode = color_mode
            try:
                dup.save_render(filepath=tmp_path)
            except Exception:
                dup.filepath_raw = tmp_path
                dup.file_format = file_format
                dup.save()
            scene.render.image_settings.file_format = old_fmt
            scene.render.image_settings.quality = old_q
            scene.render.image_settings.color_mode = old_cm
            if os.path.isfile(tmp_path):
                with open(tmp_path, "rb") as fh:
                    raw = fh.read()
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                return raw
        finally:
            bpy.data.images.remove(dup)
        return None

    try:
        # ── Full-resolution PNG (lossless, capped at full_size) ───────────
        full_scale = min(full_size / orig_w, full_size / orig_h, 1.0)
        full_w = max(1, int(orig_w * full_scale))
        full_h = max(1, int(orig_h * full_scale))
        # detect alpha channel
        has_alpha = img.channels == 4
        cm_full = "RGBA" if has_alpha else "RGB"
        png_bytes = _save_image_copy(
            img, full_w, full_h, "PNG", 100, "__fullres_dup", cm_full
        )
        if png_bytes:
            result["image_data"]        = base64.b64encode(png_bytes).decode("ascii")
            result["image_data_size"]   = [full_w, full_h]
            result["image_data_format"] = "png"
            result["image_data_bytes"]  = len(png_bytes)

        # ── Small thumbnail JPEG (quick preview) ─────────────────────────
        t_scale = min(thumb_size / orig_w, thumb_size / orig_h, 1.0)
        thumb_w = max(1, int(orig_w * t_scale))
        thumb_h = max(1, int(orig_h * t_scale))
        jpeg_bytes = _save_image_copy(
            img, thumb_w, thumb_h, "JPEG", IMAGE_JPEG_QUALITY, "__thumb_dup", "RGB"
        )
        if jpeg_bytes:
            result["thumbnail"]        = base64.b64encode(jpeg_bytes).decode("ascii")
            result["thumbnail_size"]   = [thumb_w, thumb_h]
            result["thumbnail_format"] = "jpeg"
            result["thumbnail_bytes"]  = len(jpeg_bytes)

    except Exception as e:
        print(f"    Warning: Could not extract image data for {img.name}: {e}")
        for tmp_name in ("__fullres_dup", "__thumb_dup"):
            dup_check = bpy.data.images.get(tmp_name)
            if dup_check:
                bpy.data.images.remove(dup_check)

    return result


def extract_node_group_data(node_group) -> dict | None:
    """Extract a standalone node group (not attached to any material).

    These are reusable shader/geometry setups in bpy.data.node_groups.
    """
    if node_group is None or not node_group.nodes:
        return None

    nodes = []
    links = []

    for node in node_group.nodes:
        node_data = {
            "name": node.name,
            "type": node.type,
            "bl_idname": node.bl_idname,
            "location": [round(node.location.x, 1), round(node.location.y, 1)],
        }
        inputs = {}
        for inp in node.inputs:
            if not inp.is_linked and hasattr(inp, "default_value"):
                val = inp.default_value
                try:
                    if hasattr(val, "__len__"):
                        inputs[inp.name] = [round(float(v), 4) for v in val]
                    else:
                        inputs[inp.name] = round(float(val), 4)
                except (TypeError, ValueError):
                    pass
        if inputs:
            node_data["inputs"] = inputs
        nodes.append(node_data)

    for link in node_group.links:
        links.append({
            "from_node": link.from_node.name,
            "from_socket": link.from_socket.name,
            "to_node": link.to_node.name,
            "to_socket": link.to_socket.name,
        })

    # Group interface inputs/outputs
    group_inputs = []
    group_outputs = []
    if hasattr(node_group, "interface"):
        for item in node_group.interface.items_tree:
            entry = {"name": item.name, "socket_type": item.socket_type}
            if item.in_out == "INPUT":
                group_inputs.append(entry)
            elif item.in_out == "OUTPUT":
                group_outputs.append(entry)
    elif hasattr(node_group, "inputs"):
        for inp in node_group.inputs:
            group_inputs.append({"name": inp.name, "type": inp.type})
        for out in node_group.outputs:
            group_outputs.append({"name": out.name, "type": out.type})

    return {
        "name": node_group.name,
        "type": node_group.type,
        "nodes": nodes,
        "links": links,
        "inputs": group_inputs,
        "outputs": group_outputs,
    }


def extract_mesh_data(obj, config: dict, *, _allow_realize_instances: bool = True) -> dict | None:
    """Extract mesh geometry from a Blender mesh object.

    Returns normalized vertex positions, face indices, and normals.
    The mesh is:
    1. Evaluated with modifiers applied (to get final geometry)
    2. Triangulated (consistent face format for training)
    3. Decimated if over max_faces (keeps all rich metadata intact)
    4. Normalized to [-1, 1] range centered at origin
    """
    if obj.type != "MESH":
        return None

    mesh_config = config.get("processing", {}).get("mesh_extraction", {})
    min_verts = mesh_config.get("min_vertices", 8)
    max_verts = mesh_config.get("max_vertices", 100000)
    precision = mesh_config.get("coordinate_precision", 4)
    normalize = mesh_config.get("normalize", True)
    max_faces = mesh_config.get("max_faces", 8000)

    def _has_nodes_modifier(_obj) -> bool:
        try:
            return any(getattr(m, "type", "") == "NODES" for m in getattr(_obj, "modifiers", []))
        except Exception:
            return False

    def _try_realize_instances_for_nodes(_obj):
        """Temporarily insert a Realize Instances node into the first GN modifier."""
        try:
            mods = list(getattr(_obj, "modifiers", []))
        except Exception:
            return None

        for mod in mods:
            try:
                if getattr(mod, "type", "") != "NODES":
                    continue
                orig_ng = getattr(mod, "node_group", None)
                if orig_ng is None:
                    continue
                ng = orig_ng.copy()

                out_node = None
                for n in ng.nodes:
                    if getattr(n, "type", "") == "GROUP_OUTPUT":
                        out_node = n
                        break
                if out_node is None:
                    try:
                        bpy.data.node_groups.remove(ng)
                    except Exception:
                        pass
                    continue

                geom_in = out_node.inputs.get("Geometry") if hasattr(out_node, "inputs") else None
                if geom_in is None and hasattr(out_node, "inputs") and out_node.inputs:
                    geom_in = out_node.inputs[0]
                if geom_in is None or not getattr(geom_in, "is_linked", False) or not geom_in.links:
                    try:
                        bpy.data.node_groups.remove(ng)
                    except Exception:
                        pass
                    continue

                link = geom_in.links[0]
                from_socket = link.from_socket
                try:
                    ng.links.remove(link)
                except Exception:
                    pass

                try:
                    realize = ng.nodes.new("GeometryNodeRealizeInstances")
                except Exception:
                    try:
                        bpy.data.node_groups.remove(ng)
                    except Exception:
                        pass
                    continue

                realize.location = (out_node.location.x - 200, out_node.location.y)

                in_sock = realize.inputs.get("Geometry") if hasattr(realize, "inputs") else None
                if in_sock is None and hasattr(realize, "inputs") and realize.inputs:
                    in_sock = realize.inputs[0]
                out_sock = realize.outputs.get("Geometry") if hasattr(realize, "outputs") else None
                if out_sock is None and hasattr(realize, "outputs") and realize.outputs:
                    out_sock = realize.outputs[0]

                if in_sock is None or out_sock is None:
                    try:
                        bpy.data.node_groups.remove(ng)
                    except Exception:
                        pass
                    continue

                try:
                    ng.links.new(from_socket, in_sock)
                    ng.links.new(out_sock, geom_in)
                except Exception:
                    try:
                        bpy.data.node_groups.remove(ng)
                    except Exception:
                        pass
                    continue

                try:
                    mod.node_group = ng
                except Exception:
                    try:
                        bpy.data.node_groups.remove(ng)
                    except Exception:
                        pass
                    continue

                try:
                    bpy.context.view_layer.update()
                except Exception:
                    pass

                return (mod, orig_ng, ng)
            except Exception:
                continue

        return None

    # Get evaluated mesh (with modifiers applied)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()

    if mesh is None or len(mesh.vertices) < min_verts:
        eval_obj.to_mesh_clear()

        # Geometry Nodes can output *instances* which do not become real mesh
        # geometry unless explicitly realized. Try a temporary Realize Instances.
        if _allow_realize_instances and _has_nodes_modifier(obj):
            token = _try_realize_instances_for_nodes(obj)
            if token is not None:
                mod, orig_ng, tmp_ng = token
                try:
                    realized = extract_mesh_data(obj, config, _allow_realize_instances=False)
                    if realized is not None:
                        realized["__realized_instances"] = True
                        return realized
                finally:
                    try:
                        mod.node_group = orig_ng
                    except Exception:
                        pass
                    try:
                        bpy.data.node_groups.remove(tmp_ng)
                    except Exception:
                        pass

        return None

    if len(mesh.vertices) > max_verts:
        eval_obj.to_mesh_clear()
        return None

    # Triangulate
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    num_tris = len(bm.faces)
    bm.to_mesh(mesh)
    bm.free()

    # Extract vertices
    vertices = []
    for v in mesh.vertices:
        # Transform to world space
        world_co = obj.matrix_world @ v.co
        vertices.append([world_co.x, world_co.y, world_co.z])

    # Extract faces (triangulated — always 3 indices)
    faces = []
    for poly in mesh.polygons:
        face = list(poly.vertices)
        if len(face) == 3:
            faces.append(face)

    # Extract split normals
    normals = []
    try:
        mesh.calc_normals_split()
    except AttributeError:
        pass
    for v in mesh.vertices:
        normals.append([round(v.normal.x, 3),
                        round(v.normal.y, 3),
                        round(v.normal.z, 3)])

    # ── Face material indices (which material slot each face uses) ──
    face_material_indices = [p.material_index for p in mesh.polygons]

    # ── UV coordinates — one (u,v) per loop (corner), indexed by loop ──
    # loop_start / loop_total on each polygon gives the loop range for that face.
    # After triangulation every polygon has exactly 3 loops.
    uv_layers = {}
    if mesh.uv_layers:
        for uv_layer in mesh.uv_layers:
            coords = []
            for loop in mesh.loops:
                uv = uv_layer.data[loop.index].uv
                coords.append([round(uv.x, 5), round(uv.y, 5)])
            uv_layers[uv_layer.name] = coords

    # ── Vertex color layers ──
    vertex_color_layers = {}
    for vcol in mesh.vertex_colors:
        colors = []
        for loop in mesh.loops:
            c = vcol.data[loop.index].color
            colors.append([round(c[0], 3), round(c[1], 3),
                           round(c[2], 3), round(c[3], 3)])
        vertex_color_layers[vcol.name] = colors

    # ── Per-face smooth flag ──
    face_smooth = [p.use_smooth for p in mesh.polygons]

    eval_obj.to_mesh_clear()

    if not faces:
        return None

    # Normalize to [-1, 1] centered at origin
    normalization_center = [0.0, 0.0, 0.0]
    normalization_scale = 1.0
    if normalize and vertices:
        import numpy as np
        verts_np = np.array(vertices)
        center = verts_np.mean(axis=0)
        verts_np -= center
        max_extent = np.abs(verts_np).max()
        if max_extent > 0.0001:
            verts_np /= max_extent
            normalization_scale = float(max_extent)
        normalization_center = [round(float(c), 6) for c in center]
        vertices = [[round(float(c), precision) for c in v] for v in verts_np]
    else:
        vertices = [[round(c, precision) for c in v] for v in vertices]

    result = {
        "num_vertices": len(vertices),
        "num_faces": len(faces),
        "vertices": vertices,
        "faces": faces,
        "normals": normals,
        "face_material_indices": face_material_indices,
        "face_smooth": face_smooth,
        "normalization_center": normalization_center,
        "normalization_scale": normalization_scale,
    }
    if uv_layers:
        result["uv_layers"] = uv_layers
    if vertex_color_layers:
        result["vertex_color_layers"] = vertex_color_layers
    return result


def extract_object_instances(obj, *, max_instances: int = 5000) -> list[dict]:
    """Extract evaluated instance transforms for an object.

    This captures Geometry Nodes and collection/particle instancing that may not
    produce a realized mesh via ``to_mesh()``.

    Returns a list of JSON-serializable dicts:
      {"source_object": <prototype object name>, "matrix_world": [16 floats]}
    """
    instances: list[dict] = []
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
    except Exception:
        return instances

    obj_name = getattr(obj, "name", "")
    if not obj_name:
        return instances

    try:
        for inst in depsgraph.object_instances:
            try:
                if not getattr(inst, "is_instance", False):
                    continue

                parent = getattr(inst, "parent", None)
                if parent is None:
                    continue
                parent_orig = getattr(parent, "original", parent)
                if getattr(parent_orig, "name", None) != obj_name:
                    continue

                # Blender provides both `object` and (sometimes) `instance_object`.
                # For many instancing systems (Geometry Nodes, particle instancing),
                # `instance_object` is the real prototype, while `object` may refer
                # to the evaluated instance wrapper.
                proto_obj = getattr(inst, "instance_object", None) or getattr(inst, "object", None)
                if proto_obj is None:
                    continue
                proto_orig = getattr(proto_obj, "original", proto_obj)
                proto_name = getattr(proto_orig, "name", None) or getattr(proto_obj, "name", "")
                if not proto_name:
                    continue

                mw = getattr(inst, "matrix_world", None)
                if mw is None:
                    continue
                mat = [round(float(mw[r][c]), 6) for r in range(4) for c in range(4)]

                instances.append({
                    "source_object": proto_name,
                    "matrix_world": mat,
                })

                if len(instances) >= int(max_instances):
                    instances.append({
                        "_truncated": True,
                        "total_shown": int(max_instances),
                    })
                    break
            except Exception:
                continue
    except Exception:
        return instances

    return instances


def extract_material_data(mat) -> dict | None:
    """Extract material properties and node tree structure.

    Returns a structured representation of the shader node graph
    that can be used to recreate the material.
    """
    if mat is None:
        return None

    result = {
        "name": mat.name,
        "use_nodes": mat.use_nodes,
    }

    if not mat.use_nodes or not mat.node_tree:
        # Simple material — extract basic properties
        result["type"] = "simple"
        return result

    # Extract node tree
    nodes = []
    links = []
    node_tree = mat.node_tree

    for node in node_tree.nodes:
        node_data = {
            "name": node.name,
            "type": node.type,
            "bl_idname": node.bl_idname,
            "location": [round(node.location.x, 1), round(node.location.y, 1)],
        }

        # Extract node-specific properties
        if node.type == "BSDF_PRINCIPLED":
            inputs = {}
            for inp in node.inputs:
                if inp.is_linked:
                    inputs[inp.name] = "LINKED"
                elif hasattr(inp, "default_value"):
                    val = inp.default_value
                    if hasattr(val, "__len__"):
                        inputs[inp.name] = [round(float(v), 4) for v in val]
                    else:
                        inputs[inp.name] = round(float(val), 4)
            node_data["inputs"] = inputs

        elif node.type in ("TEX_IMAGE", "TEX_NOISE", "TEX_BRICK",
                            "TEX_CHECKER", "TEX_GRADIENT", "TEX_WAVE"):
            node_data["texture_type"] = node.type
            if node.type == "TEX_IMAGE" and node.image:
                # Check if this texture is broken/missing
                _try_load_image_pixels(node.image)
                if _is_broken_image(node.image):
                    node_data["image_broken"] = True
                    # Still record the name so we know what was intended
                    node_data["image_name"] = node.image.name
                else:
                    node_data["image_name"] = node.image.name
                    node_data["image_colorspace"] = (
                        node.image.colorspace_settings.name
                        if node.image.colorspace_settings else "sRGB"
                    )
                    node_data["image_original_size"] = [
                        node.image.size[0], node.image.size[1]
                    ]
                if hasattr(node, "interpolation"):
                    node_data["interpolation"] = node.interpolation
                if hasattr(node, "extension"):
                    node_data["extension"] = node.extension
                if hasattr(node, "projection"):
                    node_data["projection"] = node.projection
            # Extract relevant properties
            if hasattr(node, "noise_dimensions"):
                node_data["noise_dimensions"] = node.noise_dimensions

        elif node.type == "MIX_RGB":
            node_data["blend_type"] = node.blend_type
            node_data["use_clamp"] = node.use_clamp

        elif node.type == "MIX":
            # ShaderNodeMix (Blender 3.4+) — replaces MIX_RGB
            if hasattr(node, "data_type"):
                node_data["data_type"] = node.data_type
            if hasattr(node, "blend_type"):
                node_data["blend_type"] = node.blend_type
            if hasattr(node, "clamp_factor"):
                node_data["clamp_factor"] = node.clamp_factor
            if hasattr(node, "clamp_result"):
                node_data["clamp_result"] = node.clamp_result
            if hasattr(node, "factor_mode"):
                node_data["factor_mode"] = node.factor_mode

        elif node.type == "MATH":
            node_data["operation"] = node.operation
            if hasattr(node, "use_clamp"):
                node_data["use_clamp"] = node.use_clamp

        elif node.type == "CLAMP":
            if hasattr(node, "clamp_type"):
                node_data["clamp_type"] = node.clamp_type

        elif node.type in ("SEPARATE_COLOR", "COMBINE_COLOR"):
            if hasattr(node, "mode"):
                node_data["mode"] = node.mode

        elif node.type in ("MAPPING", "TEX_COORD", "NORMAL_MAP",
                            "BUMP", "DISPLACEMENT"):
            pass  # Just capture type

        # ── ColorRamp / Curve elements (works for any node with color_ramp) ──
        if hasattr(node, "color_ramp"):
            try:
                cr = node.color_ramp
                cr_data = {
                    "interpolation": cr.interpolation,
                    "elements": [],
                }
                for el in cr.elements:
                    cr_data["elements"].append({
                        "position": round(float(el.position), 4),
                        "color": [round(float(c), 4) for c in el.color],
                    })
                node_data["color_ramp"] = cr_data
            except Exception:
                pass

        # Extract all input default values generically
        if "inputs" not in node_data:
            inputs = {}
            for inp in node.inputs:
                if not inp.is_linked and hasattr(inp, "default_value"):
                    val = inp.default_value
                    try:
                        if hasattr(val, "__len__"):
                            inputs[inp.name] = [round(float(v), 4) for v in val]
                        else:
                            inputs[inp.name] = round(float(val), 4)
                    except (TypeError, ValueError):
                        pass
            if inputs:
                node_data["inputs"] = inputs

        nodes.append(node_data)

    # Extract links — use socket INDEX to disambiguate duplicate names
    # (e.g. ShaderNodeMix has multiple "Result" outputs)
    for link in node_tree.links:
        link_data = {
            "from_node": link.from_node.name,
            "from_socket": link.from_socket.name,
            "to_node": link.to_node.name,
            "to_socket": link.to_socket.name,
        }
        # Add socket indices for disambiguation
        for i, s in enumerate(link.from_node.outputs):
            if s == link.from_socket:
                link_data["from_socket_index"] = i
                break
        for i, s in enumerate(link.to_node.inputs):
            if s == link.to_socket:
                link_data["to_socket_index"] = i
                break
        links.append(link_data)

    result["type"] = "node_tree"
    result["nodes"] = nodes
    result["links"] = links

    return result


def extract_modifier_stack(obj) -> list[dict]:
    """Extract modifier stack configuration."""
    modifiers = []
    for mod in obj.modifiers:
        mod_data = {
            "type": mod.type,
            "name": mod.name,
            "show_viewport": mod.show_viewport,
        }

        # Extract type-specific properties
        if mod.type == "SUBSURF":
            mod_data["levels"] = mod.levels
            mod_data["render_levels"] = mod.render_levels
            mod_data["subdivision_type"] = mod.subdivision_type
        elif mod.type == "MIRROR":
            mod_data["use_axis"] = [mod.use_axis[0], mod.use_axis[1], mod.use_axis[2]]
            mod_data["use_clip"] = mod.use_clip
        elif mod.type == "SOLIDIFY":
            mod_data["thickness"] = round(mod.thickness, 4)
            mod_data["offset"] = round(mod.offset, 4)
        elif mod.type == "BEVEL":
            mod_data["width"] = round(mod.width, 4)
            mod_data["segments"] = mod.segments
            mod_data["limit_method"] = mod.limit_method
        elif mod.type == "ARRAY":
            mod_data["count"] = mod.count
            mod_data["use_relative_offset"] = mod.use_relative_offset
            if mod.use_relative_offset:
                mod_data["relative_offset_displace"] = [
                    round(v, 4) for v in mod.relative_offset_displace
                ]
        elif mod.type == "BOOLEAN":
            mod_data["operation"] = mod.operation
            mod_data["solver"] = mod.solver
        elif mod.type == "SHRINKWRAP":
            mod_data["wrap_method"] = mod.wrap_method
            mod_data["offset"] = round(mod.offset, 4)
        elif mod.type == "SIMPLE_DEFORM":
            mod_data["deform_method"] = mod.deform_method
            mod_data["angle"] = round(mod.angle, 4)
            mod_data["deform_axis"] = mod.deform_axis
        elif mod.type == "SCREW":
            mod_data["angle"] = round(mod.angle, 4)
            mod_data["steps"] = mod.steps
            mod_data["render_steps"] = mod.render_steps
            mod_data["axis"] = mod.axis
        elif mod.type == "DECIMATE":
            mod_data["decimate_type"] = mod.decimate_type
            mod_data["ratio"] = round(mod.ratio, 4)
        elif mod.type == "SKIN":
            pass  # Complex — just capture type
        elif mod.type == "WIREFRAME":
            mod_data["thickness"] = round(mod.thickness, 4)
        elif mod.type == "NODES":
            # Geometry Nodes — store the node group name so it can be re-linked
            if mod.node_group:
                mod_data["node_group_name"] = mod.node_group.name
        elif mod.type == "ARMATURE":
            if mod.object:
                mod_data["armature_object"] = mod.object.name
            mod_data["use_vertex_groups"] = mod.use_vertex_groups
            mod_data["use_bone_envelopes"] = mod.use_bone_envelopes
        elif mod.type == "CLOTH":
            s = mod.settings
            mod_data["mass"] = round(s.mass, 4)
            mod_data["tension_stiffness"] = round(s.tension_stiffness, 4)
            mod_data["compression_stiffness"] = round(s.compression_stiffness, 4)
        elif mod.type == "CORRECTIVE_SMOOTH":
            mod_data["factor"] = round(mod.factor, 4)
            mod_data["iterations"] = mod.iterations
        elif mod.type == "SURFACE_DEFORM":
            if mod.target:
                mod_data["target"] = mod.target.name

        modifiers.append(mod_data)

    return modifiers


def extract_keyframes(obj, max_fcurves: int = 200, max_keypoints: int = 100) -> list[dict]:
    """Extract keyframe animation data for an object.

    Caps at max_fcurves and max_keypoints per curve to prevent
    multi-MB animation data on complex rigs.
    """
    keyframes = []
    if not obj.animation_data or not obj.animation_data.action:
        return keyframes

    action = obj.animation_data.action
    total_fcurves = len(action.fcurves)
    for fi, fcurve in enumerate(action.fcurves):
        if fi >= max_fcurves:
            keyframes.append({"_truncated": True, "total_fcurves": total_fcurves,
                              "shown": max_fcurves})
            break
        curve_data = {
            "data_path": fcurve.data_path,
            "array_index": fcurve.array_index,
            "num_keyframe_points": len(fcurve.keyframe_points),
            "keyframe_points": [],
        }
        for ki, kp in enumerate(fcurve.keyframe_points):
            if ki >= max_keypoints:
                curve_data["keyframes_truncated"] = True
                break
            curve_data["keyframe_points"].append({
                "co": [round(kp.co.x, 2), round(float(kp.co.y), 4)],
                "interpolation": kp.interpolation,
                "handle_left": [round(kp.handle_left.x, 2), round(float(kp.handle_left.y), 4)],
                "handle_right": [round(kp.handle_right.x, 2), round(float(kp.handle_right.y), 4)],
            })
        keyframes.append(curve_data)
    return keyframes


def extract_constraints(obj) -> list[dict]:
    """Extract constraint stack for an object."""
    constraints = []
    for con in obj.constraints:
        con_data = {
            "type": con.type,
            "name": con.name,
            "mute": con.mute,
            "influence": round(con.influence, 4),
        }
        if hasattr(con, "target") and con.target:
            con_data["target"] = con.target.name
        if hasattr(con, "subtarget") and con.subtarget:
            con_data["subtarget"] = con.subtarget
        if con.type == "COPY_LOCATION":
            con_data["use_x"] = con.use_x
            con_data["use_y"] = con.use_y
            con_data["use_z"] = con.use_z
        elif con.type == "COPY_ROTATION":
            con_data["use_x"] = con.use_x
            con_data["use_y"] = con.use_y
            con_data["use_z"] = con.use_z
        elif con.type == "TRACK_TO":
            con_data["track_axis"] = con.track_axis
            con_data["up_axis"] = con.up_axis
        elif con.type == "IK":
            con_data["chain_count"] = con.chain_count
            if hasattr(con, "pole_target") and con.pole_target:
                con_data["pole_target"] = con.pole_target.name
        elif con.type in ("LIMIT_LOCATION", "LIMIT_ROTATION", "LIMIT_SCALE"):
            for attr in ("use_min_x", "use_max_x", "use_min_y", "use_max_y",
                         "use_min_z", "use_max_z", "min_x", "max_x",
                         "min_y", "max_y", "min_z", "max_z"):
                if hasattr(con, attr):
                    val = getattr(con, attr)
                    con_data[attr] = round(val, 4) if isinstance(val, float) else val
        elif con.type == "DAMPED_TRACK":
            con_data["track_axis"] = con.track_axis
        elif con.type == "STRETCH_TO":
            con_data["rest_length"] = round(con.rest_length, 4)
            con_data["volume"] = con.volume
        constraints.append(con_data)
    return constraints


def extract_shape_keys(obj, max_deltas_per_key: int = 200) -> dict | None:
    """Extract shape keys with vertex deltas.

    Vertex deltas are CRITICAL training data — they teach the model how
    shape keys actually deform geometry (preventing clipping when bones
    move, facial expressions, etc). We keep the top deltas sorted by
    magnitude so the model learns which vertices matter most.
    """
    if obj.type != "MESH" or not obj.data.shape_keys:
        return None

    sk = obj.data.shape_keys
    result = {
        "name": sk.name,
        "use_relative": sk.use_relative,
        "key_blocks": [],
    }
    basis = sk.key_blocks[0] if sk.key_blocks else None
    for kb in sk.key_blocks:
        block = {
            "name": kb.name,
            "value": round(kb.value, 4),
            "slider_min": round(kb.slider_min, 4),
            "slider_max": round(kb.slider_max, 4),
            "mute": kb.mute,
        }
        if kb.relative_key:
            block["relative_key"] = kb.relative_key.name
        if basis and kb != basis:
            try:
                deltas = []
                for i, (kv, bv) in enumerate(zip(kb.data, basis.data)):
                    dx = kv.co.x - bv.co.x
                    dy = kv.co.y - bv.co.y
                    dz = kv.co.z - bv.co.z
                    mag = dx * dx + dy * dy + dz * dz
                    if mag > 1e-8:
                        deltas.append((mag, i, dx, dy, dz))
                block["affected_vertices"] = len(deltas)
                deltas.sort(key=lambda x: -x[0])
                if len(deltas) > max_deltas_per_key:
                    block["deltas_truncated"] = True
                    deltas = deltas[:max_deltas_per_key]
                block["vertex_deltas"] = [
                    {"vertex": d[1],
                     "delta": [round(d[2], 4), round(d[3], 4), round(d[4], 4)]}
                    for d in deltas
                ]
            except Exception:
                pass
        result["key_blocks"].append(block)

    if sk.animation_data and sk.animation_data.action:
        result["keyframes"] = []
        for fcurve in sk.animation_data.action.fcurves:
            result["keyframes"].append({
                "data_path": fcurve.data_path,
                "array_index": fcurve.array_index,
                "keyframe_points": [
                    {"frame": round(kp.co.x, 2), "value": round(float(kp.co.y), 4)}
                    for kp in fcurve.keyframe_points
                ],
            })
    return result


def extract_armature_data(obj) -> dict | None:
    """Extract armature/skeleton bone hierarchy and pose data."""
    if obj.type != "ARMATURE" or not obj.data:
        return None

    armature = obj.data
    bones = []
    for bone in armature.bones:
        bone_data = {
            "name": bone.name,
            "head": [round(v, 4) for v in bone.head_local],
            "tail": [round(v, 4) for v in bone.tail_local],
            "length": round(bone.length, 4),
            "use_connect": bone.use_connect,
            "use_deform": bone.use_deform,
        }
        if bone.parent:
            bone_data["parent"] = bone.parent.name
        children = [c.name for c in bone.children]
        if children:
            bone_data["children"] = children
        bones.append(bone_data)

    result = {
        "name": armature.name,
        "bone_count": len(bones),
        "bones": bones,
    }

    if obj.pose:
        pose_bones = []
        for pb in obj.pose.bones:
            pb_data = {
                "name": pb.name,
                "location": [round(v, 4) for v in pb.location],
                "rotation_quaternion": [round(v, 4) for v in pb.rotation_quaternion],
                "scale": [round(v, 4) for v in pb.scale],
                "rotation_mode": pb.rotation_mode,
            }
            if pb.constraints:
                pb_data["constraints"] = []
                for con in pb.constraints:
                    con_data = {
                        "type": con.type,
                        "name": con.name,
                        "influence": round(con.influence, 4),
                        "mute": con.mute,
                    }
                    if hasattr(con, "target") and con.target:
                        con_data["target"] = con.target.name
                    if hasattr(con, "subtarget") and con.subtarget:
                        con_data["subtarget"] = con.subtarget
                    pb_data["constraints"].append(con_data)
            pose_bones.append(pb_data)
        result["pose_bones"] = pose_bones

    return result


def extract_rigid_body_data(obj) -> dict | None:
    """Extract rigid body physics settings from an object.

    Returns a dict with type, mass, shape, and constraint data, or None
    if the object has no rigid body simulation.
    """
    rb = getattr(obj, "rigid_body", None)
    if rb is None:
        return None
    result = {
        "type": rb.type,            # 'ACTIVE' or 'PASSIVE'
        "mass": round(rb.mass, 4),
        "collision_shape": rb.collision_shape,
        "friction": round(rb.friction, 4),
        "restitution": round(rb.restitution, 4),
        "linear_damping": round(rb.linear_damping, 4),
        "angular_damping": round(rb.angular_damping, 4),
        "use_margin": rb.use_margin,
        "collision_margin": round(rb.collision_margin, 4),
        "enabled": rb.enabled,
        "kinematic": rb.kinematic,
    }
    # Rigid body constraint (joint)
    rbc = getattr(obj, "rigid_body_constraint", None)
    if rbc is not None:
        constraint = {
            "type": rbc.type,
            "enabled": rbc.enabled,
            "disable_collisions": rbc.disable_collisions,
        }
        if rbc.object1:
            constraint["object1"] = rbc.object1.name
        if rbc.object2:
            constraint["object2"] = rbc.object2.name
        result["constraint"] = constraint
    return result


def extract_particle_systems(obj) -> list[dict]:
    """Extract particle system settings."""
    particles = []
    for ps in obj.particle_systems:
        settings = ps.settings
        ps_data = {
            "name": ps.name,
            "type": settings.type,
            "count": settings.count,
            "emit_from": settings.emit_from,
            "distribution": settings.distribution,
            "physics_type": settings.physics_type,
            "render_type": settings.render_type,
        }
        if settings.type == "HAIR":
            ps_data["hair_length"] = round(settings.hair_length, 4)
            ps_data["hair_step"] = settings.hair_step
        elif settings.type == "EMITTER":
            ps_data["lifetime"] = round(settings.lifetime, 2)
            ps_data["frame_start"] = round(settings.frame_start, 2)
            ps_data["frame_end"] = round(settings.frame_end, 2)
        if settings.render_type == "OBJECT" and settings.instance_object:
            ps_data["instance_object"] = settings.instance_object.name
        elif settings.render_type == "COLLECTION" and settings.instance_collection:
            ps_data["instance_collection"] = settings.instance_collection.name
        particles.append(ps_data)
    return particles


def extract_uv_maps(obj) -> list[dict]:
    """Extract UV map names and basic stats."""
    if obj.type != "MESH" or not obj.data.uv_layers:
        return []
    uv_maps = []
    for uv in obj.data.uv_layers:
        uv_maps.append({
            "name": uv.name,
            "active_render": uv.active_render,
        })
    return uv_maps


def extract_camera_data(obj) -> dict | None:
    """Extract camera settings."""
    if obj.type != "CAMERA" or not obj.data:
        return None
    cam = obj.data
    return {
        "type": cam.type,
        "lens": round(cam.lens, 2) if cam.type == "PERSP" else None,
        "ortho_scale": round(cam.ortho_scale, 4) if cam.type == "ORTHO" else None,
        "clip_start": round(cam.clip_start, 4),
        "clip_end": round(cam.clip_end, 2),
        "sensor_width": round(cam.sensor_width, 2),
        "sensor_height": round(cam.sensor_height, 2),
        "dof_use": cam.dof.use_dof if hasattr(cam.dof, "use_dof") else False,
        "dof_distance": round(cam.dof.focus_distance, 4) if hasattr(cam.dof, "focus_distance") else 0,
    }


def extract_light_data(obj) -> dict | None:
    """Extract light settings."""
    if obj.type != "LIGHT" or not obj.data:
        return None
    light = obj.data
    result = {
        "type": light.type,
        "color": [round(v, 4) for v in light.color],
        "energy": round(light.energy, 2),
    }
    if light.type in ("POINT", "SPOT"):
        result["shadow_soft_size"] = round(light.shadow_soft_size, 4)
    if light.type == "SPOT":
        result["spot_size"] = round(light.spot_size, 4)
        result["spot_blend"] = round(light.spot_blend, 4)
    if light.type == "AREA":
        result["shape"] = light.shape
        result["size"] = round(light.size, 4)
    if light.type == "SUN":
        result["angle"] = round(light.angle, 4)
    return result


def extract_curve_data(obj) -> dict | None:
    """Extract curve/surface data."""
    if obj.type not in ("CURVE", "SURFACE") or not obj.data:
        return None
    curve = obj.data
    result = {
        "dimensions": curve.dimensions,
        "resolution_u": curve.resolution_u,
        "fill_mode": curve.fill_mode,
        "bevel_depth": round(curve.bevel_depth, 4),
        "bevel_resolution": curve.bevel_resolution,
        "extrude": round(curve.extrude, 4),
        "splines": [],
    }
    for spline in curve.splines:
        sp_data = {
            "type": spline.type,
            "use_cyclic_u": spline.use_cyclic_u,
            "order_u": spline.order_u,
        }
        if spline.type == "BEZIER":
            points = []
            for bp in spline.bezier_points:
                points.append({
                    "co": [round(v, 4) for v in bp.co],
                    "handle_left": [round(v, 4) for v in bp.handle_left],
                    "handle_right": [round(v, 4) for v in bp.handle_right],
                    "handle_left_type": bp.handle_left_type,
                    "handle_right_type": bp.handle_right_type,
                })
            sp_data["bezier_points"] = points
        else:
            points = []
            for p in spline.points:
                points.append({
                    "co": [round(v, 4) for v in p.co],
                    "weight": round(p.weight, 4),
                })
            sp_data["points"] = points
        result["splines"].append(sp_data)
    return result


def extract_grease_pencil(obj) -> dict | None:
    """Extract grease pencil drawing data.

    Caps: 100 points/stroke, 50 strokes/frame, 20 frames/layer.
    """
    if obj.type != "GPENCIL" or not obj.data:
        return None
    gpd = obj.data
    result = {
        "name": gpd.name,
        "layers": [],
    }
    for layer in gpd.layers:
        layer_data = {
            "name": layer.info,
            "opacity": round(layer.opacity, 4),
            "blend_mode": layer.blend_mode,
            "total_frames": len(layer.frames),
            "frames": [],
        }
        for fi, frame in enumerate(layer.frames):
            if fi >= 20:
                layer_data["frames_truncated"] = True
                break
            frame_data = {
                "frame_number": frame.frame_number,
                "total_strokes": len(frame.strokes),
                "strokes": [],
            }
            for si, stroke in enumerate(frame.strokes):
                if si >= 50:
                    frame_data["strokes_truncated"] = True
                    break
                stroke_data = {
                    "material_index": stroke.material_index,
                    "line_width": stroke.line_width,
                    "num_points": len(stroke.points),
                    "points": [],
                }
                for pi, pt in enumerate(stroke.points):
                    if pi >= 100:
                        stroke_data["points_truncated"] = True
                        break
                    stroke_data["points"].append({
                        "co": [round(v, 4) for v in pt.co],
                        "pressure": round(pt.pressure, 3),
                        "strength": round(pt.strength, 3),
                    })
                frame_data["strokes"].append(stroke_data)
            layer_data["frames"].append(frame_data)
        result["layers"].append(layer_data)
    return result


def extract_drivers(obj, max_drivers: int = 200) -> list[dict]:
    """Extract driver expressions from the object's animation data.

    Caps at max_drivers to prevent multi-MB driver lists on complex rigs.
    """
    drivers = []
    if not obj.animation_data or not obj.animation_data.drivers:
        return drivers
    total = len(obj.animation_data.drivers)
    for d in obj.animation_data.drivers:
        if len(drivers) >= max_drivers:
            drivers.append({"_truncated": True, "total_drivers": total,
                            "shown": max_drivers})
            break
        try:
            drv = d.driver
            driver_data = {
                "data_path": d.data_path,
                "array_index": d.array_index,
                "type": drv.type,
                "expression": drv.expression if drv.type == "SCRIPTED" else "",
                "variables": [],
            }
            for var in drv.variables:
                var_data = {
                    "name": var.name,
                    "type": var.type,
                }
                targets = []
                try:
                    for tgt in var.targets:
                        t = {}
                        if tgt.id:
                            t["id"] = tgt.id.name
                            t["id_type"] = tgt.id_type
                        if tgt.data_path:
                            t["data_path"] = tgt.data_path
                        if tgt.bone_target:
                            t["bone_target"] = tgt.bone_target
                        if tgt.transform_type:
                            t["transform_type"] = tgt.transform_type
                        targets.append(t)
                except Exception:
                    pass
                var_data["targets"] = targets
                driver_data["variables"].append(var_data)
            drivers.append(driver_data)
        except Exception:
            drivers.append({"data_path": getattr(d, "data_path", "?"), "error": True})
    return drivers


def extract_object_data(obj, config: dict) -> dict | None:
    """Extract complete object data: geometry, materials, modifiers,
    animation, constraints, shape keys, particles, and more.

    Handles ALL Blender object types: MESH, CURVE, SURFACE, ARMATURE,
    CAMERA, LIGHT, EMPTY, GPENCIL, LATTICE, etc.
    """
    skip_types = {"FONT"}  # Font objects rarely useful for training

    if obj.type in skip_types:
        return None

    result = {
        "name": obj.name,
        "type": obj.type,
        "transforms": {
            "location": [round(v, 4) for v in obj.location],
            "rotation_euler": [round(v, 4) for v in obj.rotation_euler],
            "rotation_mode": obj.rotation_mode,
            "scale": [round(v, 4) for v in obj.scale],
        },
        "dimensions": [round(v, 4) for v in obj.dimensions],
        "visible": not obj.hide_viewport,
        "hide_viewport": bool(obj.hide_viewport),
        "hide_render": bool(obj.hide_render),
        "hide_select": bool(obj.hide_select),
    }

    # Parent info
    if obj.parent:
        result["parent"] = obj.parent.name
        if obj.parent_type:
            result["parent_type"] = obj.parent_type
        if obj.parent_bone:
            result["parent_bone"] = obj.parent_bone

    # Collections this object belongs to
    collections = [c.name for c in obj.users_collection]
    if collections:
        result["collections"] = collections

    # ── Type-specific data ──

    if obj.type == "MESH":
        mesh_data = extract_mesh_data(obj, config)
        if mesh_data is not None:
            if bool(mesh_data.pop("__realized_instances", False)):
                result["mesh_extraction"] = {"realized_instances": True}
            result["mesh"] = mesh_data
        else:
            # Geometry Nodes / instancing objects can evaluate to an empty mesh
            # (instances are not realized). Preserve them by exporting instance
            # transforms so the validator can reconstruct the full scene.
            has_nodes = False
            try:
                has_nodes = any(getattr(m, "type", "") == "NODES" for m in getattr(obj, "modifiers", []))
            except Exception:
                has_nodes = False

            inst = extract_object_instances(obj)
            if inst and (has_nodes or getattr(obj, "instance_type", "NONE") != "NONE" or getattr(obj, "particle_systems", None)):
                result["mesh"] = {
                    "num_vertices": 0,
                    "num_faces": 0,
                    "vertices": [],
                    "faces": [],
                    "normals": [],
                    "face_material_indices": [],
                    "face_smooth": [],
                    "normalization_center": [0.0, 0.0, 0.0],
                    "normalization_scale": 1.0,
                }
                result["instances"] = inst
            else:
                return None

        # Raw (pre-modifier) mesh — stored so the validator can re-apply the
        # modifier stack correctly without double-processing evaluated geometry.
        # Only captured when modifiers exist (otherwise raw == evaluated).
        if obj.data and obj.modifiers:
            try:
                bm_r = bmesh.new()
                bm_r.from_mesh(obj.data)
                bmesh.ops.triangulate(bm_r, faces=bm_r.faces[:])
                rv = [
                    [round(float(v.co.x), 4),
                     round(float(v.co.y), 4),
                     round(float(v.co.z), 4)]
                    for v in bm_r.verts
                ]
                rf = [[v.index for v in f.verts] for f in bm_r.faces]
                bm_r.free()
                if rv and rf:
                    # Normalize using the same center+scale as the evaluated mesh
                    nc = mesh_data.get("normalization_center", [0.0, 0.0, 0.0])
                    ns = float(mesh_data.get("normalization_scale", 1.0))
                    if ns > 0.0001:
                        rv = [
                            [
                                round((v[0] - nc[0]) / ns, 4),
                                round((v[1] - nc[1]) / ns, 4),
                                round((v[2] - nc[2]) / ns, 4),
                            ]
                            for v in rv
                        ]
                    result["raw_vertices"] = rv
                    result["raw_faces"] = rf
            except Exception:
                pass

        # Shape keys (blend shapes / morph targets)
        sk_data = extract_shape_keys(obj)
        if sk_data:
            result["shape_keys"] = sk_data

        # UV maps
        uv_maps = extract_uv_maps(obj)
        if uv_maps:
            result["uv_maps"] = uv_maps

        # Vertex colors
        if obj.data.vertex_colors:
            result["vertex_color_layers"] = [vc.name for vc in obj.data.vertex_colors]

        # Smooth shading
        if obj.data:
            try:
                shade_smooth = any(p.use_smooth for p in obj.data.polygons)
                result["shade_smooth"] = shade_smooth
            except Exception:
                pass

    elif obj.type == "ARMATURE":
        arm_data = extract_armature_data(obj)
        if arm_data:
            result["armature"] = arm_data

    elif obj.type == "CAMERA":
        cam_data = extract_camera_data(obj)
        if cam_data:
            result["camera"] = cam_data

    elif obj.type == "LIGHT":
        light_data = extract_light_data(obj)
        if light_data:
            result["light"] = light_data

    elif obj.type in ("CURVE", "SURFACE"):
        curve_data = extract_curve_data(obj)
        if curve_data:
            result["curve"] = curve_data

    elif obj.type == "GPENCIL":
        gp_data = extract_grease_pencil(obj)
        if gp_data:
            result["grease_pencil"] = gp_data

    elif obj.type == "EMPTY":
        result["empty_display_type"] = obj.empty_display_type
        result["empty_display_size"] = round(obj.empty_display_size, 4)
        if obj.instance_type != "NONE":
            result["instance_type"] = obj.instance_type
            if obj.instance_collection:
                result["instance_collection"] = obj.instance_collection.name

    elif obj.type == "LATTICE" and obj.data:
        lat = obj.data
        result["lattice"] = {
            "points_u": lat.points_u,
            "points_v": lat.points_v,
            "points_w": lat.points_w,
            "interpolation_type_u": lat.interpolation_type_u,
            "interpolation_type_v": lat.interpolation_type_v,
            "interpolation_type_w": lat.interpolation_type_w,
        }

    # ── Common data for all object types ──

    # Materials
    materials = []
    for slot in obj.material_slots:
        if slot.material:
            mat_data = extract_material_data(slot.material)
            if mat_data:
                materials.append(mat_data)
    if materials:
        result["materials"] = materials

    # Modifiers
    if hasattr(obj, "modifiers") and obj.modifiers:
        mods = extract_modifier_stack(obj)
        if mods:
            result["modifiers"] = mods

    # Vertex groups
    if hasattr(obj, "vertex_groups"):
        vgroups = []
        for vg in obj.vertex_groups:
            vgroups.append({"name": vg.name, "index": vg.index})
        if vgroups:
            result["vertex_groups"] = vgroups

    # Constraints
    constraints = extract_constraints(obj)
    if constraints:
        result["constraints"] = constraints

    # Rigid body physics
    try:
        rb = extract_rigid_body_data(obj)
        if rb:
            result["rigid_body"] = rb
    except Exception:
        pass

    # Keyframe animation
    keyframes = extract_keyframes(obj)
    if keyframes:
        result["keyframes"] = keyframes
        if obj.animation_data and obj.animation_data.action:
            result["action_name"] = obj.animation_data.action.name

    # Particle systems
    if hasattr(obj, "particle_systems") and obj.particle_systems:
        particles = extract_particle_systems(obj)
        if particles:
            result["particle_systems"] = particles

    # Drivers
    drivers = extract_drivers(obj)
    if drivers:
        result["drivers"] = drivers

    # Custom properties (user-defined metadata)
    custom_props = {}
    for key in obj.keys():
        if key.startswith("_"):
            continue
        val = obj[key]
        try:
            if isinstance(val, (int, float, str, bool)):
                custom_props[key] = val
            elif hasattr(val, "to_list"):
                custom_props[key] = val.to_list()
        except Exception:
            pass
    if custom_props:
        result["custom_properties"] = custom_props

    return result


def extract_collection_hierarchy(collection, depth=0) -> dict:
    """Recursively extract collection hierarchy."""
    result = {
        "name": collection.name,
        "hide_viewport": collection.hide_viewport,
        "objects": [obj.name for obj in collection.objects],
    }
    children = []
    for child in collection.children:
        if depth < 10:
            children.append(extract_collection_hierarchy(child, depth + 1))
    if children:
        result["children"] = children
    return result


def extract_world_data(world) -> dict:
    """Extract world/environment settings including node tree."""
    result = {
        "name": world.name,
        "use_nodes": world.use_nodes,
    }
    if world.use_nodes and world.node_tree:
        nodes = []
        links = []
        for node in world.node_tree.nodes:
            node_data = {
                "name": node.name,
                "type": node.type,
                "bl_idname": node.bl_idname,
            }
            # Store image name for Environment/Image Texture nodes
            if node.type in ("TEX_ENVIRONMENT", "TEX_IMAGE"):
                if hasattr(node, "image") and node.image:
                    node_data["image_name"] = node.image.name
            inputs = {}
            for inp in node.inputs:
                if not inp.is_linked and hasattr(inp, "default_value"):
                    val = inp.default_value
                    try:
                        if hasattr(val, "__len__"):
                            inputs[inp.name] = [round(float(v), 4) for v in val]
                        else:
                            inputs[inp.name] = round(float(val), 4)
                    except (TypeError, ValueError):
                        pass
            if inputs:
                node_data["inputs"] = inputs
            nodes.append(node_data)
        for link in world.node_tree.links:
            links.append({
                "from_node": link.from_node.name,
                "from_socket": link.from_socket.name,
                "to_node": link.to_node.name,
                "to_socket": link.to_socket.name,
            })
        result["nodes"] = nodes
        result["links"] = links
    return result


def extract_scene_data(config: dict) -> dict:
    """Extract EVERYTHING from the current scene.

    Captures all object types (mesh, armature, camera, light, grease pencil,
    curve, empty, lattice), plus scene-level data like collections, world,
    render settings, timeline, and markers.
    """
    scene = bpy.context.scene

    objects = []
    for obj in scene.objects:
        try:
            obj_data = extract_object_data(obj, config)
            if obj_data:
                objects.append(obj_data)
        except Exception as e:
            print(f"  Warning: Failed to extract {obj.name}: {e}")
            continue

    # Scene-level data
    scene_data = {
        "scene_name": scene.name,
        "frame_start": scene.frame_start,
        "frame_end": scene.frame_end,
        "frame_current": scene.frame_current,
        "fps": scene.render.fps,
        "objects": objects,
        "object_count": len(objects),
    }

    # Active camera
    if scene.camera:
        scene_data["active_camera"] = scene.camera.name

    # Collection hierarchy
    try:
        scene_data["collections"] = extract_collection_hierarchy(
            scene.collection
        )
    except Exception:
        pass

    # World/environment with full node tree
    if scene.world:
        try:
            scene_data["world"] = extract_world_data(scene.world)
        except Exception:
            scene_data["world"] = {"name": scene.world.name}

    # Render settings
    try:
        r = scene.render
        scene_data["render"] = {
            "engine": r.engine,
            "resolution_x": r.resolution_x,
            "resolution_y": r.resolution_y,
            "resolution_percentage": r.resolution_percentage,
            "film_transparent": r.film_transparent,
        }
        if r.engine == "CYCLES":
            cycles = scene.cycles
            scene_data["render"]["cycles"] = {
                "samples": cycles.samples,
                "use_denoising": cycles.use_denoising,
                "device": cycles.device,
            }
        elif r.engine == "BLENDER_EEVEE" or r.engine == "BLENDER_EEVEE_NEXT":
            eevee = scene.eevee
            scene_data["render"]["eevee"] = {
                "taa_render_samples": eevee.taa_render_samples,
                "use_bloom": getattr(eevee, "use_bloom", False),
                "use_ssr": getattr(eevee, "use_ssr", False),
                "use_gtao": getattr(eevee, "use_gtao", False),
            }
    except Exception:
        pass

    # Timeline markers
    markers = []
    for marker in scene.timeline_markers:
        markers.append({
            "name": marker.name,
            "frame": marker.frame,
        })
    if markers:
        scene_data["markers"] = markers

    # NLA tracks (non-linear animation)
    nla_data = []
    for obj in scene.objects:
        if obj.animation_data and obj.animation_data.nla_tracks:
            tracks = []
            for track in obj.animation_data.nla_tracks:
                strips = []
                for strip in track.strips:
                    strip_data = {
                        "name": strip.name,
                        "action": strip.action.name if strip.action else None,
                        "frame_start": round(strip.frame_start, 2),
                        "frame_end": round(strip.frame_end, 2),
                        "blend_type": strip.blend_type,
                        "influence": round(strip.influence, 4),
                        "mute": strip.mute,
                    }
                    strips.append(strip_data)
                tracks.append({
                    "name": track.name,
                    "mute": track.mute,
                    "strips": strips,
                })
            if tracks:
                nla_data.append({
                    "object": obj.name,
                    "tracks": tracks,
                })
    if nla_data:
        scene_data["nla_tracks"] = nla_data

    # ── Images with downscaled thumbnails ──
    # Skips broken/missing textures (pink placeholders, dead external refs)
    images_data = {}
    broken_count = 0
    for img in bpy.data.images:
        try:
            img_data = extract_image_data(img)
            if img_data:
                images_data[img.name] = img_data
            else:
                broken_count += 1
        except Exception as e:
            print(f"  Warning: Failed to extract image {img.name}: {e}")
            broken_count += 1
    if images_data:
        scene_data["images"] = images_data
    if broken_count > 0:
        print(f"  Skipped {broken_count} broken/missing images")

    # ── Standalone node groups (reusable shader/geometry setups) ──
    # These exist in bpy.data.node_groups but may not be on any object
    node_groups = []
    for ng in bpy.data.node_groups:
        try:
            ng_data = extract_node_group_data(ng)
            if ng_data:
                node_groups.append(ng_data)
        except Exception as e:
            print(f"  Warning: Failed to extract node group {ng.name}: {e}")
    if node_groups:
        scene_data["node_groups"] = node_groups

    # ── Orphan materials (not assigned to any scene object) ──
    used_mats = set()
    for obj in scene.objects:
        for slot in obj.material_slots:
            if slot.material:
                used_mats.add(slot.material.name)
    orphan_mats = []
    for mat in bpy.data.materials:
        if mat.name not in used_mats:
            try:
                mat_data = extract_material_data(mat)
                if mat_data:
                    orphan_mats.append(mat_data)
            except Exception:
                pass
    if orphan_mats:
        scene_data["orphan_materials"] = orphan_mats

    # ── Orphan actions (animations not currently assigned) ──
    used_actions = set()
    for obj in scene.objects:
        if obj.animation_data and obj.animation_data.action:
            used_actions.add(obj.animation_data.action.name)
    orphan_actions = []
    for action in bpy.data.actions:
        if action.name not in used_actions:
            act_data = {
                "name": action.name,
                "frame_range": [round(action.frame_range[0], 2),
                                round(action.frame_range[1], 2)],
                "fcurve_count": len(action.fcurves),
            }
            # Sample a few fcurves to give the AI a sense of the animation
            curves = []
            for fc in action.fcurves[:20]:
                curve_data = {
                    "data_path": fc.data_path,
                    "array_index": fc.array_index,
                    "keyframe_count": len(fc.keyframe_points),
                }
                # Sample first/last keyframe values
                if fc.keyframe_points:
                    curve_data["first_keyframe"] = {
                        "frame": round(fc.keyframe_points[0].co[0], 2),
                        "value": round(fc.keyframe_points[0].co[1], 4),
                    }
                    curve_data["last_keyframe"] = {
                        "frame": round(fc.keyframe_points[-1].co[0], 2),
                        "value": round(fc.keyframe_points[-1].co[1], 4),
                    }
                curves.append(curve_data)
            if curves:
                act_data["fcurves"] = curves
            orphan_actions.append(act_data)
    if orphan_actions:
        scene_data["orphan_actions"] = orphan_actions

    return scene_data


def process_blend_file(blend_path: str, output_dir: str,
                       config: dict) -> bool:
    """Process a single .blend/.glb/.gltf/.obj/.fbx file and save extracted data."""
    blend_path = Path(blend_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = blend_path.stem
    output_file = output_dir / f"{stem}.json"

    if output_file.exists():
        return True

    ext = blend_path.suffix.lower()

    try:
        if ext == '.blend':
            bpy.ops.wm.open_mainfile(filepath=str(blend_path))
        else:
            bpy.ops.wm.read_factory_settings(use_empty=True)
            for obj in list(bpy.data.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
            if ext in ('.glb', '.gltf'):
                bpy.ops.import_scene.gltf(filepath=str(blend_path))
            elif ext == '.obj':
                bpy.ops.wm.obj_import(filepath=str(blend_path))
            elif ext == '.fbx':
                bpy.ops.import_scene.fbx(filepath=str(blend_path))
            else:
                print(f"  Unsupported format: {ext}")
                return False

        print(f"Processing: {blend_path.name}")

        # ── Resolve linked/external data ──
        # Many .blend files reference external textures, materials, and
        # linked libraries. Without packing, these get lost during extraction.
        try:
            # Make all file paths absolute so Blender can find them
            bpy.ops.file.make_paths_absolute()
        except Exception:
            pass

        # Remap any images whose stored filepath doesn't resolve —
        # searches textures/, tex/, same dir, and parent variations.
        # Must happen BEFORE pack_all so Blender can find and pack them.
        try:
            _remap_missing_image_paths(blend_path.parent)
        except Exception as _remap_err:
            print(f"  Note: texture remap pass failed: {_remap_err}")

        try:
            # Pack all external files (textures, images, fonts) into the blend
            # This ensures materials with image textures are fully captured
            bpy.ops.file.pack_all()
        except Exception:
            pass

        # Resolve linked library objects — make them local so we can
        # extract their mesh data (linked objects are read-only otherwise)
        try:
            for obj in list(bpy.data.objects):
                if obj.library is not None:
                    # Make the linked object local (editable)
                    obj.make_local()
            # Also make linked materials, meshes, and node groups local
            for mat in list(bpy.data.materials):
                if mat.library is not None:
                    mat.make_local()
            for mesh in list(bpy.data.meshes):
                if mesh.library is not None:
                    mesh.make_local()
            for ng in list(bpy.data.node_groups):
                if ng.library is not None:
                    ng.make_local()
        except Exception as e:
            print(f"  Note: Could not localize all linked data: {e}")

        # Force update the dependency graph after making things local
        bpy.context.view_layer.update()

        # Extract all data
        scene_data = extract_scene_data(config)
        scene_data["source_file"] = str(blend_path)
        scene_data["blender_version"] = list(bpy.app.version)

        if scene_data["object_count"] == 0:
            print(f"  Skipping {blend_path.name}: no valid objects")
            return False

        # ── Build rich text labels ──
        import re

        # Try to load metadata from .meta.json (SmutBase, BlendSwap, etc.)
        meta_label = None
        meta_tags = []
        meta_description = ""
        for meta_path in [
            blend_path.with_suffix(".meta.json"),
            blend_path.parent / f"{blend_path.stem}.meta.json",
            blend_path.parent.parent / "metadata" / f"{blend_path.stem}.meta.json",
        ]:
            if meta_path.exists():
                try:
                    with open(meta_path) as mf:
                        meta = json.load(mf)
                    meta_label = meta.get("title") or meta.get("name", "")
                    meta_tags = meta.get("tags", [])
                    if isinstance(meta_tags, dict):
                        meta_tags = list(meta_tags.keys())
                    meta_description = meta.get("description", "")
                    break
                except Exception:
                    pass

        # Also check sibling metadata dir (BlendSwap stores as <id>.meta.json,
        # SmutBase stores as <uuid>.meta.json with filename field inside)
        if not meta_label:
            meta_dir = blend_path.parent.parent / "metadata"
            if meta_dir.is_dir():
                blend_fname = blend_path.name  # e.g. "Hornet_mvWIzMY.blend"
                for mf_path in meta_dir.glob("*.meta.json"):
                    try:
                        with open(mf_path) as mf:
                            meta = json.load(mf)
                        # Match by blend_id/id == stem (BlendSwap)
                        bid = meta.get("blend_id") or meta.get("id", "")
                        # Match by filename field (SmutBase)
                        meta_fname = meta.get("filename", "")
                        if str(bid) == blend_path.stem or meta_fname == blend_fname:
                            meta_label = meta.get("title") or meta.get("name", "")
                            meta_tags = meta.get("tags", [])
                            if isinstance(meta_tags, dict):
                                meta_tags = list(meta_tags.keys())
                            meta_description = meta.get("description", "")
                            break
                    except Exception:
                        continue

        # Fallback: derive label from filename
        if not meta_label:
            stem_str = blend_path.stem
            meta_label = re.sub(r'([a-z])([A-Z])', r'\1 \2', stem_str)
            meta_label = meta_label.replace('_', ' ').replace('-', ' ')
            meta_label = re.sub(r'\s+', ' ', meta_label).strip()

        # Inject labels into scene and each object
        # Scene-level: keep file-level metadata for downstream context
        scene_data["text_label"] = meta_label
        if meta_tags:
            scene_data["tags"] = meta_tags[:20]
        if meta_description:
            scene_data["description"] = meta_description[:500]

        # Per-object: each object gets its OWN identity label, plus
        # file_label as context.  This prevents a "barrel.blend" file
        # from labeling its ground plane as "barrel".
        for obj_data in scene_data.get("objects", []):
            obj_name = obj_data.get("name", "")
            # Always preserve file-level label for downstream enrichment
            obj_data["file_label"] = meta_label
            if meta_tags:
                obj_data["file_tags"] = meta_tags[:20]
            # Use the object's own Blender name as its primary label.
            # Downstream smart labeling (generate_smart_label) will:
            #  - Enrich bare primitive names via Strategy 0
            #  - Use file_label as context in Strategy 2/3
            #  - Fall back to Qwen if available
            obj_data["text_label"] = obj_name

        # Save (compact JSON — indent=2 was inflating files 3-4x)
        with open(output_file, "w") as f:
            json.dump(scene_data, f, separators=(',', ':'))

        print(f"  Extracted {scene_data['object_count']} objects → {output_file.name}")
        return True

    except Exception as e:
        print(f"  ERROR processing {blend_path.name}: {e}")
        traceback.print_exc()
        return False


def main():
    # Parse args after "--" separator (Blender passes its own args before)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Extract data from .blend files")
    parser.add_argument("--input", required=True,
                        help="Input directory with .blend files")
    parser.add_argument("--output", required=True,
                        help="Output directory for JSON files")
    parser.add_argument("--config", default=None,
                        help="Path to config.yaml (default: auto-detect)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max files to process")
    args = parser.parse_args(argv)

    # Load config
    config = {}
    config_path = args.config or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config.yaml"
    )
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
        except ImportError:
            pass

    # Find all supported 3D files
    SUPPORTED_EXTS = {'.blend', '.glb', '.gltf', '.obj', '.fbx'}
    input_dir = Path(args.input)
    if input_dir.is_file() and input_dir.suffix.lower() in SUPPORTED_EXTS:
        blend_files = [input_dir]
    else:
        blend_files = sorted(
            f for f in input_dir.rglob("*")
            if f.suffix.lower() in SUPPORTED_EXTS
            and not f.name.startswith("._")
        )
    print(f"Found {len(blend_files)} 3D files in {input_dir}")

    if args.limit:
        blend_files = blend_files[:args.limit]

    # Process each file
    success_count = 0
    for i, bf in enumerate(blend_files):
        print(f"\n[{i+1}/{len(blend_files)}]")
        if process_blend_file(bf, args.output, config):
            success_count += 1

    print(f"\nDone! Extracted {success_count}/{len(blend_files)} files.")


if __name__ == "__main__":
    main()
