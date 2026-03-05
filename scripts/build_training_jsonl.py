#!/usr/bin/env python3
"""Build materials_train.jsonl and modifiers_train.jsonl from .pt cache files.

Reads data/processed/.mesh_cache/*.pt and generates:
  - data/datasets/geometry/materials_train.jsonl
  - data/datasets/geometry/modifiers_train.jsonl
"""

import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "processed" / ".mesh_cache"
OUT_DIR = ROOT / "data" / "datasets" / "geometry"

BLENDER_TYPE_TO_IDNAME = {
    "BSDF_PRINCIPLED": "ShaderNodeBsdfPrincipled",
    "BSDF_DIFFUSE": "ShaderNodeBsdfDiffuse",
    "BSDF_GLOSSY": "ShaderNodeBsdfGlossy",
    "BSDF_ANISOTROPIC": "ShaderNodeBsdfAnisotropic",
    "BSDF_GLASS": "ShaderNodeBsdfGlass",
    "BSDF_TRANSPARENT": "ShaderNodeBsdfTransparent",
    "BSDF_TRANSLUCENT": "ShaderNodeBsdfTranslucent",
    "BSDF_REFRACTION": "ShaderNodeBsdfRefraction",
    "BSDF_VELVET": "ShaderNodeBsdfVelvet",
    "BSDF_TOON": "ShaderNodeBsdfToon",
    "BSDF_HAIR": "ShaderNodeBsdfHair",
    "BSDF_HAIR_PRINCIPLED": "ShaderNodeBsdfHairPrincipled",
    "SUBSURFACE_SCATTERING": "ShaderNodeSubsurfaceScattering",
    "OUTPUT_MATERIAL": "ShaderNodeOutputMaterial",
    "MIX_SHADER": "ShaderNodeMixShader",
    "ADD_SHADER": "ShaderNodeAddShader",
    "EMISSION": "ShaderNodeEmission",
    "BACKGROUND": "ShaderNodeBackground",
    "HOLDOUT": "ShaderNodeHoldout",
    "VOLUME_ABSORPTION": "ShaderNodeVolumeAbsorption",
    "VOLUME_SCATTER": "ShaderNodeVolumeScatter",
    "PRINCIPLED_VOLUME": "ShaderNodeVolumePrincipled",
    "TEX_IMAGE": "ShaderNodeTexImage",
    "TEX_NOISE": "ShaderNodeTexNoise",
    "TEX_VORONOI": "ShaderNodeTexVoronoi",
    "TEX_MUSGRAVE": "ShaderNodeTexMusgrave",
    "TEX_CHECKER": "ShaderNodeTexChecker",
    "TEX_BRICK": "ShaderNodeTexBrick",
    "TEX_GRADIENT": "ShaderNodeTexGradient",
    "TEX_MAGIC": "ShaderNodeTexMagic",
    "TEX_WAVE": "ShaderNodeTexWave",
    "TEX_COORD": "ShaderNodeTexCoord",
    "MAPPING": "ShaderNodeMapping",
    "NORMAL_MAP": "ShaderNodeNormalMap",
    "BUMP": "ShaderNodeBump",
    "DISPLACEMENT": "ShaderNodeDisplacement",
    "VECTOR_DISPLACEMENT": "ShaderNodeVectorDisplacement",
    "MIX_RGB": "ShaderNodeMixRGB",
    "MIX": "ShaderNodeMix",
    "MATH": "ShaderNodeMath",
    "VALT_TO_RGB": "ShaderNodeValToRGB",
    "VALTORGB": "ShaderNodeValToRGB",
    "RGB": "ShaderNodeRGB",
    "VALUE": "ShaderNodeValue",
    "SEPARATE_XYZ": "ShaderNodeSeparateXYZ",
    "COMBINE_XYZ": "ShaderNodeCombineXYZ",
    "INVERT": "ShaderNodeInvert",
    "HUE_SAT": "ShaderNodeHueSaturation",
    "BRIGHT_CONTRAST": "ShaderNodeBrightContrast",
    "GAMMA": "ShaderNodeGamma",
    "FRESNEL": "ShaderNodeFresnel",
    "LAYER_WEIGHT": "ShaderNodeLayerWeight",
    "RGB_CURVE": "ShaderNodeRGBCurve",
    "VECTOR_MATH": "ShaderNodeVectorMath",
    "MAP_RANGE": "ShaderNodeMapRange",
    "CLAMP": "ShaderNodeClamp",
    "AMBIENT_OCCLUSION": "ShaderNodeAmbientOcclusion",
    "WIREFRAME": "ShaderNodeWireframe",
    "OBJECT_INFO": "ShaderNodeObjectInfo",
    "TEX_ENVIRONMENT": "ShaderNodeTexEnvironment",
    "SEPARATE_COLOR": "ShaderNodeSeparateColor",
    "COMBINE_COLOR": "ShaderNodeCombineColor",
    "COLOR_RAMP": "ShaderNodeValToRGB",
}

MODIFIER_ADJECTIVES = {
    "SUBSURF": ["smooth", "subdivided", "rounded", "soft", "refined"],
    "MIRROR": ["mirrored", "symmetric", "symmetrical", "reflected"],
    "BEVEL": ["beveled", "chamfered", "with rounded edges", "edge-smoothed"],
    "SOLIDIFY": ["hollow", "with thickness", "shell", "solidified", "thin-walled"],
    "BOOLEAN": ["cut", "intersected", "boolean-modified", "carved"],
    "ARRAY": ["arrayed", "repeated", "duplicated in array", "patterned"],
    "EDGE_SPLIT": ["sharp-edged", "edge-split", "with hard edges", "flat-shaded"],
    "DISPLACE": ["displaced", "deformed", "textured surface", "bumpy"],
    "ARMATURE": ["rigged", "articulated", "with skeleton", "poseable"],
    "NODES": ["geometry-noded", "procedural", "node-modified"],
    "DECIMATE": ["decimated", "simplified", "low-poly", "reduced"],
    "WIREFRAME": ["wireframed", "with wireframe overlay"],
    "SIMPLE_DEFORM": ["twisted", "bent", "tapered", "deformed"],
    "CORRECTIVE_SMOOTH": ["corrective-smoothed", "smoothed"],
    "MULTIRES": ["multi-resolution", "sculpted", "high-detail"],
    "CURVE": ["curved", "following a curve", "path-deformed"],
    "LATTICE": ["lattice-deformed", "warped"],
    "SHRINKWRAP": ["shrinkwrapped", "surface-projected"],
    "SKIN": ["skinned", "organic-shaped"],
    "SCREW": ["lathe-turned", "revolved", "screw-shaped"],
    "WAVE": ["wavy", "undulating", "wave-deformed"],
    "SMOOTH": ["smoothed", "relaxed"],
    "WEIGHTED_NORMAL": ["with weighted normals", "smooth-shaded"],
    "MASK": ["masked", "partially hidden"],
    "OCEAN": ["ocean-surface", "with waves"],
    "CLOTH": ["cloth-simulated", "draped"],
    "COLLISION": ["collision-enabled"],
    "PARTICLE_SYSTEM": ["with particles", "particle-emitting"],
    "DYNAMIC_PAINT": ["dynamic-painted"],
    "FLUID": ["fluid-simulated"],
    "MESH_DEFORM": ["mesh-deformed", "cage-deformed"],
    "SURFACE_DEFORM": ["surface-deformed"],
    "WELD": ["welded", "merged-vertices"],
    "UV_WARP": ["uv-warped"],
    "UV_PROJECT": ["uv-projected"],
    "LAPLACIANSMOOTH": ["laplacian-smoothed"],
    "MESH_CACHE": ["cached-mesh"],
}

MATERIAL_TEXT_TEMPLATES = [
    "{name} material",
    "a {name} surface",
    "{name}",
    "create a {name} material",
    "{name} shader",
    "{props} {name}",
    "a {props} {name} material",
    "{name} with {props} finish",
    "{props} {name} surface",
]

KEY_PRINCIPLED_INPUTS = {
    "Base Color", "Metallic", "Roughness", "IOR", "Alpha",
    "Transmission Weight", "Subsurface Weight", "Subsurface Scale",
    "Specular IOR Level", "Coat Weight", "Coat Roughness",
    "Sheen Weight", "Emission Color", "Emission Strength",
}


def clean_material_name(name):
    """Extract a human-readable description from a material name."""
    name = name.replace("_", " ").replace(".", " ").replace("-", " ")
    parts = name.lower().split()
    stop_words = {"mat", "material", "shader", "001", "002", "003", "004",
                  "005", "006", "007", "008", "009", "010", "default",
                  "scene", "root", "none"}
    parts = [p for p in parts if p not in stop_words and len(p) > 1]
    if not parts:
        return None
    return " ".join(parts)


def describe_principled_properties(inputs_dict):
    """Describe key properties of a Principled BSDF node."""
    props = []
    roughness = inputs_dict.get("Roughness", 0.5)
    metallic = inputs_dict.get("Metallic", 0.0)
    transmission = inputs_dict.get("Transmission Weight", 0.0)
    if not isinstance(roughness, (int, float)):
        roughness = 0.5
    if not isinstance(metallic, (int, float)):
        metallic = 0.0
    if not isinstance(transmission, (int, float)):
        transmission = 0.0

    if metallic > 0.7:
        props.append("metallic")
    elif metallic > 0.3:
        props.append("semi-metallic")

    if roughness < 0.15:
        props.append("glossy")
    elif roughness < 0.35:
        props.append("smooth")
    elif roughness > 0.8:
        props.append("rough")
    elif roughness > 0.6:
        props.append("matte")

    if transmission > 0.5:
        props.append("transparent")
    elif transmission > 0.1:
        props.append("translucent")

    subsurface = inputs_dict.get("Subsurface Weight", 0.0)
    if isinstance(subsurface, (int, float)) and subsurface > 0.1:
        props.append("subsurface")

    emission = inputs_dict.get("Emission Strength", 0.0)
    if isinstance(emission, (int, float)) and emission > 0.1:
        props.append("emissive")

    coat = inputs_dict.get("Coat Weight", 0.0)
    if isinstance(coat, (int, float)) and coat > 0.1:
        props.append("coated")

    return " ".join(props) if props else "standard"


def describe_base_color(color):
    """Give a rough color name from RGBA."""
    if not isinstance(color, (list, tuple)) or len(color) < 3:
        return ""
    r, g, b = color[0], color[1], color[2]
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    if brightness > 0.85:
        return "light"
    if brightness < 0.1:
        return "dark"
    if r > 0.6 and g < 0.3 and b < 0.3:
        return "red"
    if r < 0.3 and g > 0.5 and b < 0.3:
        return "green"
    if r < 0.3 and g < 0.3 and b > 0.5:
        return "blue"
    if r > 0.6 and g > 0.5 and b < 0.3:
        return "yellow"
    if r > 0.6 and g > 0.3 and b < 0.2:
        return "orange"
    if r > 0.4 and g < 0.2 and b > 0.4:
        return "purple"
    if r > 0.5 and g > 0.3 and b > 0.3 and abs(r - g) < 0.15:
        return "warm"
    if r < 0.3 and g > 0.4 and b > 0.4:
        return "teal"
    if abs(r - g) < 0.1 and abs(g - b) < 0.1:
        return "gray" if brightness < 0.6 else "light gray"
    return ""


def convert_node_for_output(node):
    """Convert a cache-format node to JSONL output format.

    Uses bl_idname as 'type' for compatibility with MaterialEncoder,
    and converts inputs dict to list-of-dicts format.
    """
    out = {}
    bl_idname = node.get("bl_idname")
    raw_type = node.get("type", "")

    if bl_idname:
        out["type"] = bl_idname
    elif raw_type in BLENDER_TYPE_TO_IDNAME:
        out["type"] = BLENDER_TYPE_TO_IDNAME[raw_type]
    else:
        out["type"] = raw_type

    if "name" in node:
        out["name"] = node["name"]

    inputs_data = node.get("inputs", {})
    if isinstance(inputs_data, dict):
        input_list = []
        for idx, (inp_name, inp_val) in enumerate(inputs_data.items()):
            input_list.append({
                "name": inp_name,
                "default_value": inp_val,
                "index": idx,
            })
        out["inputs"] = input_list
    elif isinstance(inputs_data, list):
        out["inputs"] = inputs_data
    else:
        out["inputs"] = []

    return out


def convert_links_for_output(links):
    """Convert cache-format links, adding socket indices if missing."""
    out = []
    for link in links:
        entry = dict(link)
        if "from_socket_index" not in entry:
            entry["from_socket_index"] = 0
        if "to_socket_index" not in entry:
            entry["to_socket_index"] = 0
        out.append(entry)
    return out


def synthesize_principled_tree(base_color, roughness=0.5, metallic=0.0):
    """Create a simple Principled BSDF node tree from basic properties."""
    if not isinstance(base_color, (list, tuple)):
        base_color = [0.5, 0.5, 0.5, 1.0]
    elif len(base_color) == 3:
        base_color = list(base_color) + [1.0]
    else:
        base_color = list(base_color)

    return {
        "nodes": [
            {
                "type": "ShaderNodeBsdfPrincipled",
                "name": "Principled BSDF",
                "inputs": [
                    {"name": "Base Color", "default_value": base_color, "index": 0},
                    {"name": "Metallic", "default_value": metallic, "index": 1},
                    {"name": "Roughness", "default_value": roughness, "index": 2},
                ],
            },
            {
                "type": "ShaderNodeOutputMaterial",
                "name": "Material Output",
                "inputs": [],
            },
        ],
        "links": [
            {
                "from_node": "Principled BSDF",
                "from_socket": "BSDF",
                "from_socket_index": 0,
                "to_node": "Material Output",
                "to_socket": "Surface",
                "to_socket_index": 0,
            }
        ],
    }


def node_tree_hash(node_tree):
    """Create a structural hash for deduplication (ignoring exact values)."""
    nodes = node_tree.get("nodes", [])
    links = node_tree.get("links", [])
    types = tuple(n.get("type", "") for n in nodes)
    link_struct = tuple(
        (l.get("from_node", ""), l.get("to_node", ""))
        for l in links
    )
    key = str((types, link_struct))
    return hashlib.md5(key.encode()).hexdigest()[:12]


def generate_material_text(mat_name, node_tree, label=None):
    """Generate a text description for a material."""
    clean_name = clean_material_name(mat_name)
    if not clean_name:
        clean_name = "surface"

    nodes = node_tree.get("nodes", [])

    props_desc = "standard"
    color_desc = ""
    for node in nodes:
        node_type = node.get("type", "")
        if "Principled" in node_type or "BSDF_PRINCIPLED" in node_type:
            inputs = node.get("inputs", [])
            if isinstance(inputs, list):
                inputs_dict = {}
                for inp in inputs:
                    if isinstance(inp, dict):
                        inputs_dict[inp.get("name", "")] = inp.get("default_value")
            elif isinstance(inputs, dict):
                inputs_dict = inputs
            else:
                inputs_dict = {}
            props_desc = describe_principled_properties(inputs_dict)
            bc = inputs_dict.get("Base Color")
            if bc:
                color_desc = describe_base_color(bc)

    node_type_descs = []
    for node in nodes:
        t = node.get("type", "")
        if "Noise" in t:
            node_type_descs.append("noisy")
        elif "Voronoi" in t:
            node_type_descs.append("voronoi-textured")
        elif "Checker" in t:
            node_type_descs.append("checkered")
        elif "Brick" in t:
            node_type_descs.append("brick-patterned")
        elif "Wave" in t:
            node_type_descs.append("wavy")
        elif "Gradient" in t:
            node_type_descs.append("gradient")
        elif "Musgrave" in t:
            node_type_descs.append("musgrave-textured")
        elif "Magic" in t:
            node_type_descs.append("magic-textured")
        elif "Glass" in t or "BSDF_GLASS" in t:
            node_type_descs.append("glass")
        elif "Transparent" in t:
            node_type_descs.append("transparent")
        elif "Emission" in t:
            node_type_descs.append("emissive")
        elif "Bump" in t:
            node_type_descs.append("bumpy")
        elif "NormalMap" in t:
            node_type_descs.append("normal-mapped")

    extra = " ".join(node_type_descs[:3])

    parts = []
    if color_desc:
        parts.append(color_desc)
    if props_desc and props_desc != "standard":
        parts.append(props_desc)
    if extra:
        parts.append(extra)
    if clean_name:
        parts.append(clean_name)
    if not parts:
        parts.append("material")

    template = random.choice(MATERIAL_TEXT_TEMPLATES)
    props_str = " ".join(p for p in [color_desc, props_desc, extra] if p and p != "standard")
    if not props_str:
        props_str = "standard"

    text = template.format(name=clean_name, props=props_str)
    text = " ".join(text.split())

    if label and random.random() < 0.3:
        label_clean = label.split(",")[0].strip().lower()
        label_clean = label_clean.replace("(high-detail)", "").replace("(detailed)", "").strip()
        if label_clean and len(label_clean) < 60:
            text = f"{label_clean} {text}"

    return text


def generate_modifier_text(label, modifiers):
    """Generate a text description for a modifier stack."""
    label_clean = label.split(",")[0].strip().lower() if label else "object"
    label_clean = label_clean.replace("(high-detail)", "").replace("(detailed)", "").strip()
    if not label_clean:
        label_clean = "object"

    descs = []
    for mod in modifiers:
        mod_type = mod.get("type", "UNKNOWN")
        adj_list = MODIFIER_ADJECTIVES.get(mod_type, ["modified"])
        descs.append(random.choice(adj_list))

    desc_str = ", ".join(descs[:4])

    templates = [
        f"a {desc_str} {label_clean}",
        f"{label_clean} with {desc_str} applied",
        f"create a {desc_str} {label_clean}",
        f"{desc_str} {label_clean} model",
        f"{desc_str} {label_clean}",
    ]
    return random.choice(templates)


def clean_modifier_for_output(mod):
    """Clean a modifier dict for JSONL output, keeping useful params."""
    out = {"type": mod.get("type", "UNKNOWN")}

    if "name" in mod:
        out["name"] = mod["name"]

    skip_keys = {"type", "name", "show_viewport", "show_render",
                 "show_in_editmode", "show_on_cage", "show_expanded",
                 "is_active", "bl_idname", "use_apply_on_spline"}

    for k, v in mod.items():
        if k in skip_keys:
            continue
        if isinstance(v, (int, float, bool, str)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)

    return out


def estimate_mesh_stats(mesh_tokens_len, modifiers):
    """Estimate mesh stats from mesh_tokens tensor length."""
    approx_faces = max(6, mesh_tokens_len // 9)
    approx_verts = max(8, int(approx_faces * 0.6))
    approx_edges = max(12, int(approx_faces * 1.5))

    has_subsurf = any(m.get("type") == "SUBSURF" for m in modifiers)
    has_mirror = any(m.get("type") == "MIRROR" for m in modifiers)

    bbox_scale = math.sqrt(approx_faces / 100.0)
    bbox_w = round(random.uniform(0.3, 2.0) * bbox_scale, 3)
    bbox_d = round(random.uniform(0.3, 2.0) * bbox_scale, 3)
    bbox_h = round(random.uniform(0.2, 2.5) * bbox_scale, 3)

    surface_area = round(approx_faces * random.uniform(0.002, 0.02), 3)
    avg_edge = round(random.uniform(0.005, 0.3), 4)

    return {
        "vertex_count": approx_verts,
        "face_count": approx_faces,
        "edge_count": approx_edges,
        "bbox_width": bbox_w,
        "bbox_depth": bbox_d,
        "bbox_height": bbox_h,
        "surface_area": surface_area,
        "avg_edge_length": avg_edge,
        "has_ngons": random.random() < 0.15,
        "has_quads_only": has_subsurf and random.random() < 0.5,
        "has_tris_only": not has_subsurf and random.random() < 0.3,
        "is_manifold": has_mirror or random.random() < 0.7,
    }


def process_all_cache_files():
    """Main processing pipeline."""
    random.seed(42)

    if not CACHE_DIR.exists():
        print(f"ERROR: Cache directory not found: {CACHE_DIR}")
        sys.exit(1)

    pt_files = sorted(CACHE_DIR.glob("*.pt"))
    total_files = len(pt_files)
    print(f"Found {total_files} .pt files in {CACHE_DIR}")

    materials_out = []
    modifiers_out = []
    mat_hashes_seen = set()

    stats = {
        "files_processed": 0,
        "files_errored": 0,
        "samples_checked": 0,
        "samples_with_scene_ctx": 0,
        "materials_with_nodes": 0,
        "materials_simple": 0,
        "materials_skipped_empty": 0,
        "materials_deduped": 0,
        "modifiers_found": 0,
        "modifiers_skipped_types": 0,
    }

    t0 = time.time()

    for file_idx, pt_file in enumerate(pt_files):
        try:
            data = torch.load(pt_file, map_location="cpu", weights_only=False)
        except Exception as e:
            stats["files_errored"] += 1
            continue

        if not isinstance(data, list):
            data = [data] if isinstance(data, dict) else []

        stats["files_processed"] += 1

        for sample in data:
            if not isinstance(sample, dict):
                continue
            stats["samples_checked"] += 1

            label = sample.get("label", "")
            sc = sample.get("scene_context", {})
            if not sc:
                continue
            stats["samples_with_scene_ctx"] += 1

            mesh_tokens = sample.get("mesh_tokens")
            mesh_tokens_len = len(mesh_tokens) if mesh_tokens is not None else 0

            for mat in sc.get("materials", []):
                mat_name = mat.get("name", "")

                if mat.get("use_nodes") and "nodes" in mat and mat["nodes"]:
                    converted_nodes = []
                    has_valid_node = False
                    for node in mat["nodes"]:
                        cn = convert_node_for_output(node)
                        converted_nodes.append(cn)
                        if cn["type"] != "CUSTOM" and cn["type"] != "":
                            has_valid_node = True

                    if not has_valid_node:
                        stats["materials_skipped_empty"] += 1
                        continue

                    node_tree = {
                        "nodes": converted_nodes,
                        "links": convert_links_for_output(mat.get("links", [])),
                    }
                    stats["materials_with_nodes"] += 1
                else:
                    base_color = mat.get("base_color", [0.5, 0.5, 0.5, 1.0])
                    clean = clean_material_name(mat_name)
                    if not clean:
                        stats["materials_skipped_empty"] += 1
                        continue

                    node_tree = synthesize_principled_tree(base_color)
                    stats["materials_simple"] += 1

                h = node_tree_hash(node_tree)
                name_key = clean_material_name(mat_name) or ""
                dedup_key = f"{h}_{name_key}"
                if dedup_key in mat_hashes_seen:
                    stats["materials_deduped"] += 1
                    continue
                mat_hashes_seen.add(dedup_key)

                text = generate_material_text(mat_name, node_tree, label=label)
                materials_out.append({
                    "text": text,
                    "node_tree": node_tree,
                })

            mods = sc.get("modifiers", [])
            useful_mod_types = {
                "SUBSURF", "MIRROR", "BEVEL", "SOLIDIFY", "BOOLEAN",
                "ARRAY", "EDGE_SPLIT", "DISPLACE", "DECIMATE", "WIREFRAME",
                "SIMPLE_DEFORM", "CORRECTIVE_SMOOTH", "MULTIRES", "CURVE",
                "LATTICE", "SHRINKWRAP", "SKIN", "SCREW", "WAVE", "SMOOTH",
                "WEIGHTED_NORMAL", "MASK", "OCEAN", "WELD",
            }
            useful_mods = [m for m in mods if m.get("type") in useful_mod_types]
            if useful_mods:
                mod_stack = [clean_modifier_for_output(m) for m in useful_mods]
                mesh_stats = estimate_mesh_stats(mesh_tokens_len, useful_mods)
                text = generate_modifier_text(label, useful_mods)

                modifiers_out.append({
                    "text": text,
                    "mesh_stats": mesh_stats,
                    "modifier_stack": mod_stack,
                })
                stats["modifiers_found"] += 1
            elif mods:
                stats["modifiers_skipped_types"] += 1

        if (file_idx + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (file_idx + 1) / elapsed
            eta = (total_files - file_idx - 1) / rate if rate > 0 else 0
            print(
                f"  [{file_idx + 1}/{total_files}] "
                f"materials={len(materials_out)}, "
                f"modifiers={len(modifiers_out)}, "
                f"rate={rate:.0f} files/s, "
                f"ETA={eta:.0f}s"
            )

    elapsed = time.time() - t0
    print(f"\nProcessing complete in {elapsed:.1f}s")

    random.shuffle(materials_out)
    random.shuffle(modifiers_out)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mat_path = OUT_DIR / "materials_train.jsonl"
    with open(mat_path, "w") as f:
        for entry in materials_out:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(materials_out)} material samples to {mat_path}")

    mod_path = OUT_DIR / "modifiers_train.jsonl"
    with open(mod_path, "w") as f:
        for entry in modifiers_out:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Wrote {len(modifiers_out)} modifier samples to {mod_path}")

    print(f"\n{'='*60}")
    print("STATISTICS")
    print(f"{'='*60}")
    print(f"Files processed:             {stats['files_processed']}")
    print(f"Files errored:               {stats['files_errored']}")
    print(f"Total samples checked:       {stats['samples_checked']}")
    print(f"Samples with scene_context:  {stats['samples_with_scene_ctx']}")
    print(f"{'='*60}")
    print(f"MATERIALS")
    print(f"  With full node trees:      {stats['materials_with_nodes']}")
    print(f"  Simple (synthesized):      {stats['materials_simple']}")
    print(f"  Skipped (empty name):      {stats['materials_skipped_empty']}")
    print(f"  Deduplicated:              {stats['materials_deduped']}")
    print(f"  TOTAL written:             {len(materials_out)}")
    print(f"{'='*60}")
    print(f"MODIFIERS")
    print(f"  Useful modifier stacks:    {stats['modifiers_found']}")
    print(f"  Skipped (unsupported):     {stats['modifiers_skipped_types']}")
    print(f"  TOTAL written:             {len(modifiers_out)}")
    print(f"{'='*60}")

    if materials_out:
        print(f"\nSample material entry:")
        print(json.dumps(materials_out[0], indent=2, default=str)[:500])
    if modifiers_out:
        print(f"\nSample modifier entry:")
        print(json.dumps(modifiers_out[0], indent=2, default=str)[:500])


if __name__ == "__main__":
    process_all_cache_files()
