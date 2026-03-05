"""Deterministic procedural displacement helpers for Blender headless workers.

This module keeps the procedural workflow compact and action-friendly:
- one canonical node group: ``PROC_HEIGHT_BASIC``
- bucketed parameter mapping for low-entropy control
- deterministic randomization via explicit seed
"""

from __future__ import annotations

from dataclasses import dataclass

import bpy  # type: ignore


@dataclass(frozen=True)
class ProceduralNoiseParams:
    seed: int
    scale: float
    detail: float
    roughness: float
    distortion: float
    strength: float
    midlevel: float


def _bucket_to_range(bucket: int, low: float, high: float, bins: int = 32) -> float:
    b = max(0, min(int(bucket), int(bins) - 1))
    t = float(b) / float(max(1, int(bins) - 1))
    return float(low + (high - low) * t)


def params_from_buckets(
    *,
    seed: int,
    scale_bucket: int,
    detail_bucket: int,
    roughness_bucket: int,
    distortion_bucket: int,
    strength_bucket: int,
    midlevel_bucket: int,
) -> ProceduralNoiseParams:
    return ProceduralNoiseParams(
        seed=int(seed),
        scale=_bucket_to_range(scale_bucket, 0.5, 18.0),
        detail=_bucket_to_range(detail_bucket, 0.0, 15.0),
        roughness=_bucket_to_range(roughness_bucket, 0.0, 1.0),
        distortion=_bucket_to_range(distortion_bucket, 0.0, 1.0),
        strength=_bucket_to_range(strength_bucket, 0.02, 0.8),
        midlevel=_bucket_to_range(midlevel_bucket, 0.0, 1.0),
    )


def ensure_proc_height_basic_group() -> object:
    """Create/reuse the canonical procedural height node group.

    Group interface:
    - Input: ``Vector``
    - Output: ``Height``
    Internal node: Noise Texture → ColorRamp → output
    """
    name = "PROC_HEIGHT_BASIC"
    existing = bpy.data.node_groups.get(name)
    if existing is not None:
        return existing

    group = bpy.data.node_groups.new(name=name, type="ShaderNodeTree")
    nodes = group.nodes
    links = group.links

    in_node = nodes.new("NodeGroupInput")
    out_node = nodes.new("NodeGroupOutput")
    in_node.location = (-600, 0)
    out_node.location = (220, 0)

    group.interface.new_socket(name="Vector", in_out="INPUT", socket_type="NodeSocketVector")
    group.interface.new_socket(name="Height", in_out="OUTPUT", socket_type="NodeSocketFloat")

    noise = nodes.new("ShaderNodeTexNoise")
    noise.name = "Noise"
    noise.label = "Noise"
    noise.location = (-300, 0)
    noise.noise_dimensions = "3D"

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.name = "HeightRamp"
    ramp.label = "HeightRamp"
    ramp.location = (-20, 0)
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[1].position = 0.75

    links.new(in_node.outputs["Vector"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Fac"], out_node.inputs["Height"])

    return group


def _ensure_displace_modifier(obj) -> object:
    for mod in obj.modifiers:
        if mod.type == "DISPLACE":
            return mod
    return obj.modifiers.new(name="Displace", type="DISPLACE")


def _ensure_noise_texture(tex_name: str = "PROC_HEIGHT_BASIC_NOISE"):
    tex = bpy.data.textures.get(tex_name)
    if tex is None:
        tex = bpy.data.textures.new(name=tex_name, type="CLOUDS")
    return tex


def apply_proc_height_displacement(obj, params: ProceduralNoiseParams) -> dict:
    """Attach/update a deterministic procedural displace workflow on *obj*.

    Blender's Displace modifier is texture-driven; we keep a texture-backed
    deterministic setup for geometry displacement while also guaranteeing the
    canonical node group exists for material-side reuse and future expansion.
    """
    ensure_proc_height_basic_group()

    mod = _ensure_displace_modifier(obj)
    tex = _ensure_noise_texture()

    tex.noise_scale = float(params.scale)
    tex.noise_depth = int(max(0, min(15, round(params.detail))))
    tex.nabla = float(max(0.001, 0.001 + params.roughness * 0.02))
    tex.noise_basis = "BLENDER_ORIGINAL"
    try:
        tex.intensity = float(max(0.0, min(1.0, params.roughness)))
    except Exception:
        pass

    mod.texture = tex
    mod.texture_coords = "LOCAL"
    mod.strength = float(params.strength)
    mod.mid_level = float(params.midlevel)

    obj["proc_height_seed"] = int(params.seed)
    obj["proc_height_scale"] = float(params.scale)
    obj["proc_height_detail"] = float(params.detail)
    obj["proc_height_roughness"] = float(params.roughness)
    obj["proc_height_distortion"] = float(params.distortion)
    obj["proc_height_strength"] = float(params.strength)
    obj["proc_height_midlevel"] = float(params.midlevel)

    return {
        "modifier": str(mod.name),
        "texture": str(tex.name),
        "group": "PROC_HEIGHT_BASIC",
        "scale": float(params.scale),
        "detail": float(params.detail),
        "roughness": float(params.roughness),
        "distortion": float(params.distortion),
        "strength": float(params.strength),
        "midlevel": float(params.midlevel),
        "seed": int(params.seed),
    }
