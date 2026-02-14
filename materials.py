"""
Materials Module — procedural PBR shader-node material builders.

General-purpose material creation functions used by blender_tools and AI code.

Available material types:
  * Stucco/Plaster — noise-based bump
  * Brick — Brick Texture node with mortar
  * Wood — wave + noise grain pattern
  * Concrete — voronoi + noise aggregate
  * Stone — voronoi cells with bump
  * Glass — transmission + thin-film tint
  * Metal — brushed metallic
  * Grass — green noise ground
"""

import bpy  # type: ignore
import math


# ── Colour helpers ───────────────────────────────────────────────────────

def _srgb_to_linear(c):
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _color4(rgb):
    """Convert 0-1 sRGB list to linear RGBA tuple."""
    r, g, b = [max(0.0, min(1.0, v)) for v in rgb]
    return (_srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b), 1.0)


def _vary(color4, amount=0.08):
    """Return a slightly darker version of color4."""
    return tuple(max(0, c - amount) for c in color4[:3]) + (1.0,)


# ── Node-graph helpers ───────────────────────────────────────────────────

def _new_mat(name):
    """Create a fresh material with nodes, return (mat, nodes, links, bsdf, output)."""
    if name in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials[name], do_unlink=True)
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    return mat, nodes, links, bsdf, output


def _add_tex_coord(nodes, links):
    tc = nodes.new('ShaderNodeTexCoord')
    tc.location = (-900, 0)
    return tc


def _add_mapping(nodes, links, tc, scale=(1, 1, 1)):
    mp = nodes.new('ShaderNodeMapping')
    mp.location = (-700, 0)
    mp.inputs['Scale'].default_value = scale
    links.new(tc.outputs['Object'], mp.inputs['Vector'])
    return mp


def _connect_bump(nodes, links, bsdf, height_socket, strength=0.3):
    bump = nodes.new('ShaderNodeBump')
    bump.location = (-200, -300)
    bump.inputs['Strength'].default_value = strength
    links.new(height_socket, bump.inputs['Height'])
    links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    return bump


# ── Material builders (public API) ───────────────────────────────────────

def make_stucco(name, color=(0.93, 0.91, 0.89), roughness=0.85):
    """Create a stucco/plaster material with noise bump."""
    color4 = _color4(color) if len(color) == 3 else color
    mat, nodes, links, bsdf, out = _new_mat(name)
    bsdf.inputs['Base Color'].default_value = color4
    bsdf.inputs['Roughness'].default_value = roughness
    tc = _add_tex_coord(nodes, links)
    mp = _add_mapping(nodes, links, tc, scale=(8, 8, 8))
    noise = nodes.new('ShaderNodeTexNoise')
    noise.location = (-500, -200)
    noise.inputs['Scale'].default_value = 120.0
    noise.inputs['Detail'].default_value = 10.0
    noise.inputs['Roughness'].default_value = 0.7
    links.new(mp.outputs['Vector'], noise.inputs['Vector'])
    mix = nodes.new('ShaderNodeMix')
    mix.data_type = 'RGBA'
    mix.location = (-200, 100)
    mix.inputs['Factor'].default_value = 0.08
    mix.inputs[6].default_value = color4
    mix.inputs[7].default_value = _vary(color4, 0.06)
    links.new(noise.outputs['Fac'], mix.inputs['Factor'])
    links.new(mix.outputs[2], bsdf.inputs['Base Color'])
    _connect_bump(nodes, links, bsdf, noise.outputs['Fac'], 0.15)
    return mat


def make_brick(name, color=(0.72, 0.38, 0.25), roughness=0.80):
    """Create a procedural brick material."""
    color4 = _color4(color) if len(color) == 3 else color
    mat, nodes, links, bsdf, out = _new_mat(name)
    bsdf.inputs['Roughness'].default_value = roughness
    tc = _add_tex_coord(nodes, links)
    mp = _add_mapping(nodes, links, tc, scale=(1, 1, 1))
    brick = nodes.new('ShaderNodeTexBrick')
    brick.location = (-500, 0)
    brick.inputs['Color1'].default_value = color4
    brick.inputs['Color2'].default_value = _vary(color4, 0.12)
    brick.inputs['Mortar'].default_value = (0.75, 0.73, 0.70, 1.0)
    brick.inputs['Scale'].default_value = 6.0
    brick.inputs['Mortar Size'].default_value = 0.015
    links.new(mp.outputs['Vector'], brick.inputs['Vector'])
    links.new(brick.outputs['Color'], bsdf.inputs['Base Color'])
    _connect_bump(nodes, links, bsdf, brick.outputs['Fac'], 0.25)
    return mat


def make_concrete(name, color=(0.6, 0.58, 0.55), roughness=0.92):
    """Create a procedural concrete material."""
    color4 = _color4(color) if len(color) == 3 else color
    mat, nodes, links, bsdf, out = _new_mat(name)
    bsdf.inputs['Base Color'].default_value = color4
    bsdf.inputs['Roughness'].default_value = roughness
    tc = _add_tex_coord(nodes, links)
    mp = _add_mapping(nodes, links, tc, scale=(4, 4, 4))
    voronoi = nodes.new('ShaderNodeTexVoronoi')
    voronoi.location = (-500, -200)
    voronoi.inputs['Scale'].default_value = 30.0
    links.new(mp.outputs['Vector'], voronoi.inputs['Vector'])
    noise = nodes.new('ShaderNodeTexNoise')
    noise.location = (-500, -400)
    noise.inputs['Scale'].default_value = 60.0
    noise.inputs['Detail'].default_value = 6.0
    links.new(mp.outputs['Vector'], noise.inputs['Vector'])
    mix = nodes.new('ShaderNodeMix')
    mix.data_type = 'FLOAT'
    mix.location = (-300, -300)
    mix.inputs['Factor'].default_value = 0.5
    links.new(voronoi.outputs['Distance'], mix.inputs[2])
    links.new(noise.outputs['Fac'], mix.inputs[3])
    _connect_bump(nodes, links, bsdf, mix.outputs[0], 0.12)
    return mat


def make_wood(name, color=(0.35, 0.22, 0.12), roughness=0.65):
    """Create a procedural wood grain material."""
    color4 = _color4(color) if len(color) == 3 else color
    mat, nodes, links, bsdf, out = _new_mat(name)
    bsdf.inputs['Roughness'].default_value = roughness
    tc = _add_tex_coord(nodes, links)
    mp = _add_mapping(nodes, links, tc, scale=(1, 8, 1))
    wave = nodes.new('ShaderNodeTexWave')
    wave.location = (-500, 0)
    wave.wave_type = 'BANDS'
    wave.inputs['Scale'].default_value = 3.0
    wave.inputs['Distortion'].default_value = 5.0
    wave.inputs['Detail'].default_value = 4.0
    links.new(mp.outputs['Vector'], wave.inputs['Vector'])
    ramp = nodes.new('ShaderNodeValToRGB')
    ramp.location = (-300, 0)
    ramp.color_ramp.elements[0].color = color4
    ramp.color_ramp.elements[1].color = _vary(color4, 0.15)
    links.new(wave.outputs['Fac'], ramp.inputs['Fac'])
    links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    _connect_bump(nodes, links, bsdf, wave.outputs['Fac'], 0.08)
    return mat


def make_stone(name, color=(0.5, 0.45, 0.4), roughness=0.78):
    """Create a procedural stone material."""
    color4 = _color4(color) if len(color) == 3 else color
    mat, nodes, links, bsdf, out = _new_mat(name)
    bsdf.inputs['Roughness'].default_value = roughness
    tc = _add_tex_coord(nodes, links)
    mp = _add_mapping(nodes, links, tc, scale=(2, 2, 2))
    voronoi = nodes.new('ShaderNodeTexVoronoi')
    voronoi.location = (-500, 0)
    voronoi.inputs['Scale'].default_value = 5.0
    links.new(mp.outputs['Vector'], voronoi.inputs['Vector'])
    ramp = nodes.new('ShaderNodeValToRGB')
    ramp.location = (-300, 0)
    ramp.color_ramp.elements[0].color = _vary(color4, 0.1)
    ramp.color_ramp.elements[1].color = color4
    links.new(voronoi.outputs['Distance'], ramp.inputs['Fac'])
    links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    _connect_bump(nodes, links, bsdf, voronoi.outputs['Distance'], 0.30)
    return mat


def make_glass(name, color=(0.9, 0.95, 1.0)):
    """Create a glass material with transmission."""
    mat, nodes, links, bsdf, out = _new_mat(name)
    bsdf.inputs['Base Color'].default_value = (color[0], color[1], color[2], 1.0)
    bsdf.inputs['Roughness'].default_value = 0.0
    if 'Transmission Weight' in bsdf.inputs:
        bsdf.inputs['Transmission Weight'].default_value = 0.95
    elif 'Transmission' in bsdf.inputs:
        bsdf.inputs['Transmission'].default_value = 0.95
    if 'IOR' in bsdf.inputs:
        bsdf.inputs['IOR'].default_value = 1.52
    return mat


def make_metal(name, color=(0.8, 0.8, 0.82), roughness=0.35):
    """Create a metallic material."""
    color4 = _color4(color) if len(color) == 3 else color
    mat, nodes, links, bsdf, out = _new_mat(name)
    bsdf.inputs['Base Color'].default_value = color4
    bsdf.inputs['Roughness'].default_value = roughness
    bsdf.inputs['Metallic'].default_value = 0.95
    tc = _add_tex_coord(nodes, links)
    mp = _add_mapping(nodes, links, tc, scale=(1, 20, 1))
    noise = nodes.new('ShaderNodeTexNoise')
    noise.location = (-500, -200)
    noise.inputs['Scale'].default_value = 200.0
    noise.inputs['Detail'].default_value = 2.0
    links.new(mp.outputs['Vector'], noise.inputs['Vector'])
    _connect_bump(nodes, links, bsdf, noise.outputs['Fac'], 0.03)
    return mat


def make_grass(name="Grass"):
    """Create a procedural grass material for ground planes."""
    mat, nodes, links, bsdf, out = _new_mat(name)
    bsdf.inputs['Roughness'].default_value = 0.95
    tc = _add_tex_coord(nodes, links)
    mp = _add_mapping(nodes, links, tc, scale=(6, 6, 6))
    noise = nodes.new('ShaderNodeTexNoise')
    noise.location = (-500, 0)
    noise.inputs['Scale'].default_value = 40.0
    noise.inputs['Detail'].default_value = 8.0
    links.new(mp.outputs['Vector'], noise.inputs['Vector'])
    ramp = nodes.new('ShaderNodeValToRGB')
    ramp.location = (-300, 0)
    ramp.color_ramp.elements[0].color = (0.02, 0.12, 0.01, 1.0)
    ramp.color_ramp.elements[1].color = (0.04, 0.22, 0.03, 1.0)
    links.new(noise.outputs['Fac'], ramp.inputs['Fac'])
    links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    _connect_bump(nodes, links, bsdf, noise.outputs['Fac'], 0.4)
    return mat
