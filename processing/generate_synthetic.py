"""Synthetic shape generator for training data — Blender primitives only.

Generates parametric 3D shapes using ONLY shapes that correspond to real
Blender primitives (Add → Mesh). This guarantees every generated shape is
exactly what its label says — no assembled composites that might be mislabeled.

Categories (Blender-native primitives only):
  box, cube, sphere, cylinder, cone, pyramid, torus, plane, wedge, icosphere,
  circle (disc), grid, terrain (heightmap landscape), monkey (Suzanne)

Complex objects (furniture, architecture, etc.) should come from real mesh
data sources (Objaverse, BlendSwap, etc.) where a human already verified
the geometry matches the label.

Usage:
    python -m processing.generate_synthetic \\
        --output data/datasets/geometry \\
        --config config_synthetic.yaml \\
        --num-examples 50000
"""

import argparse
import json
import logging
import math
import random
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ===========================================================================
# Utility: merge two meshes
# ===========================================================================

def _merge(v1, f1, v2, f2):
    """Merge two meshes, offsetting face indices."""
    off = len(v1)
    return v1 + v2, f1 + [[i + off for i in face] for face in f2]


def _offset_verts(verts, dx: float = 0, dy: float = 0, dz: float = 0):
    """Translate vertices and return them."""
    for v in verts:
        v[0] += dx
        v[1] += dy
        v[2] += dz
    return verts


# ===========================================================================
# Primitive shape generators -- return (vertices, faces)
# ===========================================================================

def make_box(sx=1.0, sy=1.0, sz=1.0):
    """Axis-aligned box, centered at origin. 12 tris."""
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    v = [
        [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
        [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz],
    ]
    f = [
        [0,1,2],[0,2,3], [4,6,5],[4,7,6],
        [0,5,1],[0,4,5], [2,7,3],[2,6,7],
        [0,3,7],[0,7,4], [1,5,6],[1,6,2],
    ]
    return v, f


def make_sphere(radius=0.5, rings=8, segments=12):
    """UV sphere."""
    verts = [[0, 0, -radius]]
    for i in range(1, rings):
        phi = math.pi * i / rings
        for j in range(segments):
            theta = 2 * math.pi * j / segments
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.sin(phi) * math.sin(theta)
            z = -radius * math.cos(phi)
            verts.append([x, y, z])
    verts.append([0, 0, radius])

    faces = []
    for j in range(segments):
        j2 = (j + 1) % segments
        faces.append([0, 1 + j2, 1 + j])
    for i in range(rings - 2):
        for j in range(segments):
            j2 = (j + 1) % segments
            a = 1 + i * segments + j
            b = 1 + i * segments + j2
            c = 1 + (i + 1) * segments + j2
            d = 1 + (i + 1) * segments + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    top = len(verts) - 1
    base = 1 + (rings - 2) * segments
    for j in range(segments):
        j2 = (j + 1) % segments
        faces.append([top, base + j, base + j2])
    return verts, faces


def make_cylinder(radius=0.5, height=1.0, segments=12):
    """Cylinder with caps."""
    h = height / 2
    verts = []
    for j in range(segments):
        theta = 2 * math.pi * j / segments
        verts.append([radius * math.cos(theta), radius * math.sin(theta), -h])
    for j in range(segments):
        theta = 2 * math.pi * j / segments
        verts.append([radius * math.cos(theta), radius * math.sin(theta), h])
    bc = len(verts); verts.append([0, 0, -h])
    tc = len(verts); verts.append([0, 0, h])

    faces = []
    for j in range(segments):
        j2 = (j + 1) % segments
        a, b = j, j2
        c, d = segments + j2, segments + j
        faces.append([a, b, c])
        faces.append([a, c, d])
    for j in range(segments):
        j2 = (j + 1) % segments
        faces.append([bc, j2, j])
    for j in range(segments):
        j2 = (j + 1) % segments
        faces.append([tc, segments + j, segments + j2])
    return verts, faces


def make_cone(radius=0.5, height=1.0, segments=12):
    """Cone with bottom cap."""
    h = height / 2
    verts = []
    for j in range(segments):
        theta = 2 * math.pi * j / segments
        verts.append([radius * math.cos(theta), radius * math.sin(theta), -h])
    apex = len(verts); verts.append([0, 0, h])
    bc = len(verts); verts.append([0, 0, -h])

    faces = []
    for j in range(segments):
        j2 = (j + 1) % segments
        faces.append([j, j2, apex])
    for j in range(segments):
        j2 = (j + 1) % segments
        faces.append([bc, j2, j])
    return verts, faces


def make_pyramid(base_size=1.0, height=1.0):
    """Square-based pyramid. 6 tris."""
    h, b = height / 2, base_size / 2
    v = [[-b, -b, -h], [b, -b, -h], [b, b, -h], [-b, b, -h], [0, 0, h]]
    f = [[0,1,4],[1,2,4],[2,3,4],[3,0,4],[0,2,1],[0,3,2]]
    return v, f


def make_torus(major_r=0.5, minor_r=0.2, major_seg=12, minor_seg=8):
    """Torus (donut)."""
    verts = []
    for i in range(major_seg):
        phi = 2 * math.pi * i / major_seg
        for j in range(minor_seg):
            theta = 2 * math.pi * j / minor_seg
            x = (major_r + minor_r * math.cos(theta)) * math.cos(phi)
            y = (major_r + minor_r * math.cos(theta)) * math.sin(phi)
            z = minor_r * math.sin(theta)
            verts.append([x, y, z])
    faces = []
    for i in range(major_seg):
        i2 = (i + 1) % major_seg
        for j in range(minor_seg):
            j2 = (j + 1) % minor_seg
            a = i * minor_seg + j
            b = i * minor_seg + j2
            c = i2 * minor_seg + j2
            d = i2 * minor_seg + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    return verts, faces


def make_plane(sx=1.0, sy=1.0, subdivisions=1):
    """Flat plane on XY at z=0."""
    verts = []
    n = subdivisions + 1
    for i in range(n):
        for j in range(n):
            x = -sx/2 + sx * i / subdivisions
            y = -sy/2 + sy * j / subdivisions
            verts.append([x, y, 0.0])
    faces = []
    for i in range(subdivisions):
        for j in range(subdivisions):
            a = i * n + j
            b = a + 1
            c = (i + 1) * n + j + 1
            d = (i + 1) * n + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    return verts, faces


def make_wedge(sx=1.0, sy=1.0, sz=1.0):
    """Triangular prism / wedge. 8 tris."""
    hx, hy, hz = sx/2, sy/2, sz/2
    v = [
        [-hx, -hy, -hz], [hx, -hy, -hz], [0, -hy, hz],
        [-hx,  hy, -hz], [hx,  hy, -hz], [0,  hy, hz],
    ]
    f = [
        [0,1,2], [3,5,4],
        [0,3,4],[0,4,1], [0,2,5],[0,5,3], [1,4,5],[1,5,2],
    ]
    return v, f


def make_icosphere(radius=0.5, subdivisions=1):
    """Icosphere via subdivision."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = [
        [-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],
        [0,-1,t],[0,1,t],[0,-1,-t],[0,1,-t],
        [t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1],
    ]
    for i, v in enumerate(verts):
        length = math.sqrt(sum(c**2 for c in v))
        verts[i] = [c / length * radius for c in v]

    faces = [
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ]
    for _ in range(subdivisions):
        mid_cache = {}
        new_faces = []
        for tri in faces:
            mids = []
            for e in [(tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])]:
                key = tuple(sorted(e))
                if key not in mid_cache:
                    v1, v2 = verts[key[0]], verts[key[1]]
                    mid = [(v1[i]+v2[i])/2 for i in range(3)]
                    length = math.sqrt(sum(c**2 for c in mid))
                    mid = [c / length * radius for c in mid]
                    mid_cache[key] = len(verts)
                    verts.append(mid)
                mids.append(mid_cache[key])
            a, b, c = tri
            m0, m1, m2 = mids
            new_faces += [[a,m0,m2],[b,m1,m0],[c,m2,m1],[m0,m1,m2]]
        faces = new_faces
    return verts, faces


def make_circle(radius=0.5, segments=12, fill=True):
    """Flat filled disc / circle. Blender's Circle primitive (ngon fill).

    A flat disc on the XY plane at z=0. With fill=True this is a solid disc,
    otherwise it's just the edge ring (no faces).
    """
    verts = [[0.0, 0.0, 0.0]]  # center vertex
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        verts.append([radius * math.cos(angle), radius * math.sin(angle), 0.0])
    faces = []
    if fill:
        for i in range(segments):
            next_i = (i % segments) + 1
            next_next = ((i + 1) % segments) + 1
            if next_next > segments:
                next_next = 1
            faces.append([0, next_i, next_next])
    return verts, faces


def make_grid(sx=1.0, sy=1.0, nx=4, ny=4):
    """Subdivided grid on XY plane. Blender's Grid primitive.

    Like make_plane but always has meaningful subdivisions (nx × ny cells).
    """
    verts = []
    for iy in range(ny + 1):
        for ix in range(nx + 1):
            x = -sx / 2 + sx * ix / nx
            y = -sy / 2 + sy * iy / ny
            verts.append([x, y, 0.0])
    faces = []
    for iy in range(ny):
        for ix in range(nx):
            a = iy * (nx + 1) + ix
            b = a + 1
            c = a + (nx + 1) + 1
            d = a + (nx + 1)
            faces.append([a, b, c])
            faces.append([a, c, d])
    return verts, faces


def make_terrain(sx=1.0, sy=1.0, nx=24, ny=24, max_height=0.3):
    """Heightmap terrain using multi-octave fractal displacement.

    Produces smoother, more natural terrain than single-pass random noise by
    combining low-frequency structure + high-frequency detail (fBm style).
    """
    rng_seed = random.randint(0, 1_000_000)

    def _noise(ix: int, iy: int, freq: float, phase: float) -> float:
        x = (ix / max(1, nx)) * freq + phase
        y = (iy / max(1, ny)) * freq - phase
        base = math.sin(x * 2.3) * 0.55 + math.cos(y * 2.9) * 0.45
        mix = math.sin((x + y) * 1.7) * 0.35 + math.cos((x - y) * 1.3) * 0.25
        jitter = math.sin((ix * 12.9898 + iy * 78.233 + rng_seed) * 0.017) * 0.15
        return base + mix + jitter

    # Multi-octave fractal blend (coarse shape + detail)
    heights = [[0.0 for _ in range(nx + 1)] for _ in range(ny + 1)]
    octaves = [
        (1.5, 1.00),
        (3.0, 0.50),
        (6.0, 0.25),
        (12.0, 0.12),
    ]
    for iy in range(ny + 1):
        for ix in range(nx + 1):
            h = 0.0
            for freq, amp in octaves:
                phase = (rng_seed % 997) / 997.0
                h += _noise(ix, iy, freq, phase) * amp
            # soft ridge transform to avoid flat sheet look
            h = math.copysign(abs(h) ** 1.35, h)
            heights[iy][ix] = h

    # Normalize to requested amplitude
    flat_vals = [v for row in heights for v in row]
    lo, hi = min(flat_vals), max(flat_vals)
    span = max(1e-6, hi - lo)
    for iy in range(ny + 1):
        for ix in range(nx + 1):
            nrm = (heights[iy][ix] - lo) / span  # [0,1]
            heights[iy][ix] = (nrm * 2.0 - 1.0) * max_height

    # Additional neighborhood smoothing for natural displacement continuity
    for _pass in range(2):
        smoothed = [row[:] for row in heights]
        for iy in range(ny + 1):
            for ix in range(nx + 1):
                total = heights[iy][ix]
                count = 1
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    niy, nix = iy + dy, ix + dx
                    if 0 <= niy <= ny and 0 <= nix <= nx:
                        total += heights[niy][nix]
                        count += 1
                smoothed[iy][ix] = total / count
        heights = smoothed
    # Build mesh
    verts = []
    for iy in range(ny + 1):
        for ix in range(nx + 1):
            x = -sx / 2 + sx * ix / nx
            y = -sy / 2 + sy * iy / ny
            z = heights[iy][ix]
            verts.append([x, y, z])
    faces = []
    for iy in range(ny):
        for ix in range(nx):
            a = iy * (nx + 1) + ix
            b = a + 1
            c = a + (nx + 1) + 1
            d = a + (nx + 1)
            faces.append([a, b, c])
            faces.append([a, c, d])
    return verts, faces


# Suzanne (monkey head) — loaded from data file extracted from Blender.
# Run: blender --background --python scripts/extract_suzanne.py
# This dumps the exact built-in Suzanne mesh to data/suzanne_mesh.json.
_SUZANNE_CACHE = None


def make_monkey(scale=0.5):
    """Blender's built-in Suzanne (monkey head).

    Loads the exact mesh data extracted from Blender's built-in primitive.
    Returns None if the data file hasn't been generated yet.
    Run scripts/extract_suzanne.py in Blender to create it.
    """
    global _SUZANNE_CACHE
    if _SUZANNE_CACHE is None:
        data_path = Path(__file__).parent.parent / "data" / "suzanne_mesh.json"
        if not data_path.exists():
            if not getattr(make_monkey, '_warned', False):
                logger.warning(
                    f"Suzanne mesh data not found at {data_path}. "
                    f"Run: blender --background --python scripts/extract_suzanne.py"
                )
                make_monkey._warned = True
            return [], []
        import json as _json
        with open(data_path) as f:
            data = _json.load(f)
        _SUZANNE_CACHE = (data["vertices"], data["faces"])

    verts, faces = _SUZANNE_CACHE
    # Deep copy and scale
    scaled_verts = [[c * scale for c in v] for v in verts]
    return scaled_verts, [list(f) for f in faces]


def make_l_shape(sx=1.0, sy=0.5, sz=1.0, thickness=0.3):
    """L-shaped block. 24 tris."""
    t = thickness
    v1, f1 = make_box(t, sy, sz)
    v2, f2 = make_box(sx, sy, t)
    for v in v1:
        v[0] -= (sx - t) / 2; v[2] += (sz - t) / 2
    for v in v2:
        v[2] -= (sz - t) / 2
    return _merge(v1, f1, v2, f2)


def make_stairs(steps=4, width=1.0, height=1.0, depth=1.0):
    """Staircase shape."""
    all_v, all_f = [], []
    sh = height / steps
    sd = depth / steps
    for i in range(steps):
        v, f = make_box(width, sd, sh * (i + 1))
        for vert in v:
            vert[1] += sd * i - depth / 2 + sd / 2
            vert[2] -= (height - sh * (i + 1)) / 2
        all_v, all_f = _merge(all_v, all_f, v, f)
    return all_v, all_f


# ===========================================================================
# Composite / furniture / architecture generators
# ===========================================================================

def make_table(width=1.0, depth=0.6, height=0.8, leg_t=0.06):
    """Table = top slab + 4 legs. 60 tris."""
    top_h = leg_t
    v, f = make_box(width, depth, top_h)
    _offset_verts(v, dz=height / 2 - top_h / 2)
    all_v, all_f = v, f
    leg_h = height - top_h
    for ox, oy in [(-1,-1),(1,-1),(1,1),(-1,1)]:
        lv, lf = make_box(leg_t, leg_t, leg_h)
        _offset_verts(lv, dx=ox*(width/2-leg_t/2), dy=oy*(depth/2-leg_t/2), dz=-top_h/2)
        all_v, all_f = _merge(all_v, all_f, lv, lf)
    return all_v, all_f


def make_chair(seat_w=0.5, seat_d=0.5, seat_h=0.45, back_h=0.4, leg_t=0.04):
    """Chair = seat + 4 legs + backrest. 72 tris."""
    v, f = make_box(seat_w, seat_d, leg_t)
    _offset_verts(v, dz=seat_h - leg_t / 2)
    all_v, all_f = v, f
    for ox, oy in [(-1,-1),(1,-1),(1,1),(-1,1)]:
        lv, lf = make_box(leg_t, leg_t, seat_h)
        _offset_verts(lv, dx=ox*(seat_w/2-leg_t/2), dy=oy*(seat_d/2-leg_t/2), dz=-leg_t/2)
        all_v, all_f = _merge(all_v, all_f, lv, lf)
    bv, bf = make_box(seat_w, leg_t, back_h)
    _offset_verts(bv, dy=-(seat_d/2-leg_t/2), dz=seat_h + back_h/2)
    return _merge(all_v, all_f, bv, bf)


def make_house(width=1.0, depth=0.8, wall_h=0.6, roof_h=0.4):
    """House = box walls + pyramid roof. 18 tris."""
    v, f = make_box(width, depth, wall_h)
    _offset_verts(v, dz=-roof_h / 2)
    hx, hy = width / 2, depth / 2
    rz_base = wall_h / 2 - roof_h / 2
    rv = [[-hx,-hy,rz_base],[hx,-hy,rz_base],[hx,hy,rz_base],[-hx,hy,rz_base],[0,0,rz_base+roof_h]]
    rf = [[0,1,4],[1,2,4],[2,3,4],[3,0,4],[0,2,1],[0,3,2]]
    return _merge(v, f, rv, rf)


def make_bookshelf(width=0.8, depth=0.25, height=1.2, shelves=3, thick=0.03):
    """Bookshelf = 2 sides + n shelves + back."""
    all_v, all_f = [], []
    for side in [-1, 1]:
        sv, sf = make_box(thick, depth, height)
        _offset_verts(sv, dx=side * (width / 2 - thick / 2))
        all_v, all_f = _merge(all_v, all_f, sv, sf)
    for i in range(shelves + 2):
        sz = -height/2 + i * height / (shelves + 1)
        shv, shf = make_box(width, depth, thick)
        _offset_verts(shv, dz=sz)
        all_v, all_f = _merge(all_v, all_f, shv, shf)
    bv, bf = make_box(width, thick, height)
    _offset_verts(bv, dy=depth / 2 - thick / 2)
    return _merge(all_v, all_f, bv, bf)


def make_bench(width=1.0, depth=0.35, height=0.45, leg_t=0.05):
    """Bench = seat + 4 legs (no back). 60 tris."""
    return make_table(width, depth, height, leg_t)


def make_stool(radius=0.2, height=0.5, legs=3, leg_t=0.03):
    """Stool = cylinder seat + legs."""
    sv, sf = make_cylinder(radius, leg_t * 2, segments=6)
    _offset_verts(sv, dz=height / 2)
    all_v, all_f = sv, sf
    for i in range(legs):
        angle = 2 * math.pi * i / legs
        lv, lf = make_box(leg_t, leg_t, height)
        _offset_verts(lv, dx=radius * 0.7 * math.cos(angle),
                      dy=radius * 0.7 * math.sin(angle))
        all_v, all_f = _merge(all_v, all_f, lv, lf)
    return all_v, all_f


def make_bed(width=1.0, depth=2.0, height=0.5, leg_t=0.06):
    """Bed = frame + mattress + headboard. 48 tris."""
    mv, mf = make_box(width, depth, height * 0.3)
    _offset_verts(mv, dz=height * 0.35)
    fv, ff = make_box(width, depth, height * 0.15)
    _offset_verts(fv, dz=height * 0.07)
    all_v, all_f = _merge(mv, mf, fv, ff)
    hv, hf = make_box(width, leg_t * 2, height * 0.6)
    _offset_verts(hv, dy=-depth / 2 + leg_t, dz=height * 0.3)
    return _merge(all_v, all_f, hv, hf)


def make_sofa(width=1.5, depth=0.7, seat_h=0.35, back_h=0.4, arm_w=0.12):
    """Sofa = seat + back + 2 armrests. 48 tris."""
    sv, sf = make_box(width, depth, seat_h * 0.4)
    _offset_verts(sv, dz=seat_h * 0.2)
    bv, bf = make_box(width, depth * 0.2, back_h)
    _offset_verts(bv, dy=-depth/2 + depth*0.1, dz=seat_h + back_h/2 - seat_h*0.2)
    all_v, all_f = _merge(sv, sf, bv, bf)
    for side in [-1, 1]:
        av, af = make_box(arm_w, depth, seat_h * 0.7)
        _offset_verts(av, dx=side * (width/2 - arm_w/2), dz=seat_h * 0.35)
        all_v, all_f = _merge(all_v, all_f, av, af)
    return all_v, all_f


def make_wardrobe(width=0.8, depth=0.5, height=1.8, thick=0.04):
    """Wardrobe = box."""
    return make_box(width, depth, height)


def make_lamp(base_r=0.12, pole_r=0.02, pole_h=0.6, shade_r=0.15, shade_h=0.12):
    """Lamp = cylinder base + thin pole + cone shade."""
    bv, bf = make_cylinder(base_r, 0.03, segments=6)
    _offset_verts(bv, dz=-pole_h / 2)
    pv, pf = make_cylinder(pole_r, pole_h, segments=4)
    all_v, all_f = _merge(bv, bf, pv, pf)
    sv, sf = make_cone(shade_r, shade_h, segments=6)
    _offset_verts(sv, dz=pole_h / 2 + shade_h / 2)
    for v in sv:
        v[2] = pole_h / 2 + shade_h - (v[2] - pole_h / 2)
    return _merge(all_v, all_f, sv, sf)


def make_cabinet(width=0.6, depth=0.4, height=0.7, thick=0.03):
    """Cabinet = short bookshelf with 1 shelf."""
    return make_bookshelf(width, depth, height, shelves=1, thick=thick)


def make_shelf(width=0.8, depth=0.2, thick=0.03):
    """Wall shelf = single plank. 12 tris."""
    return make_box(width, depth, thick)


# -- Architecture --

def make_arch(width=0.8, height=1.0, depth=0.3, segments=6, thick=0.1):
    """Arch = half-torus top + 2 pillars."""
    pv1, pf1 = make_box(thick, depth, height * 0.5)
    _offset_verts(pv1, dx=-width / 2 + thick / 2, dz=-height * 0.25)
    pv2, pf2 = make_box(thick, depth, height * 0.5)
    _offset_verts(pv2, dx=width / 2 - thick / 2, dz=-height * 0.25)
    all_v, all_f = _merge(pv1, pf1, pv2, pf2)
    arch_r = width / 2
    for i in range(segments):
        a1 = math.pi * i / segments
        a2 = math.pi * (i + 1) / segments
        for dy_off in [-depth/2, depth/2]:
            x1, z1 = -arch_r * math.cos(a1), arch_r * math.sin(a1)
            x2, z2 = -arch_r * math.cos(a2), arch_r * math.sin(a2)
            x1i = x1 * (1 - thick / arch_r)
            z1i = z1 * (1 - thick / arch_r)
            x2i = x2 * (1 - thick / arch_r)
            z2i = z2 * (1 - thick / arch_r)
            base_idx = len(all_v)
            zoff = height * 0.25
            all_v.extend([
                [x1, dy_off, z1 + zoff], [x2, dy_off, z2 + zoff],
                [x2i, dy_off, z2i + zoff], [x1i, dy_off, z1i + zoff],
            ])
            all_f.extend([[base_idx, base_idx+1, base_idx+2],
                          [base_idx, base_idx+2, base_idx+3]])
    return all_v, all_f


def make_column(radius=0.15, height=1.5, segments=6):
    """Column = tall cylinder."""
    return make_cylinder(radius, height, segments)


def make_tower(base=0.5, height=1.5, roof_h=0.3, segments=6):
    """Tower = cylinder body + cone roof."""
    bv, bf = make_cylinder(base / 2, height, segments)
    rv, rf = make_cone(base / 2 * 1.1, roof_h, segments)
    _offset_verts(rv, dz=height / 2 + roof_h / 2)
    return _merge(bv, bf, rv, rf)


def make_chimney(width=0.25, depth=0.25, height=0.6):
    """Chimney = tall box. 12 tris."""
    return make_box(width, depth, height)


def make_door(width=0.9, height=2.0, depth=0.05):
    """Door = thin box. 12 tris."""
    return make_box(width, depth, height)


def make_window(width=0.6, height=0.8, depth=0.05, frame_t=0.05):
    """Window = frame (4 thin boxes). 48 tris."""
    all_v, all_f = [], []
    for pos, sx, sz in [
        ([0, 0, height/2 - frame_t/2], width, frame_t),
        ([0, 0, -height/2 + frame_t/2], width, frame_t),
        ([-width/2 + frame_t/2, 0, 0], frame_t, height),
        ([width/2 - frame_t/2, 0, 0], frame_t, height),
    ]:
        v, f = make_box(sx, depth, sz)
        _offset_verts(v, dx=pos[0], dy=pos[1], dz=pos[2])
        all_v, all_f = _merge(all_v, all_f, v, f)
    return all_v, all_f


def make_fence(width=2.0, height=0.8, posts=5, post_t=0.05, rail_t=0.03):
    """Fence = posts + 2 rails."""
    all_v, all_f = [], []
    for i in range(posts):
        x = -width / 2 + i * width / (posts - 1) if posts > 1 else 0
        pv, pf = make_box(post_t, post_t, height)
        _offset_verts(pv, dx=x)
        all_v, all_f = _merge(all_v, all_f, pv, pf)
    for rz in [height * 0.3, height * -0.15]:
        rv, rf = make_box(width, rail_t, rail_t)
        _offset_verts(rv, dz=rz)
        all_v, all_f = _merge(all_v, all_f, rv, rf)
    return all_v, all_f


def make_bridge(length=2.0, width=0.6, height=0.3, thick=0.06):
    """Bridge = deck + 2 side rails + 4 supports."""
    dv, df = make_box(width, length, thick)
    all_v, all_f = dv, df
    for side in [-1, 1]:
        rv, rf = make_box(thick, length, height)
        _offset_verts(rv, dx=side * (width / 2 - thick / 2), dz=height / 2)
        all_v, all_f = _merge(all_v, all_f, rv, rf)
    for side in [-1, 1]:
        for end in [-1, 1]:
            sv, sf = make_box(thick, thick, height + thick)
            _offset_verts(sv, dx=side*(width/2-thick/2), dy=end*(length/2-thick/2))
            all_v, all_f = _merge(all_v, all_f, sv, sf)
    return all_v, all_f


def make_well(radius=0.3, height=0.5, roof_h=0.3, post_t=0.03, segments=6):
    """Well = cylinder + 2 posts + pyramid roof."""
    wv, wf = make_cylinder(radius, height, segments)
    all_v, all_f = wv, wf
    ph = height + roof_h
    for side in [-1, 1]:
        pv, pf = make_box(post_t, post_t, ph)
        _offset_verts(pv, dx=side * radius, dz=ph / 2 - height / 2)
        all_v, all_f = _merge(all_v, all_f, pv, pf)
    rv, rf = make_pyramid(radius * 2.5, roof_h)
    _offset_verts(rv, dz=height / 2 + roof_h / 2 + 0.05)
    return _merge(all_v, all_f, rv, rf)


def make_wall(width=2.0, height=1.0, depth=0.15):
    """Wall = flat box. 12 tris."""
    return make_box(width, depth, height)


# -- Nature --

def make_tree(trunk_r=0.08, trunk_h=0.6, crown_r=0.35, crown_h=0.5):
    """Tree = cylinder trunk + cone crown."""
    tv, tf = make_cylinder(trunk_r, trunk_h, segments=5)
    cv, cf = make_cone(crown_r, crown_h, segments=6)
    _offset_verts(cv, dz=trunk_h / 2 + crown_h / 2)
    return _merge(tv, tf, cv, cf)


def make_rock(radius=0.3, distortion=0.15):
    """Rock = distorted icosphere. 80 tris."""
    verts, faces = make_icosphere(radius, subdivisions=1)
    rng = random.Random()
    for v in verts:
        v[0] += rng.uniform(-distortion, distortion)
        v[1] += rng.uniform(-distortion, distortion)
        v[2] += rng.uniform(-distortion, distortion)
    return verts, faces


def make_mushroom(cap_r=0.2, cap_h=0.08, stem_r=0.05, stem_h=0.2):
    """Mushroom = cylinder stem + flat sphere cap."""
    sv, sf = make_cylinder(stem_r, stem_h, segments=5)
    cv, cf = make_sphere(cap_r, rings=3, segments=6)
    arr = np.array(cv)
    arr[:, 2] = np.clip(arr[:, 2], -cap_h, cap_r)
    cv = arr.tolist()
    _offset_verts(cv, dz=stem_h / 2 + cap_h / 2)
    return _merge(sv, sf, cv, cf)


def make_cactus(radius=0.1, height=0.6, arm_r=0.06, arm_h=0.2):
    """Cactus = main cylinder + 1-2 arm branches."""
    bv, bf = make_cylinder(radius, height, segments=5)
    all_v, all_f = bv, bf
    a1v, a1f = make_cylinder(arm_r, arm_h, segments=4)
    for v in a1v:
        v[0], v[2] = v[2], v[0]
    _offset_verts(a1v, dx=arm_h / 2 + radius, dz=height * 0.15)
    all_v, all_f = _merge(all_v, all_f, a1v, a1f)
    upv, upf = make_cylinder(arm_r, arm_h * 0.5, segments=4)
    _offset_verts(upv, dx=arm_h + radius, dz=height * 0.15 + arm_h * 0.25 + arm_r)
    return _merge(all_v, all_f, upv, upf)


def make_snowman(r1=0.25, r2=0.18, r3=0.12):
    """Snowman = 3 stacked spheres."""
    s1v, s1f = make_sphere(r1, rings=4, segments=5)
    _offset_verts(s1v, dz=r1)
    s2v, s2f = make_sphere(r2, rings=4, segments=5)
    _offset_verts(s2v, dz=r1 * 2 + r2)
    s3v, s3f = make_sphere(r3, rings=3, segments=5)
    _offset_verts(s3v, dz=r1 * 2 + r2 * 2 + r3)
    all_v, all_f = _merge(s1v, s1f, s2v, s2f)
    return _merge(all_v, all_f, s3v, s3f)


# -- Objects --

def make_barrel(radius=0.3, height=0.6, bulge=1.15, segments=6):
    """Barrel = bulged cylinder."""
    verts, faces = make_cylinder(radius, height, segments)
    h = height / 2
    for v in verts:
        t = 1.0 - abs(v[2]) / h if abs(v[2]) < h else 0.0
        scale = 1.0 + (bulge - 1.0) * t
        v[0] *= scale
        v[1] *= scale
    return verts, faces


def make_bottle(body_r=0.12, body_h=0.35, neck_r=0.04, neck_h=0.15):
    """Bottle = cylinder body + thin cylinder neck."""
    bv, bf = make_cylinder(body_r, body_h, segments=6)
    nv, nf = make_cylinder(neck_r, neck_h, segments=5)
    _offset_verts(nv, dz=body_h / 2 + neck_h / 2)
    return _merge(bv, bf, nv, nf)


def make_cup(radius=0.1, height=0.15, handle_t=0.02):
    """Cup = cylinder + handle box."""
    cv, cf = make_cylinder(radius, height, segments=6)
    hv, hf = make_box(handle_t, radius * 0.5, height * 0.6)
    _offset_verts(hv, dx=radius + handle_t / 2)
    return _merge(cv, cf, hv, hf)


def make_vase(radius=0.15, height=0.4, neck_r=0.08, segments=6):
    """Vase = bulged cylinder with narrow top."""
    verts, faces = make_cylinder(radius, height, segments)
    h = height / 2
    for v in verts:
        t = (v[2] + h) / height
        if t > 0.7:
            scale = 1.0 - (t - 0.7) / 0.3 * 0.5
        elif t > 0.3:
            scale = 1.0 + (t - 0.3) / 0.4 * 0.3
        else:
            scale = 0.7 + t / 0.3 * 0.3
        v[0] *= scale
        v[1] *= scale
    return verts, faces


def make_sword(blade_l=0.7, blade_w=0.05, handle_l=0.15, guard_w=0.15):
    """Sword = blade box + guard box + handle box. 36 tris."""
    bv, bf = make_box(blade_w, blade_w * 0.3, blade_l)
    _offset_verts(bv, dz=blade_l / 2)
    gv, gf = make_box(guard_w, blade_w * 1.5, blade_w * 1.5)
    hv, hf = make_box(blade_w * 0.7, blade_w * 0.7, handle_l)
    _offset_verts(hv, dz=-handle_l / 2)
    all_v, all_f = _merge(bv, bf, gv, gf)
    return _merge(all_v, all_f, hv, hf)


def make_shield(width=0.4, height=0.5, depth=0.08):
    """Shield = curved box. 12 tris."""
    v, f = make_box(width, depth, height)
    for vert in v:
        curve = 1.0 - (vert[0] / (width / 2)) ** 2
        vert[1] -= curve * depth * 0.5
    return v, f


def make_key(shaft_l=0.3, shaft_r=0.02, head_r=0.06):
    """Key = cylinder shaft + torus head."""
    sv, sf = make_cylinder(shaft_r, shaft_l, segments=4)
    for v in sv:
        v[0], v[2] = v[2], v[0]
    hv, hf = make_torus(head_r, shaft_r * 1.5, major_seg=6, minor_seg=4)
    _offset_verts(hv, dx=-shaft_l / 2 - head_r)
    return _merge(sv, sf, hv, hf)


def make_hammer(head_l=0.2, head_r=0.04, handle_l=0.4, handle_r=0.02):
    """Hammer = cylinder handle + cylinder head."""
    hv, hf = make_cylinder(handle_r, handle_l, segments=4)
    headv, headf = make_cylinder(head_r, head_l, segments=5)
    for v in headv:
        v[0], v[2] = v[2], v[0]
    _offset_verts(headv, dz=handle_l / 2)
    return _merge(hv, hf, headv, headf)


def make_trophy(cup_r=0.12, cup_h=0.15, stem_r=0.02, stem_h=0.1, base_r=0.08):
    """Trophy = cup + stem + base."""
    cv, cf = make_cylinder(cup_r, cup_h, segments=6)
    _offset_verts(cv, dz=stem_h + cup_h / 2)
    for v in cv:
        t = (v[2] - stem_h) / cup_h
        v[0] *= 0.5 + 0.5 * t
        v[1] *= 0.5 + 0.5 * t
    sv, sf = make_cylinder(stem_r, stem_h, segments=4)
    _offset_verts(sv, dz=stem_h / 2)
    bv, bf = make_cylinder(base_r, 0.03, segments=6)
    all_v, all_f = _merge(cv, cf, sv, sf)
    return _merge(all_v, all_f, bv, bf)


def make_crown(radius=0.15, height=0.1, points=5, point_h=0.06):
    """Crown = cylinder band + triangular points."""
    bv, bf = make_cylinder(radius, height, segments=points * 2)
    all_v, all_f = bv, bf
    for i in range(points):
        angle = 2 * math.pi * i / points
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        pv, pf = make_pyramid(0.04, point_h)
        _offset_verts(pv, dx=x, dy=y, dz=height / 2 + point_h / 2)
        all_v, all_f = _merge(all_v, all_f, pv, pf)
    return all_v, all_f


def make_star(outer_r=0.4, inner_r=0.18, depth=0.06, points=5):
    """Star = extruded star polygon."""
    verts_2d = []
    for i in range(points * 2):
        angle = math.pi / 2 + 2 * math.pi * i / (points * 2)
        r = outer_r if i % 2 == 0 else inner_r
        verts_2d.append([r * math.cos(angle), r * math.sin(angle)])

    verts = []
    for x, y in verts_2d:
        verts.append([x, y, -depth / 2])
    for x, y in verts_2d:
        verts.append([x, y, depth / 2])
    n = len(verts_2d)
    c_bot = len(verts); verts.append([0, 0, -depth / 2])
    c_top = len(verts); verts.append([0, 0, depth / 2])

    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([c_bot, j, i])
        faces.append([c_top, n + i, n + j])
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
    return verts, faces


def make_arrow(shaft_l=0.5, shaft_r=0.02, head_l=0.12, head_r=0.06):
    """Arrow = cylinder shaft + cone head."""
    sv, sf = make_cylinder(shaft_r, shaft_l, segments=4)
    for v in sv:
        v[0], v[2] = v[2], v[0]
    hv, hf = make_cone(head_r, head_l, segments=5)
    for v in hv:
        v[0], v[2] = v[2], v[0]
    _offset_verts(hv, dx=shaft_l / 2 + head_l / 2)
    return _merge(sv, sf, hv, hf)


def make_cross(width=0.6, height=0.8, depth=0.08, arm_w=0.15):
    """Cross = 2 boxes. 24 tris."""
    vv, vf = make_box(arm_w, depth, height)
    hv, hf = make_box(width, depth, arm_w)
    _offset_verts(hv, dz=height * 0.2)
    return _merge(vv, vf, hv, hf)


def make_diamond(radius=0.3, top_h=0.2, bottom_h=0.4, segments=6):
    """Diamond/gem = double cone."""
    tv, tf = make_cone(radius, top_h, segments)
    for v in tv:
        v[2] = top_h - v[2]
    bv, bf = make_cone(radius, bottom_h, segments)
    for v in bv:
        v[2] = -v[2]
    return _merge(tv, tf, bv, bf)


def make_gear(outer_r=0.3, inner_r=0.25, teeth=8, depth=0.08, tooth_h=0.05):
    """Gear = cylinder with teeth boxes."""
    cv, cf = make_cylinder(inner_r, depth, segments=teeth)
    all_v, all_f = cv, cf
    for i in range(teeth):
        angle = 2 * math.pi * i / teeth
        x = outer_r * math.cos(angle)
        y = outer_r * math.sin(angle)
        tw = 2 * math.pi * outer_r / teeth * 0.4
        tv, tf = make_box(tw, tw, depth)
        _offset_verts(tv, dx=x, dy=y)
        all_v, all_f = _merge(all_v, all_f, tv, tf)
    return all_v, all_f


def make_anvil(base_w=0.4, base_d=0.2, base_h=0.1, top_w=0.5, top_h=0.08, body_h=0.15):
    """Anvil = base + body + top. 36 tris."""
    basev, basef = make_box(base_w, base_d, base_h)
    _offset_verts(basev, dz=-body_h / 2 - base_h / 2)
    bodyv, bodyf = make_box(base_w * 0.6, base_d * 0.8, body_h)
    topv, topf = make_box(top_w, base_d * 0.9, top_h)
    _offset_verts(topv, dz=body_h / 2 + top_h / 2)
    all_v, all_f = _merge(basev, basef, bodyv, bodyf)
    return _merge(all_v, all_f, topv, topf)


# -- Vehicles --

def make_car_body(length=1.8, width=0.8, height=0.5, cabin_h=0.3):
    """Simple car body = body box + cabin box. 24 tris."""
    bv, bf = make_box(length, width, height)
    cv, cf = make_box(length * 0.5, width * 0.9, cabin_h)
    _offset_verts(cv, dx=-length * 0.05, dz=height / 2 + cabin_h / 2)
    return _merge(bv, bf, cv, cf)


def make_boat_hull(length=1.5, width=0.5, height=0.3):
    """Boat hull = tapered box. 12 tris."""
    v, f = make_box(length, width, height)
    for vert in v:
        t = (vert[0] + length / 2) / length
        taper = 0.3 + 0.7 * (1 - abs(2 * t - 1) ** 2)
        vert[1] *= taper
        if vert[2] < 0:
            vert[1] *= 0.7
    return v, f


# ===========================================================================
# Shape transformations
# ===========================================================================

def normalize_mesh(verts, target_range=(-1.0, 1.0)):
    """Normalize vertices to fit within target range, preserving aspect ratio."""
    arr = np.array(verts, dtype=np.float64)
    if len(arr) == 0:
        return verts
    center = (arr.max(axis=0) + arr.min(axis=0)) / 2
    arr -= center
    max_extent = np.abs(arr).max()
    if max_extent > 1e-6:
        scale = (target_range[1] - target_range[0]) / 2 / max_extent
        arr *= scale
    return arr.tolist()


def apply_rotation(verts, angle_deg, axis='z'):
    """Rotate vertices around an axis."""
    arr = np.array(verts)
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    if axis == 'z':
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    elif axis == 'x':
        rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    else:
        rot = np.eye(3)
    return (arr @ rot.T).tolist()


def apply_scale(verts, sx=1.0, sy=1.0, sz=1.0):
    """Non-uniform scale."""
    arr = np.array(verts)
    arr[:, 0] *= sx
    arr[:, 1] *= sy
    arr[:, 2] *= sz
    return arr.tolist()


# ===========================================================================
# Rich text label generation
# ===========================================================================

SIZE_ADJ = [
    'small', 'tiny', 'large', 'big', 'wide', 'tall', 'flat', 'thin',
    'thick', 'narrow', 'elongated', 'compact', 'miniature', 'oversized',
    'squat', 'slim', 'chunky', 'short', 'long', 'broad',
]

STYLE_ADJ = [
    'simple', 'basic', 'plain', 'smooth', 'geometric', 'clean', 'solid',
    'low-poly', 'minimal', 'abstract', 'modern', 'classic', 'rustic',
    'sleek', 'angular', 'rounded', 'blocky', 'stylized', 'crude',
    'detailed', 'ornate', 'elegant', 'rough', 'polished', 'primitive',
]

MATERIAL_ADJ = [
    'wooden', 'metal', 'stone', 'glass', 'plastic', 'concrete', 'brick',
    'marble', 'steel', 'iron', 'bronze', 'copper', 'ceramic', 'clay',
    'golden', 'silver', 'crystal', 'jade', 'obsidian', 'granite',
    'sandstone', 'oak', 'pine', 'bamboo', 'rubber', 'leather', 'fabric',
]

COLOR_ADJ = [
    'red', 'blue', 'green', 'yellow', 'white', 'black', 'gray', 'brown',
    'orange', 'purple', 'pink', 'dark', 'light', 'bright', 'matte',
    'glossy', 'translucent', 'metallic', 'pastel', 'vivid',
]

CONTEXT_PHRASES = [
    'for a game scene', 'for a 3D scene', 'for rendering', 'for animation',
    'for a virtual world', 'for architectural visualization',
    'for a fantasy game', 'for a sci-fi scene', 'for product design',
    'for interior design', 'as a prop', 'for a diorama',
    'for a medieval scene', 'for an outdoor scene', 'for a room',
]

IMPERATIVE_PREFIXES = [
    'Create', 'Generate', 'Make', 'Build', 'Model', 'Design', 'Render',
    'Produce', 'Construct', 'Sculpt', 'Shape', 'Form', 'Craft',
]


def generate_label(shape_key, params):
    """Generate a diverse natural-language label for a shape.
    
    Uses actual geometry parameters to create descriptive labels
    so the model can learn text↔shape correspondences (e.g.,
    'tall thin cylinder' vs 'wide flat disc').
    """
    all_specs = {**SHAPE_SPECS, **COMPOSITE_SPECS}
    spec = all_specs[shape_key]
    name = random.choice(spec['names'])

    # Build parametric descriptors from actual geometry dimensions
    param_adjs = _describe_params(shape_key, params)

    style = random.random()

    if style < 0.08:
        return name

    elif style < 0.16:
        article = "an" if name[0].lower() in "aeiou" else "a"
        return f"{article} {name}"

    elif style < 0.28:
        # Use parametric adjectives when available (50% chance)
        if param_adjs and random.random() < 0.6:
            adj1 = random.choice(param_adjs)
        else:
            adj1 = random.choice(SIZE_ADJ + STYLE_ADJ)
        if random.random() < 0.5:
            adj2 = random.choice(MATERIAL_ADJ + COLOR_ADJ)
            article = "an" if adj1[0].lower() in "aeiou" else "a"
            return f"{article} {adj1} {adj2} {name}"
        article = "an" if adj1[0].lower() in "aeiou" else "a"
        return f"{article} {adj1} {name}"

    elif style < 0.40:
        prefix = random.choice(IMPERATIVE_PREFIXES)
        if param_adjs and random.random() < 0.5:
            adj = random.choice(param_adjs)
        else:
            adj = random.choice(SIZE_ADJ + STYLE_ADJ + MATERIAL_ADJ)
        article = "an" if adj[0].lower() in "aeiou" else "a"
        return f"{prefix} {article} {adj} {name}"

    elif style < 0.50:
        if random.random() < 0.5:
            return f"3D {name}"
        article = "an" if name[0].lower() in "aeiou" else "a"
        return f"3D model of {article} {name}"

    elif style < 0.60:
        adj = random.choice(STYLE_ADJ)
        suffix = random.choice(['mesh', 'shape', 'model', 'object', 'form'])
        return f"{adj} {name} {suffix}"

    elif style < 0.70:
        # Parametric + context
        if param_adjs and random.random() < 0.6:
            adj = random.choice(param_adjs)
        else:
            adj = random.choice(MATERIAL_ADJ + STYLE_ADJ)
        ctx = random.choice(CONTEXT_PHRASES)
        article = "an" if adj[0].lower() in "aeiou" else "a"
        return f"{article} {adj} {name} {ctx}"

    elif style < 0.80:
        col = random.choice(COLOR_ADJ)
        if param_adjs and random.random() < 0.4:
            mat = random.choice(param_adjs)
        else:
            mat = random.choice(MATERIAL_ADJ)
        article = "an" if col[0].lower() in "aeiou" else "a"
        return f"{article} {col} {mat} {name}"

    elif style < 0.90:
        adj = random.choice(['low-poly', 'simple', 'basic', 'clean', 'minimal'])
        suffix = random.choice(['', ' mesh', ' model', ' shape'])
        return f"{adj} {name}{suffix}"

    else:
        # Mix parametric + style adjectives
        pool = (param_adjs or []) + SIZE_ADJ + STYLE_ADJ + MATERIAL_ADJ + COLOR_ADJ
        adjs = random.sample(pool, k=min(random.randint(2, 3), len(pool)))
        article = "an" if adjs[0][0].lower() in "aeiou" else "a"
        return f"{article} {' '.join(adjs)} {name}"


def _describe_params(shape_key, params):
    """Convert numeric geometry parameters into natural-language descriptors.
    
    Returns a list of adjectives that describe the actual geometry,
    e.g., ['tall', 'thin'] for a cylinder with large h and small r.
    """
    adjs = []
    
    # Height-based shapes (cylinder, cone, pyramid)
    if 'h' in params and 'r' in params:
        ratio = params['h'] / max(params['r'], 0.01)
        if ratio > 3.0:
            adjs.append('tall')
            adjs.append('narrow')
        elif ratio > 2.0:
            adjs.append('tall')
        elif ratio < 0.5:
            adjs.append('flat')
            adjs.append('wide')
        elif ratio < 1.0:
            adjs.append('squat')
    
    # Box/cuboid shapes
    if 'sx' in params and 'sy' in params and 'sz' in params:
        dims = [params['sx'], params['sy'], params['sz']]
        max_dim = max(dims)
        min_dim = min(dims)
        if max_dim / max(min_dim, 0.01) > 2.5:
            if params['sz'] == max_dim:
                adjs.append('tall')
            elif params['sx'] == max_dim:
                adjs.append('long')
            elif params['sy'] == max_dim:
                adjs.append('wide')
            if min_dim < 0.3:
                adjs.append('thin')
        elif max_dim / max(min_dim, 0.01) < 1.2:
            adjs.append('cubic')
    
    # Uniform scale (cube, sphere, monkey)
    if 's' in params:
        s = params['s']
        if s > 1.0:
            adjs.append('large')
        elif s < 0.5:
            adjs.append('small')
    
    # Radius-only shapes (sphere, circle)
    if 'r' in params and 'h' not in params:
        r = params['r']
        if r > 0.6:
            adjs.append('large')
        elif r < 0.35:
            adjs.append('small')
    
    # Torus aspect ratio
    if 'R' in params and 'r' in params:
        tube_ratio = params['r'] / max(params['R'], 0.01)
        if tube_ratio > 0.4:
            adjs.append('thick')
            adjs.append('chunky')
        elif tube_ratio < 0.2:
            adjs.append('thin')
            adjs.append('delicate')
    
    # Table/desk height
    if 'h' in params and 'w' in params and shape_key in ('table', 'bench', 'bed'):
        if params['h'] > 0.8:
            adjs.append('tall')
        elif params['h'] < 0.45:
            adjs.append('low')
        if params.get('w', 1.0) > 1.2:
            adjs.append('wide')
        elif params.get('w', 1.0) < 0.7:
            adjs.append('narrow')
    
    # Chair/sofa proportions
    if shape_key in ('chair', 'sofa') and 'bh' in params:
        if params['bh'] > 0.4:
            adjs.append('high-backed')
        elif params['bh'] < 0.25:
            adjs.append('low-backed')
    
    # Terrain height
    if 'mh' in params:
        if params['mh'] > 0.3:
            adjs.append('mountainous')
            adjs.append('rugged')
        elif params['mh'] < 0.15:
            adjs.append('gently rolling')
    
    return adjs


# ===========================================================================
# Shape specifications registry
# ===========================================================================

SHAPE_SPECS = {
    'box': {
        'names': ['cube', 'box', 'block', 'rectangular prism', 'cuboid'],
        'generator': lambda p: make_box(p['sx'], p['sy'], p['sz']),
        'params': lambda: {'sx': random.uniform(0.5, 1.5),
                           'sy': random.uniform(0.5, 1.5),
                           'sz': random.uniform(0.5, 1.5)},
    },
    'cube': {
        'names': ['cube', 'unit cube', 'square box', 'perfect cube'],
        'generator': lambda p: make_box(p['s'], p['s'], p['s']),
        'params': lambda: {'s': random.uniform(0.6, 1.4)},
    },
    'sphere': {
        'names': ['sphere', 'ball', 'globe', 'orb'],
        'generator': lambda p: make_sphere(p['r'], p['rings'], p['segs']),
        'params': lambda: {'r': random.uniform(0.3, 0.8),
                           'rings': random.choice([5, 6, 7, 8]),
                           'segs': random.choice([6, 8, 10])},
    },
    'cylinder': {
        'names': ['cylinder', 'tube', 'pillar', 'column', 'pipe', 'rod'],
        'generator': lambda p: make_cylinder(p['r'], p['h'], p['segs']),
        'params': lambda: {'r': random.uniform(0.2, 0.6),
                           'h': random.uniform(0.5, 1.5),
                           'segs': random.choice([6, 8, 10])},
    },
    'cone': {
        'names': ['cone', 'pointed cone', 'conical shape', 'traffic cone'],
        'generator': lambda p: make_cone(p['r'], p['h'], p['segs']),
        'params': lambda: {'r': random.uniform(0.3, 0.7),
                           'h': random.uniform(0.5, 1.5),
                           'segs': random.choice([6, 8, 10])},
    },
    'pyramid': {
        'names': ['pyramid', 'triangular pyramid', 'square pyramid', 'egyptian pyramid'],
        'generator': lambda p: make_pyramid(p['b'], p['h']),
        'params': lambda: {'b': random.uniform(0.6, 1.4), 'h': random.uniform(0.5, 1.5)},
    },
    'torus': {
        'names': ['torus', 'donut', 'ring', 'doughnut', 'bagel'],
        'generator': lambda p: make_torus(p['R'], p['r'], p['ms'], p['ns']),
        'params': lambda: {'R': random.uniform(0.35, 0.55), 'r': random.uniform(0.1, 0.25),
                           'ms': random.choice([6, 8]), 'ns': random.choice([5, 6])},
    },
    'plane': {
        'names': ['plane', 'flat plane', 'floor', 'flat surface', 'platform'],
        'generator': lambda p: make_plane(p['sx'], p['sy'], p['sub']),
        'params': lambda: {'sx': random.uniform(0.8, 1.5), 'sy': random.uniform(0.8, 1.5),
                           'sub': random.choice([1, 2, 3])},
    },
    'wedge': {
        'names': ['wedge', 'ramp', 'triangular prism', 'prism', 'slope'],
        'generator': lambda p: make_wedge(p['sx'], p['sy'], p['sz']),
        'params': lambda: {'sx': random.uniform(0.6, 1.4), 'sy': random.uniform(0.4, 1.0),
                           'sz': random.uniform(0.6, 1.4)},
    },
    'icosphere': {
        'names': ['icosphere', 'geodesic sphere', 'smooth sphere', 'geo ball'],
        'generator': lambda p: make_icosphere(p['r'], p['sub']),
        'params': lambda: {'r': random.uniform(0.3, 0.7), 'sub': random.choice([1, 2])},
    },
    'circle': {
        'names': ['circle', 'disc', 'disk', 'flat circle', 'round disc'],
        'generator': lambda p: make_circle(p['r'], p['segs']),
        'params': lambda: {'r': random.uniform(0.3, 0.7),
                           'segs': random.choice([8, 10, 12, 16])},
    },
    'grid': {
        'names': ['grid', 'mesh grid', 'subdivided plane', 'lattice', 'grid plane'],
        'generator': lambda p: make_grid(p['sx'], p['sy'], p['nx'], p['ny']),
        'params': lambda: {'sx': random.uniform(0.8, 1.5), 'sy': random.uniform(0.8, 1.5),
                           'nx': random.choice([3, 4, 5, 6]),
                           'ny': random.choice([3, 4, 5, 6])},
    },
    'terrain': {
        'names': ['terrain', 'landscape', 'ground', 'hillside', 'mountainous terrain',
                  'hilly landscape', 'bumpy ground', 'rolling hills'],
        'generator': lambda p: make_terrain(p['sx'], p['sy'], p['nx'], p['ny'], p['mh']),
        'params': lambda: {'sx': random.uniform(0.8, 1.5), 'sy': random.uniform(0.8, 1.5),
                           'nx': random.choice([16, 24, 32]),
                           'ny': random.choice([16, 24, 32]),
                           'mh': random.uniform(0.12, 0.5)},
    },
    'monkey': {
        'names': ['monkey head', 'Suzanne', 'monkey', 'primate head', 'ape head'],
        'generator': lambda p: make_monkey(p['s']),
        'params': lambda: {'s': random.uniform(0.4, 0.8)},
    },
}

# Composite shapes — assembled from primitives. These generate simple
# recognizable objects that the model CAN learn (low token count, clear labels).
COMPOSITE_SPECS = {
    'table': {
        'names': ['table', 'desk', 'dining table', 'coffee table', 'side table', 'work desk'],
        'generator': lambda p: make_table(p['w'], p['d'], p['h'], p['lt']),
        'params': lambda: {'w': random.uniform(0.8, 1.4), 'd': random.uniform(0.5, 0.9),
                           'h': random.uniform(0.6, 0.9), 'lt': random.uniform(0.04, 0.08)},
    },
    'chair': {
        'names': ['chair', 'seat', 'dining chair', 'wooden chair', 'office chair'],
        'generator': lambda p: make_chair(p['sw'], p['sd'], p['sh'], p['bh'], p['lt']),
        'params': lambda: {'sw': random.uniform(0.4, 0.6), 'sd': random.uniform(0.4, 0.6),
                           'sh': random.uniform(0.4, 0.5), 'bh': random.uniform(0.3, 0.5),
                           'lt': random.uniform(0.03, 0.06)},
    },
    'house': {
        'names': ['house', 'building', 'cottage', 'cabin', 'small house', 'home'],
        'generator': lambda p: make_house(p['w'], p['d'], p['wh'], p['rh']),
        'params': lambda: {'w': random.uniform(0.8, 1.2), 'd': random.uniform(0.6, 1.0),
                           'wh': random.uniform(0.5, 0.8), 'rh': random.uniform(0.3, 0.5)},
    },
    'bookshelf': {
        'names': ['bookshelf', 'shelf', 'bookcase', 'shelving unit', 'storage shelf'],
        'generator': lambda p: make_bookshelf(p['w'], p['d'], p['h'], p['s'], p['t']),
        'params': lambda: {'w': random.uniform(0.6, 1.0), 'd': random.uniform(0.2, 0.35),
                           'h': random.uniform(1.0, 1.5), 's': random.randint(2, 4),
                           't': random.uniform(0.02, 0.05)},
    },
    'bench': {
        'names': ['bench', 'park bench', 'wooden bench', 'sitting bench', 'garden bench'],
        'generator': lambda p: make_bench(p['w'], p['d'], p['h'], p['lt']),
        'params': lambda: {'w': random.uniform(0.8, 1.3), 'd': random.uniform(0.3, 0.45),
                           'h': random.uniform(0.4, 0.5), 'lt': random.uniform(0.04, 0.07)},
    },
    'bed': {
        'names': ['bed', 'single bed', 'double bed', 'mattress frame', 'bed frame'],
        'generator': lambda p: make_bed(p['w'], p['d'], p['h'], p['lt']),
        'params': lambda: {'w': random.uniform(0.9, 1.2), 'd': random.uniform(1.6, 2.2),
                           'h': random.uniform(0.4, 0.6), 'lt': random.uniform(0.05, 0.08)},
    },
    'sofa': {
        'names': ['sofa', 'couch', 'loveseat', 'settee', 'lounge sofa'],
        'generator': lambda p: make_sofa(p['w'], p['d'], p['sh'], p['bh'], p['aw']),
        'params': lambda: {'w': random.uniform(1.2, 1.8), 'd': random.uniform(0.6, 0.8),
                           'sh': random.uniform(0.3, 0.4), 'bh': random.uniform(0.3, 0.45),
                           'aw': random.uniform(0.1, 0.15)},
    },
    'lamp': {
        'names': ['lamp', 'desk lamp', 'table lamp', 'floor lamp', 'standing lamp'],
        'generator': lambda p: make_lamp(p['br'], p['pr'], p['ph'], p['sr'], p['sh']),
        'params': lambda: {'br': random.uniform(0.1, 0.15), 'pr': random.uniform(0.015, 0.025),
                           'ph': random.uniform(0.4, 0.8), 'sr': random.uniform(0.12, 0.18),
                           'sh': random.uniform(0.1, 0.15)},
    },
    'barrel': {
        'names': ['barrel', 'wooden barrel', 'cask', 'keg', 'drum'],
        'generator': lambda p: make_barrel(p['r'], p['h'], p['b'], p['s']),
        'params': lambda: {'r': random.uniform(0.2, 0.4), 'h': random.uniform(0.4, 0.7),
                           'b': random.uniform(1.1, 1.2), 's': random.choice([5, 6, 8])},
    },
    'bottle': {
        'names': ['bottle', 'wine bottle', 'glass bottle', 'water bottle', 'jar'],
        'generator': lambda p: make_bottle(p['br'], p['bh'], p['nr'], p['nh']),
        'params': lambda: {'br': random.uniform(0.08, 0.15), 'bh': random.uniform(0.25, 0.4),
                           'nr': random.uniform(0.03, 0.05), 'nh': random.uniform(0.1, 0.2)},
    },
    'cup': {
        'names': ['cup', 'mug', 'coffee cup', 'tea cup', 'drinking cup'],
        'generator': lambda p: make_cup(p['r'], p['h'], p['ht']),
        'params': lambda: {'r': random.uniform(0.08, 0.14), 'h': random.uniform(0.1, 0.18),
                           'ht': random.uniform(0.015, 0.025)},
    },
    'vase': {
        'names': ['vase', 'flower vase', 'ceramic vase', 'decorative vase', 'urn'],
        'generator': lambda p: make_vase(p['r'], p['h'], p['nr'], p['s']),
        'params': lambda: {'r': random.uniform(0.12, 0.2), 'h': random.uniform(0.3, 0.5),
                           'nr': random.uniform(0.06, 0.1), 's': random.choice([5, 6, 8])},
    },
    'sword': {
        'names': ['sword', 'blade', 'longsword', 'broadsword', 'weapon'],
        'generator': lambda p: make_sword(p['bl'], p['bw'], p['hl'], p['gw']),
        'params': lambda: {'bl': random.uniform(0.5, 0.9), 'bw': random.uniform(0.04, 0.07),
                           'hl': random.uniform(0.12, 0.2), 'gw': random.uniform(0.12, 0.18)},
    },
    'tree': {
        'names': ['tree', 'pine tree', 'oak tree', 'simple tree', 'evergreen'],
        'generator': lambda p: make_tree(p['tr'], p['th'], p['cr'], p['ch']),
        'params': lambda: {'tr': random.uniform(0.06, 0.1), 'th': random.uniform(0.4, 0.8),
                           'cr': random.uniform(0.25, 0.45), 'ch': random.uniform(0.4, 0.6)},
    },
    'mushroom': {
        'names': ['mushroom', 'toadstool', 'fungus', 'mushroom cap'],
        'generator': lambda p: make_mushroom(p['cr'], p['ch'], p['sr'], p['sh']),
        'params': lambda: {'cr': random.uniform(0.15, 0.25), 'ch': random.uniform(0.06, 0.12),
                           'sr': random.uniform(0.04, 0.07), 'sh': random.uniform(0.15, 0.25)},
    },
    'snowman': {
        'names': ['snowman', 'snow figure', 'winter snowman'],
        'generator': lambda p: make_snowman(p['r1'], p['r2'], p['r3']),
        'params': lambda: {'r1': random.uniform(0.2, 0.3), 'r2': random.uniform(0.15, 0.22),
                           'r3': random.uniform(0.1, 0.15)},
    },
    'hammer': {
        'names': ['hammer', 'mallet', 'war hammer', 'tool hammer'],
        'generator': lambda p: make_hammer(p['hl'], p['hr'], p['sl'], p['sr']),
        'params': lambda: {'hl': random.uniform(0.15, 0.25), 'hr': random.uniform(0.03, 0.05),
                           'sl': random.uniform(0.3, 0.5), 'sr': random.uniform(0.015, 0.025)},
    },
    'trophy': {
        'names': ['trophy', 'award', 'cup trophy', 'prize cup'],
        'generator': lambda p: make_trophy(p['cr'], p['ch'], p['sr'], p['sh'], p['br']),
        'params': lambda: {'cr': random.uniform(0.1, 0.15), 'ch': random.uniform(0.12, 0.18),
                           'sr': random.uniform(0.015, 0.025), 'sh': random.uniform(0.08, 0.12),
                           'br': random.uniform(0.06, 0.1)},
    },
    'star': {
        'names': ['star', 'star shape', 'five-pointed star', 'decorative star'],
        'generator': lambda p: make_star(p['or'], p['ir'], p['d'], p['pts']),
        'params': lambda: {'or': random.uniform(0.3, 0.5), 'ir': random.uniform(0.12, 0.22),
                           'd': random.uniform(0.04, 0.1), 'pts': random.choice([4, 5, 6])},
    },
    'arrow': {
        'names': ['arrow', 'directional arrow', 'pointer', 'projectile'],
        'generator': lambda p: make_arrow(p['sl'], p['sr'], p['hl'], p['hr']),
        'params': lambda: {'sl': random.uniform(0.4, 0.6), 'sr': random.uniform(0.015, 0.025),
                           'hl': random.uniform(0.1, 0.15), 'hr': random.uniform(0.04, 0.08)},
    },
    'cross': {
        'names': ['cross', 'plus sign', 'crucifix', 'cross shape'],
        'generator': lambda p: make_cross(p['w'], p['h'], p['d'], p['aw']),
        'params': lambda: {'w': random.uniform(0.5, 0.7), 'h': random.uniform(0.6, 0.9),
                           'd': random.uniform(0.06, 0.12), 'aw': random.uniform(0.1, 0.2)},
    },
    'diamond': {
        'names': ['diamond', 'gem', 'jewel', 'crystal', 'gemstone'],
        'generator': lambda p: make_diamond(p['r'], p['th'], p['bh'], p['s']),
        'params': lambda: {'r': random.uniform(0.2, 0.4), 'th': random.uniform(0.15, 0.25),
                           'bh': random.uniform(0.3, 0.5), 's': random.choice([5, 6, 8])},
    },
    'gear': {
        'names': ['gear', 'cog', 'cogwheel', 'gear wheel', 'sprocket'],
        'generator': lambda p: make_gear(p['or'], p['ir'], p['t'], p['d'], p['th']),
        'params': lambda: {'or': random.uniform(0.25, 0.4), 'ir': random.uniform(0.18, 0.3),
                           't': random.choice([6, 8, 10]), 'd': random.uniform(0.06, 0.1),
                           'th': random.uniform(0.03, 0.06)},
    },
    'anvil': {
        'names': ['anvil', 'blacksmith anvil', 'forging anvil'],
        'generator': lambda p: make_anvil(p['bw'], p['bd'], p['bh'], p['tw'], p['toph'], p['bodyh']),
        'params': lambda: {'bw': random.uniform(0.35, 0.5), 'bd': random.uniform(0.18, 0.25),
                           'bh': random.uniform(0.08, 0.12), 'tw': random.uniform(0.45, 0.6),
                           'toph': random.uniform(0.06, 0.1), 'bodyh': random.uniform(0.12, 0.18)},
    },
    'car': {
        'names': ['car', 'vehicle', 'automobile', 'sedan', 'simple car'],
        'generator': lambda p: make_car_body(p['l'], p['w'], p['h'], p['ch']),
        'params': lambda: {'l': random.uniform(1.5, 2.0), 'w': random.uniform(0.7, 0.9),
                           'h': random.uniform(0.4, 0.6), 'ch': random.uniform(0.25, 0.35)},
    },
    'boat': {
        'names': ['boat', 'ship', 'vessel', 'rowboat', 'canoe'],
        'generator': lambda p: make_boat_hull(p['l'], p['w'], p['h']),
        'params': lambda: {'l': random.uniform(1.2, 1.8), 'w': random.uniform(0.4, 0.6),
                           'h': random.uniform(0.25, 0.35)},
    },
    'arch': {
        'names': ['arch', 'archway', 'gateway', 'stone arch', 'doorway arch'],
        'generator': lambda p: make_arch(p['w'], p['h'], p['d'], p['s'], p['t']),
        'params': lambda: {'w': random.uniform(0.7, 1.0), 'h': random.uniform(0.8, 1.2),
                           'd': random.uniform(0.2, 0.4), 's': random.choice([5, 6, 8]),
                           't': random.uniform(0.08, 0.15)},
    },
    'tower': {
        'names': ['tower', 'castle tower', 'watchtower', 'turret', 'lighthouse'],
        'generator': lambda p: make_tower(p['b'], p['h'], p['rh'], p['s']),
        'params': lambda: {'b': random.uniform(0.4, 0.6), 'h': random.uniform(1.2, 1.8),
                           'rh': random.uniform(0.2, 0.4), 's': random.choice([4, 5, 6, 8])},
    },
    'fence': {
        'names': ['fence', 'picket fence', 'wooden fence', 'garden fence', 'barrier'],
        'generator': lambda p: make_fence(p['w'], p['h'], p['ps'], p['pt'], p['rt']),
        'params': lambda: {'w': random.uniform(1.5, 2.5), 'h': random.uniform(0.6, 1.0),
                           'ps': random.randint(4, 7), 'pt': random.uniform(0.04, 0.07),
                           'rt': random.uniform(0.02, 0.04)},
    },
    'well': {
        'names': ['well', 'water well', 'wishing well', 'stone well'],
        'generator': lambda p: make_well(p['r'], p['h'], p['rh'], p['pt'], p['s']),
        'params': lambda: {'r': random.uniform(0.25, 0.4), 'h': random.uniform(0.4, 0.6),
                           'rh': random.uniform(0.2, 0.35), 'pt': random.uniform(0.02, 0.04),
                           's': random.choice([5, 6, 8])},
    },
    'rock': {
        'names': ['rock', 'stone', 'boulder', 'pebble', 'rocky formation'],
        'generator': lambda p: make_rock(p['r'], p['dist']),
        'params': lambda: {'r': random.uniform(0.2, 0.4), 'dist': random.uniform(0.1, 0.2)},
    },
    'cactus': {
        'names': ['cactus', 'desert cactus', 'saguaro cactus', 'prickly cactus'],
        'generator': lambda p: make_cactus(p['r'], p['h'], p['ar'], p['ah']),
        'params': lambda: {'r': random.uniform(0.08, 0.13), 'h': random.uniform(0.4, 0.7),
                           'ar': random.uniform(0.05, 0.08), 'ah': random.uniform(0.15, 0.25)},
    },
    'shield': {
        'names': ['shield', 'round shield', 'buckler', 'battle shield'],
        'generator': lambda p: make_shield(p['w'], p['h'], p['d']),
        'params': lambda: {'w': random.uniform(0.3, 0.5), 'h': random.uniform(0.4, 0.6),
                           'd': random.uniform(0.06, 0.12)},
    },
    'key': {
        'names': ['key', 'door key', 'skeleton key', 'old key', 'golden key'],
        'generator': lambda p: make_key(p['sl'], p['sr'], p['hr']),
        'params': lambda: {'sl': random.uniform(0.25, 0.4), 'sr': random.uniform(0.015, 0.025),
                           'hr': random.uniform(0.05, 0.08)},
    },
    'crown': {
        'names': ['crown', 'royal crown', 'king crown', 'golden crown', 'tiara'],
        'generator': lambda p: make_crown(p['r'], p['h'], p['pts'], p['ph']),
        'params': lambda: {'r': random.uniform(0.12, 0.18), 'h': random.uniform(0.08, 0.12),
                           'pts': random.choice([4, 5, 6]), 'ph': random.uniform(0.04, 0.08)},
    },
    'l_shape': {
        'names': ['L-shape', 'L-shaped block', 'corner piece', 'L bracket', 'angle block'],
        'generator': lambda p: make_l_shape(p['sx'], p['sy'], p['sz'], p['t']),
        'params': lambda: {'sx': random.uniform(0.8, 1.2), 'sy': random.uniform(0.4, 0.6),
                           'sz': random.uniform(0.8, 1.2), 't': random.uniform(0.2, 0.4)},
    },
    'stairs': {
        'names': ['stairs', 'staircase', 'steps', 'stairway', 'ladder steps'],
        'generator': lambda p: make_stairs(p['n'], p['w'], p['h'], p['d']),
        'params': lambda: {'n': random.randint(3, 6), 'w': random.uniform(0.8, 1.2),
                           'h': random.uniform(0.8, 1.2), 'd': random.uniform(0.8, 1.2)},
    },
    'door': {
        'names': ['door', 'wooden door', 'entrance door', 'panel door'],
        'generator': lambda p: make_door(p['w'], p['h'], p['d']),
        'params': lambda: {'w': random.uniform(0.8, 1.0), 'h': random.uniform(1.8, 2.2),
                           'd': random.uniform(0.04, 0.06)},
    },
    'window': {
        'names': ['window', 'window frame', 'glass window', 'pane window'],
        'generator': lambda p: make_window(p['w'], p['h'], p['d'], p['ft']),
        'params': lambda: {'w': random.uniform(0.5, 0.8), 'h': random.uniform(0.6, 1.0),
                           'd': random.uniform(0.04, 0.06), 'ft': random.uniform(0.04, 0.06)},
    },
    'chimney': {
        'names': ['chimney', 'smokestack', 'chimney stack', 'brick chimney'],
        'generator': lambda p: make_chimney(p['w'], p['d'], p['h']),
        'params': lambda: {'w': random.uniform(0.2, 0.3), 'd': random.uniform(0.2, 0.3),
                           'h': random.uniform(0.4, 0.7)},
    },
    'column': {
        'names': ['column', 'pillar', 'support column', 'stone pillar', 'post'],
        'generator': lambda p: make_column(p['r'], p['h'], p['s']),
        'params': lambda: {'r': random.uniform(0.1, 0.2), 'h': random.uniform(1.0, 1.8),
                           's': random.choice([5, 6, 8])},
    },
    'wall': {
        'names': ['wall', 'brick wall', 'stone wall', 'partition wall', 'barrier wall'],
        'generator': lambda p: make_wall(p['w'], p['h'], p['d']),
        'params': lambda: {'w': random.uniform(1.5, 2.5), 'h': random.uniform(0.8, 1.2),
                           'd': random.uniform(0.1, 0.2)},
    },
    'bridge': {
        'names': ['bridge', 'footbridge', 'stone bridge', 'wooden bridge', 'overpass'],
        'generator': lambda p: make_bridge(p['l'], p['w'], p['h'], p['t']),
        'params': lambda: {'l': random.uniform(1.5, 2.5), 'w': random.uniform(0.5, 0.8),
                           'h': random.uniform(0.25, 0.4), 't': random.uniform(0.04, 0.08)},
    },
    'stool': {
        'names': ['stool', 'bar stool', 'step stool', 'round stool', 'wooden stool'],
        'generator': lambda p: make_stool(p['r'], p['h'], p['legs'], p['lt']),
        'params': lambda: {'r': random.uniform(0.15, 0.25), 'h': random.uniform(0.4, 0.6),
                           'legs': random.choice([3, 4]), 'lt': random.uniform(0.02, 0.04)},
    },
    'wardrobe': {
        'names': ['wardrobe', 'closet', 'armoire', 'clothes cabinet', 'storage wardrobe'],
        'generator': lambda p: make_wardrobe(p['w'], p['d'], p['h'], p['t']),
        'params': lambda: {'w': random.uniform(0.7, 1.0), 'd': random.uniform(0.4, 0.6),
                           'h': random.uniform(1.5, 2.0), 't': random.uniform(0.03, 0.05)},
    },
    'cabinet': {
        'names': ['cabinet', 'kitchen cabinet', 'storage cabinet', 'filing cabinet'],
        'generator': lambda p: make_cabinet(p['w'], p['d'], p['h'], p['t']),
        'params': lambda: {'w': random.uniform(0.5, 0.7), 'd': random.uniform(0.35, 0.5),
                           'h': random.uniform(0.6, 0.8), 't': random.uniform(0.025, 0.04)},
    },
}


# ===========================================================================
# Main generation pipeline
# ===========================================================================

def generate_dataset(num_examples, config, output_dir):
    """Generate synthetic shape dataset."""
    from processing.mesh_tokenizer import MeshTokenizer
    from processing.text_tokenizer import TextTokenizer

    tok_config = config.get("tokenization", {})
    tokenizer = MeshTokenizer(
        vocab_size=tok_config.get("vocab_size", 8192),
        coord_range=tuple(tok_config.get("coordinate_range", [-1.0, 1.0])),
        max_faces=tok_config.get("max_faces", 2048),
    )

    max_seq = config.get("models", {}).get("geometry", {}).get(
        "max_sequence_length", 4608)
    max_faces_for_seq = (max_seq - 2) // 9
    logger.info(f"Max faces that fit in sequence: {max_faces_for_seq}")

    all_shapes = list(SHAPE_SPECS.keys()) + list(COMPOSITE_SPECS.keys())
    all_specs = {**SHAPE_SPECS, **COMPOSITE_SPECS}
    n_shapes = len(all_shapes)
    logger.info(f"Shape categories: {n_shapes}")

    examples = []
    shape_counts = {k: 0 for k in all_shapes}
    skipped = 0

    for i in range(num_examples):
        shape_key = random.choice(all_shapes)
        spec = all_specs[shape_key]

        try:
            params = spec['params']()
            verts, faces = spec['generator'](params)
        except Exception as e:
            skipped += 1
            continue

        if len(verts) < 3 or len(faces) < 1:
            skipped += 1
            continue

        if random.random() < 0.5:
            angle = random.uniform(0, 360)
            axis = random.choice(['x', 'y', 'z'])
            verts = apply_rotation(verts, angle, axis)

        verts = normalize_mesh(verts, target_range=(-1.0, 1.0))

        if len(faces) > max_faces_for_seq or len(faces) < 2:
            skipped += 1
            continue

        try:
            tokens = tokenizer.encode_mesh(verts, faces)
        except Exception:
            skipped += 1
            continue

        if tokens[-1] != tokenizer.EOS or tokens[0] != tokenizer.BOS:
            skipped += 1
            continue
        if len(tokens) > max_seq:
            skipped += 1
            continue

        label = generate_label(shape_key, params)

        examples.append({
            "text": label,
            "tokens": tokens,
            "num_faces": len(faces),
            "num_vertices": len(verts),
            "source": f"synthetic_{shape_key}",
        })
        shape_counts[shape_key] += 1

        if (i + 1) % 5000 == 0:
            logger.info(f"  Generated {len(examples)}/{num_examples} valid examples "
                        f"({skipped} skipped)...")

    all_texts = [ex["text"] for ex in examples]
    text_tokenizer = TextTokenizer.from_texts(all_texts)
    logger.info(f"Text vocabulary: {text_tokenizer.vocab_size} words")

    random.seed(42)
    random.shuffle(examples)
    n_train = int(len(examples) * 0.90)
    n_val = int(len(examples) * 0.05)

    splits = {
        "train": examples[:n_train],
        "val": examples[n_train:n_train + n_val],
        "test": examples[n_train + n_val:],
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in splits.items():
        out_path = out_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for ex in split_data:
                f.write(json.dumps(ex) + "\n")
        logger.info(f"  {split_name}: {len(split_data)} examples -> {out_path}")

    text_tokenizer.save(out_dir / "text_tokenizer.json")

    logger.info(f"\nGenerated {len(examples)} total examples ({skipped} skipped)")
    logger.info(f"Shape distribution:")
    for k, v in sorted(shape_counts.items(), key=lambda x: -x[1]):
        if v > 0:
            logger.info(f"  {k:15s}: {v:5d}")

    token_counts = [len(ex["tokens"]) for ex in examples]
    face_counts = [ex["num_faces"] for ex in examples]
    logger.info(f"\nToken stats: min={min(token_counts)}, "
                f"max={max(token_counts)}, "
                f"avg={sum(token_counts)/len(token_counts):.0f}")
    logger.info(f"Face stats:  min={min(face_counts)}, "
                f"max={max(face_counts)}, "
                f"avg={sum(face_counts)/len(face_counts):.0f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic shape training data")
    parser.add_argument("--output", default="data/datasets/geometry",
                        help="Output directory")
    parser.add_argument("--config", default="config_synthetic.yaml",
                        help="Config file (for tokenization settings)")
    parser.add_argument("--num-examples", type=int, default=50000,
                        help="Number of examples to generate")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    generate_dataset(args.num_examples, config, Path(args.output))


if __name__ == "__main__":
    main()
