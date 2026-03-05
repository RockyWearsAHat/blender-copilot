"""Render synthetic 3D shapes to small images for visual grounding.

Produces 64x64 RGB images of each shape from a consistent camera angle.
Uses matplotlib's 3D projection — no GPU or Blender required.

These images are paired with text descriptions during training so the
model learns to associate visual features with words (e.g. "chair" →
image of a chair shape).

Usage:
    from processing.render_shapes import render_mesh_to_image
    img = render_mesh_to_image(vertices, faces)  # (64, 64, 3) uint8
"""

import io
import math
from typing import Optional

import numpy as np


def render_mesh_to_image(vertices: list[list[float]],
                         faces: list[list[int]],
                         size: int = 64,
                         elevation: float = 25.0,
                         azimuth: float = 45.0,
                         bg_color: str = '#f0f0f0') -> np.ndarray:
    """Render a mesh to a small RGB image using matplotlib.

    Args:
        vertices: List of [x, y, z] coordinates
        faces: List of vertex index triples (triangles)
        size: Output image size (square)
        elevation: Camera elevation angle in degrees
        azimuth: Camera azimuth angle in degrees
        bg_color: Background color

    Returns:
        (size, size, 3) uint8 numpy array
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        return _render_simple_projection(vertices, faces, size)

    verts = np.array(vertices)
    if len(verts) == 0:
        return np.ones((size, size, 3), dtype=np.uint8) * 240

    fig = plt.figure(figsize=(1, 1), dpi=size)
    ax = fig.add_subplot(111, projection='3d')

    # Build polygon collection from faces
    polys = []
    for face in faces:
        valid = [fi for fi in face if fi < len(verts)]
        if len(valid) >= 3:
            polys.append([verts[vi] for vi in valid])

    if polys:
        collection = Poly3DCollection(polys,
                                       facecolors='#6699cc',
                                       edgecolors='#334d66',
                                       linewidths=0.3,
                                       alpha=0.9)
        ax.add_collection3d(collection)

    # Set camera and limits
    ax.view_init(elev=elevation, azim=azimuth)

    # Auto-scale to mesh bounds
    if len(verts) > 0:
        v_min = verts.min(axis=0)
        v_max = verts.max(axis=0)
        center = (v_min + v_max) / 2
        extent = max((v_max - v_min).max(), 0.1) * 0.6
        ax.set_xlim(center[0] - extent, center[0] + extent)
        ax.set_ylim(center[1] - extent, center[1] + extent)
        ax.set_zlim(center[2] - extent, center[2] + extent)

    ax.set_facecolor(bg_color)
    ax.grid(False)
    ax.set_axis_off()
    fig.patch.set_facecolor(bg_color)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Render to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0,
                facecolor=bg_color, dpi=size)
    plt.close(fig)
    buf.seek(0)

    # Read back as numpy array
    from PIL import Image
    img = Image.open(buf).convert('RGB').resize((size, size))
    return np.array(img, dtype=np.uint8)


def _render_simple_projection(vertices: list[list[float]],
                                faces: list[list[int]],
                                size: int = 64) -> np.ndarray:
    """Fallback: simple orthographic projection without matplotlib.

    Creates a basic wireframe-like image by projecting faces onto a 2D plane.
    """
    img = np.ones((size, size, 3), dtype=np.uint8) * 240  # light gray bg

    if not vertices or not faces:
        return img

    verts = np.array(vertices, dtype=np.float32)

    # Simple rotation: 30° around X, 45° around Y
    cos30, sin30 = math.cos(math.radians(30)), math.sin(math.radians(30))
    cos45, sin45 = math.cos(math.radians(45)), math.sin(math.radians(45))

    # Rotate around Y
    x = verts[:, 0] * cos45 + verts[:, 2] * sin45
    z = -verts[:, 0] * sin45 + verts[:, 2] * cos45
    y = verts[:, 1]

    # Rotate around X
    y2 = y * cos30 - z * sin30
    z2 = y * sin30 + z * cos30

    # Project to 2D (orthographic)
    proj_x = x
    proj_y = -y2  # flip Y for image coords

    # Normalize to image space with margin
    if len(proj_x) > 0:
        margin = 4
        x_min, x_max = proj_x.min(), proj_x.max()
        y_min, y_max = proj_y.min(), proj_y.max()
        x_range = max(x_max - x_min, 0.001)
        y_range = max(y_max - y_min, 0.001)
        scale = min((size - 2 * margin) / x_range, (size - 2 * margin) / y_range)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        px = ((proj_x - cx) * scale + size / 2).astype(int)
        py = ((proj_y - cy) * scale + size / 2).astype(int)

        # Draw faces as filled triangles (simple scanline)
        for face in faces:
            valid = [fi for fi in face if fi < len(verts)]
            if len(valid) >= 3:
                # Simple: draw edges
                for j in range(len(valid)):
                    i1, i2 = valid[j], valid[(j + 1) % len(valid)]
                    _draw_line(img, px[i1], py[i1], px[i2], py[i2],
                               color=(102, 153, 204))

    return img


def _draw_line(img, x0, y0, x1, y1, color=(0, 0, 0)):
    """Bresenham's line on a numpy image."""
    h, w = img.shape[:2]
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    steps = 0
    max_steps = dx + dy + 1

    while steps < max_steps:
        if 0 <= x0 < w and 0 <= y0 < h:
            img[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
        steps += 1


def save_image(img: np.ndarray, path: str):
    """Save a numpy RGB image to disk."""
    try:
        from PIL import Image
        Image.fromarray(img).save(path)
    except ImportError:
        # Fallback: save as raw numpy
        np.save(path.replace('.png', '.npy'), img)


def render_and_encode(vertices, faces, size=64) -> list[int]:
    """Render mesh to image and return as flat uint8 list for JSON storage.

    Stores as a flat list of R,G,B values. Small at 64x64 = 12,288 values.
    Compressed well in JSONL since backgrounds are uniform.
    """
    img = render_mesh_to_image(vertices, faces, size=size)
    return img.flatten().tolist()


def decode_image(flat_list: list[int], size: int = 64) -> np.ndarray:
    """Reconstruct image from flat list."""
    return np.array(flat_list, dtype=np.uint8).reshape(size, size, 3)
