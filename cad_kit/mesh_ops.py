"""Watertight STL / 3MF mesh boolean modification helpers powered by manifold3d.

Allows modifying existing STL/3MF meshes directly using precision build123d cutters
and additive bosses/brackets when full STEP reverse-reconstruction is not needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import trimesh

from .verify import load_any_as_trimesh, manifold_to_trimesh, trimesh_to_manifold


def modify_mesh(
    base_mesh_or_path: Any,
    add_parts: Iterable[Any] = (),
    cut_parts: Iterable[Any] = (),
    translate_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> trimesh.Trimesh:
    """Apply additive and subtractive 3D booleans to an existing STL/3MF mesh cleanly."""
    base_mesh = load_any_as_trimesh(base_mesh_or_path).copy()
    if any(abs(v) > 1e-9 for v in translate_xyz):
        base_mesh.apply_translation(np.asarray(translate_xyz, dtype=float))

    m = trimesh_to_manifold(base_mesh)
    for add_item in add_parts:
        if add_item is not None:
            m = m + trimesh_to_manifold(load_any_as_trimesh(add_item))
    for cut_item in cut_parts:
        if cut_item is not None:
            m = m - trimesh_to_manifold(load_any_as_trimesh(cut_item))

    return manifold_to_trimesh(m)


def center_mesh_on_bed(mesh_or_path: Any) -> tuple[trimesh.Trimesh, tuple[float, float, float]]:
    """Translate a mesh so XY is centered at (0, 0) and bottom (-Z) rests at Z = 0.

    Returns (aligned_mesh, applied_offset_xyz).
    """
    mesh = load_any_as_trimesh(mesh_or_path).copy()
    bmin, bmax = mesh.bounds
    cx = 0.5 * (bmin[0] + bmax[0])
    cy = 0.5 * (bmin[1] + bmax[1])
    z0 = bmin[2]
    offset = (-float(cx), -float(cy), -float(z0))
    mesh.apply_translation(np.asarray(offset, dtype=float))
    return mesh, offset
