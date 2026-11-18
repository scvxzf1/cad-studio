"""Geometric sanity, DFM validation, and assembly interference / collision checker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import manifold3d
import numpy as np
import trimesh

from .inspector import shape_to_trimesh


def trimesh_to_manifold(mesh: trimesh.Trimesh) -> manifold3d.Manifold:
    """Convert a trimesh.Trimesh into a manifold3d.Manifold (healing minor non-manifold seams if needed)."""
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    m_mesh = manifold3d.Mesh(vert_properties=verts, tri_verts=faces)
    return manifold3d.Manifold(m_mesh)


def manifold_to_trimesh(m: manifold3d.Manifold) -> trimesh.Trimesh:
    """Convert a manifold3d.Manifold back into a trimesh.Trimesh."""
    out = m.to_mesh()
    verts = np.asarray(out.vert_properties[:, :3], dtype=np.float64)
    faces = np.asarray(out.tri_verts, dtype=np.int64)
    if len(verts) == 0 or len(faces) == 0:
        return trimesh.Trimesh()
    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)


def load_any_as_trimesh(target: Any) -> trimesh.Trimesh:
    """Load a build123d Part, trimesh.Trimesh, or file path (.step/.stl/.3mf) into trimesh.Trimesh."""
    if isinstance(target, trimesh.Trimesh):
        return target
    if isinstance(target, (str, Path)):
        p = Path(target).expanduser().resolve()
        ext = p.suffix.lower()
        if ext in (".step", ".stp", ".brep"):
            import build123d as bd
            part = bd.import_brep(str(p)) if ext == ".brep" else bd.import_step(str(p))
            return shape_to_trimesh(part)
        loaded = trimesh.load(str(p), force="mesh")
        return loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
    return shape_to_trimesh(target)


def verify_part_geometry(
    part: Any,
    expected_solids: int = 1,
    ref_files: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Validate solid connectivity, B-Rep validity, watertightness, and optional mating interference."""
    issues: list[str] = []
    warnings: list[str] = []

    is_brep = hasattr(part, "solids") and hasattr(part, "is_valid")
    if is_brep:
        solids = part.solids()
        solids_count = len(solids)
        is_valid = bool(part.is_valid)
        vol = float(part.volume)
        bb = part.bounding_box()
        bmin = [round(bb.min.X, 3), round(bb.min.Y, 3), round(bb.min.Z, 3)]
        bmax = [round(bb.max.X, 3), round(bb.max.Y, 3), round(bb.max.Z, 3)]
        bsize = [round(bb.size.X, 3), round(bb.size.Y, 3), round(bb.size.Z, 3)]
        mesh = shape_to_trimesh(part)
    else:
        mesh = load_any_as_trimesh(part)
        solids_count = len(mesh.split(only_watertight=False)) if len(mesh.faces) > 0 else 0
        is_valid = bool(mesh.is_watertight)
        vol = float(mesh.volume) if len(mesh.faces) > 0 else 0.0
        bmin = [round(float(x), 3) for x in mesh.bounds[0]] if len(mesh.faces) > 0 else [0, 0, 0]
        bmax = [round(float(x), 3) for x in mesh.bounds[1]] if len(mesh.faces) > 0 else [0, 0, 0]
        bsize = [round(bmax[i] - bmin[i], 3) for i in range(3)]

    if not is_valid:
        issues.append("几何有效性检查未通过 (B-Rep invalid 或 STL 非水密破面)！")
    if vol <= 1e-3:
        issues.append(f"实体体积异常 ({vol:.4f} mm³ <= 0)，可能被完全切空！")
    if expected_solids > 0 and solids_count != expected_solids:
        issues.append(
            f"连通实体数量为 {solids_count}（期望为 {expected_solids}），请检查是否有悬空碎块或零件被意外切断分离！"
        )
    if not mesh.is_watertight:
        warnings.append("导出三角网格非严格水密 (non-watertight)，请在切片前留意修复提示。")

    # Assembly collision / interference check against reference parts (cached LOD + C++ ROI pre-crop!)
    interferences: list[dict[str, Any]] = []
    if ref_files:
        from .large_cad import check_large_file_warnings, crop_manifold_to_bbox, load_mesh_cached

        warnings.extend(check_large_file_warnings(ref_files))
        try:
            m_part = trimesh_to_manifold(mesh)
            p_min_arr = np.asarray(bmin, dtype=float)
            p_max_arr = np.asarray(bmax, dtype=float)
            for rf in ref_files:
                rf_path = Path(rf).expanduser().resolve()
                if not rf_path.exists():
                    warnings.append(f"参考文件不存在，跳过干涉检查: {rf_path}")
                    continue
                ref_mesh = load_mesh_cached(rf_path, tier="work")
                r_min, r_max = ref_mesh.bounds
                # Instant O(1) AABB separation test: skip boolean completely if bounding boxes don't touch!
                if np.any(p_max_arr < r_min - 1e-4) or np.any(r_max < p_min_arr - 1e-4):
                    interferences.append({"ref_file": str(rf_path), "collision_volume_mm3": 0.0, "has_collision": False})
                    continue

                m_ref = trimesh_to_manifold(ref_mesh)
                if len(ref_mesh.faces) > 25_000:
                    m_ref = crop_manifold_to_bbox(m_ref, p_min_arr, p_max_arr, margin=2.0)
                inter = m_part ^ m_ref  # manifold3d intersection operator
                inter_mesh = manifold_to_trimesh(inter)
                inter_vol = float(inter_mesh.volume) if len(inter_mesh.faces) > 0 else 0.0
                entry: dict[str, Any] = {
                    "ref_file": str(rf_path),
                    "collision_volume_mm3": round(inter_vol, 4),
                    "has_collision": inter_vol > 0.01,
                }
                if inter_vol > 0.01:
                    ibmin, ibmax = inter_mesh.bounds
                    entry["collision_bbox_min_mm"] = [round(float(x), 3) for x in ibmin]
                    entry["collision_bbox_max_mm"] = [round(float(x), 3) for x in ibmax]
                    issues.append(
                        f"装配干涉警告: 与参考件 `{rf_path.name}` 存在 {inter_vol:.2f} mm³ 体积碰撞 "
                        f"(范围 X[{ibmin[0]:.1f},{ibmax[0]:.1f}] Y[{ibmin[1]:.1f},{ibmax[1]:.1f}] Z[{ibmin[2]:.1f},{ibmax[2]:.1f}])！"
                    )
                interferences.append(entry)
        except Exception as exc:
            warnings.append(f"干涉布尔计算跳过: {exc}")

    return {
        "ok": len(issues) == 0,
        "is_valid": is_valid,
        "is_watertight": bool(mesh.is_watertight),
        "solids_count": solids_count,
        "volume_mm3": round(vol, 2),
        "bbox": {
            "min_mm": bmin,
            "max_mm": bmax,
            "size_mm": bsize,
        },
        "interferences": interferences,
        "issues": issues,
        "warnings": warnings,
    }
