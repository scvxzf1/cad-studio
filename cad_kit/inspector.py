"""Reverse-engineering & geometry inspection probe for existing STEP / STL / 3MF files.

Extracts:
  - Bounding box, origin alignment / offset warning, volume, surface area, solid/mesh health
  - Major planar datum surfaces (+Z/-Z/+Y/-Y/+X/-X step heights & areas)
  - Analytic cylindrical holes & bosses from STEP B-Rep (center, axis, diameter, span, fastener hint)
  - Cross-section circle-fitted holes & wall profiles from tessellated STL/3MF meshes
  - Annotated 2D CAD cross-section blueprint (`inspect_slices.png`) with 1mm/5mm/10mm grid
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import trimesh

from .tolerances import FASTENER_TABLE


_CJK_FONT_CACHE: str | None = None


def _configure_cjk_font() -> str:
    """Configure matplotlib to render Chinese labels cleanly on Linux (cached once per process)."""
    global _CJK_FONT_CACHE
    if _CJK_FONT_CACHE is not None:
        return _CJK_FONT_CACHE
    preferred_paths = [
        "/usr/share/fonts/adobe-source-han-sans/SourceHanSansCN-Regular.otf",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/wenquanyi/wqy-zenhei/wqy-zenhei.ttc",
    ]
    for fp in preferred_paths:
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            prop = font_manager.FontProperties(fname=fp)
            name = prop.get_name()
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            _CJK_FONT_CACHE = name
            return name
    plt.rcParams["axes.unicode_minus"] = False
    _CJK_FONT_CACHE = "DejaVu Sans"
    return _CJK_FONT_CACHE


def _guess_fastener_hint(diameter: float, is_hole: bool) -> str:
    if not is_hole:
        return ""
    scored: list[tuple[float, str]] = []
    for sz, f in FASTENER_TABLE.items():
        if abs(diameter - f.iso_close_d) <= 0.16:
            scored.append((abs(diameter - f.iso_close_d), f"{sz}通孔(精)"))
        if abs(diameter - f.iso_medium_d) <= 0.22:
            scored.append((abs(diameter - f.iso_medium_d) + 0.02, f"{sz}通孔"))
        if abs(diameter - f.nominal_d) <= 0.10:
            scored.append((abs(diameter - f.nominal_d) + 0.03, f"{sz}公称孔"))
        if abs(diameter - f.cnc_tap_drill_d) <= 0.12:
            scored.append((abs(diameter - f.cnc_tap_drill_d) + 0.04, f"{sz}攻丝底孔"))
        if abs(diameter - f.fdm_self_tap_d) <= 0.12:
            scored.append((abs(diameter - f.fdm_self_tap_d) + 0.04, f"{sz}自攻孔"))
        if abs(diameter - f.heatset_d) <= 0.15:
            scored.append((abs(diameter - f.heatset_d) + 0.05, f"{sz}热熔螺母孔"))
        if diameter >= 3.6 and abs(diameter - (f.shcs_head_d + 0.5)) <= 0.55:
            scored.append((abs(diameter - (f.shcs_head_d + 0.5)) + 0.25, f"{sz}沉头帽孔"))
    scored.sort(key=lambda item: item[0])
    seen: list[str] = []
    for _, label in scored:
        if label not in seen:
            seen.append(label)
        if len(seen) >= 2:
            break
    return " / ".join(seen)


def shape_to_trimesh(part: Any, angular_deflection: float = 0.1, linear_deflection: float = 0.03) -> trimesh.Trimesh:
    """Convert a build123d Shape/Part into a clean trimesh.Trimesh."""
    verts, tris = part.tessellate(tolerance=linear_deflection, angular_tolerance=angular_deflection)
    v_arr = np.array([[float(v.X), float(v.Y), float(v.Z)] for v in verts], dtype=np.float64)
    f_arr = np.array(tris, dtype=np.int64)
    if len(v_arr) == 0 or len(f_arr) == 0:
        return trimesh.Trimesh()
    mesh = trimesh.Trimesh(vertices=v_arr, faces=f_arr, process=True)
    return mesh


def _axis_label(vec: np.ndarray) -> str:
    v = vec / (np.linalg.norm(vec) + 1e-12)
    axes = [
        (np.array([1, 0, 0]), "+X (右)"),
        (np.array([-1, 0, 0]), "-X (左)"),
        (np.array([0, 1, 0]), "+Y (后)"),
        (np.array([0, -1, 0]), "-Y (前)"),
        (np.array([0, 0, 1]), "+Z (上)"),
        (np.array([0, 0, -1]), "-Z (下)"),
    ]
    for ref, name in axes:
        if float(np.dot(v, ref)) > 0.985:
            return name
    return f"({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})"


def _inspect_step_brep(path: Path, min_solid_diag_mm: float = 0.0) -> tuple[dict[str, Any], trimesh.Trimesh]:
    import build123d as bd
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from .large_cad import load_mesh_cached, load_step_smart

    suffix = path.suffix.lower()
    part, step_meta = load_step_smart(path, min_solid_diag_mm=min_solid_diag_mm, auto_defeature_large=True)
    mesh = load_mesh_cached(path, tier="work", min_solid_diag_mm=step_meta["min_solid_diag_mm"])

    solids = part.solids()
    bb = part.bounding_box()
    bmin = [round(bb.min.X, 3), round(bb.min.Y, 3), round(bb.min.Z, 3)]
    bmax = [round(bb.max.X, 3), round(bb.max.Y, 3), round(bb.max.Z, 3)]
    bsize = [round(bb.size.X, 3), round(bb.size.Y, 3), round(bb.size.Z, 3)]
    bcenter = [round((bmin[i] + bmax[i]) / 2.0, 3) for i in range(3)]

    all_faces = part.faces()
    total_faces_count = len(all_faces)
    # Fast pre-filter when inspecting huge STEP assemblies (e.g. 20,000+ SMD/thread faces):
    # only inspect structural faces above adaptive area threshold, capped at top 900 largest faces!
    if total_faces_count > 700:
        sized_faces = [(float(f.area), f) for f in all_faces]
        sized_faces.sort(key=lambda item: -item[0])
        inspect_faces = [f for area, f in sized_faces[:900] if area >= 0.4]
    else:
        inspect_faces = list(all_faces)

    # 1. Extract major planar faces
    plane_groups: dict[tuple[str, float], float] = {}
    for f in inspect_faces:
        if f.geom_type == bd.GeomType.PLANE:
            c = f.center()
            n = f.normal_at(c)
            nv = np.array([n.X, n.Y, n.Z], dtype=float)
            lbl = _axis_label(nv)
            # Coordinate along primary axis or signed origin distance
            dist = round(float(nv[0] * c.X + nv[1] * c.Y + nv[2] * c.Z), 2)
            # Express actual coordinate for principal axes
            if lbl.startswith("-"):
                coord = round(-dist, 2)
            else:
                coord = dist
            key = (lbl, coord)
            plane_groups[key] = plane_groups.get(key, 0.0) + float(f.area)

    major_planes = [
        {"orientation": k[0], "coord_mm": k[1], "area_mm2": round(area, 2)}
        for k, area in sorted(plane_groups.items(), key=lambda kv: -kv[1])
        if area >= 1.0
    ][:18]

    # 2. Extract analytic cylindrical features (holes vs bosses vs fillets)
    cyl_buckets: list[dict[str, Any]] = []
    for f in inspect_faces:
        if f.geom_type != bd.GeomType.CYLINDER:
            continue
        try:
            surf = BRepAdaptor_Surface(f.wrapped, True)
            cyl = surf.Cylinder()
            radius = float(cyl.Radius())
            ax_loc = cyl.Axis().Location()
            ax_dir = cyl.Axis().Direction()
            loc = np.array([ax_loc.X(), ax_loc.Y(), ax_loc.Z()], dtype=float)
            dvec = np.array([ax_dir.X(), ax_dir.Y(), ax_dir.Z()], dtype=float)
            dvec = dvec / (np.linalg.norm(dvec) + 1e-12)
            # Canonicalize axis direction so +X / +Y / +Z is positive
            for comp in dvec:
                if abs(comp) > 1e-4:
                    if comp < 0:
                        dvec = -dvec
                    break

            # Determine concave (hole) vs convex (boss) using outward face normal
            fc = f.center()
            fn = f.normal_at(fc)
            fpt = np.array([fc.X, fc.Y, fc.Z], dtype=float)
            nvec = np.array([fn.X, fn.Y, fn.Z], dtype=float)
            # Vector from surface point perpendicularly toward cylinder axis
            v_rel = fpt - loc
            v_perp = v_rel - np.dot(v_rel, dvec) * dvec
            to_axis = -v_perp / (np.linalg.norm(v_perp) + 1e-12)
            is_concave = float(np.dot(nvec, to_axis)) > 0.0

            # Sample face vertices to measure axial span and angular wrap around axis
            f_verts = np.array([[v.X, v.Y, v.Z] for v in f.vertices()], dtype=float)
            if len(f_verts) < 2:
                f_bb = f.bounding_box()
                f_verts = np.array(
                    [[f_bb.min.X, f_bb.min.Y, f_bb.min.Z], [f_bb.max.X, f_bb.max.Y, f_bb.max.Z]],
                    dtype=float,
                )
            t_vals = np.dot(f_verts - loc, dvec)
            t_min, t_max = float(np.min(t_vals)), float(np.max(t_vals))

            # Merge coaxial same-radius same-concavity faces (e.g. two 180-deg half cylinders in STEP!)
            merged = False
            for b in cyl_buckets:
                if (
                    b["is_concave"] == is_concave
                    and abs(b["radius"] - radius) < 0.015
                    and abs(float(np.dot(b["dvec"], dvec))) > 0.998
                ):
                    diff = b["loc"] - loc
                    perp_dist = np.linalg.norm(diff - np.dot(diff, dvec) * dvec)
                    if perp_dist < 0.03 and not (t_max < b["t_min"] - 0.5 or t_min > b["t_max"] + 0.5):
                        b["t_min"] = min(b["t_min"], t_min)
                        b["t_max"] = max(b["t_max"], t_max)
                        b["area"] += float(f.area)
                        merged = True
                        break
            if not merged:
                cyl_buckets.append(
                    {
                        "loc": loc,
                        "dvec": dvec,
                        "radius": radius,
                        "is_concave": is_concave,
                        "t_min": t_min,
                        "t_max": t_max,
                        "area": float(f.area),
                    }
                )
        except Exception:
            continue

    cylinders_out: list[dict[str, Any]] = []
    for b in cyl_buckets:
        r = b["radius"]
        span = max(0.0, b["t_max"] - b["t_min"])
        full_cyl_area = 2.0 * math.pi * r * max(span, 1e-3)
        wrap_deg = min(360.0, (b["area"] / full_cyl_area) * 360.0) if span > 0.05 else 360.0
        # Filter out minor 90-degree edge fillets unless large bore
        if wrap_deg < 170.0:
            continue
        mid_t = 0.5 * (b["t_min"] + b["t_max"])
        center_pt = b["loc"] + mid_t * b["dvec"]
        diam = round(2.0 * r, 3)
        kind = "hole (内孔)" if b["is_concave"] else "boss (外圆柱)"
        axis_str = _axis_label(b["dvec"]).split(" ")[0].lstrip("+")
        cylinders_out.append(
            {
                "type": kind,
                "axis": axis_str,
                "center_mm": [round(float(x), 3) for x in center_pt],
                "diameter_mm": diam,
                "length_mm": round(span, 3),
                "wrap_deg": round(wrap_deg, 1),
                "hint": _guess_fastener_hint(diam, b["is_concave"]),
            }
        )

    cylinders_out.sort(key=lambda c: (0 if "hole" in c["type"] else 1, c["diameter_mm"], c["center_mm"]))

    info = {
        "file": str(path),
        "file_size_mb": step_meta["file_size_mb"],
        "format": suffix.lstrip(".").upper(),
        "representation": "B-Rep Exact Solid (OpenCASCADE)",
        "solids_count": len(solids),
        "total_raw_solids": step_meta["total_solids"],
        "filtered_micro_solids": step_meta["filtered_micro_solids"],
        "faces_count": total_faces_count,
        "is_valid": bool(part.is_valid),
        "volume_mm3": round(float(mesh.volume), 2) if total_faces_count > 1500 else round(float(part.volume), 2),
        "area_mm2": round(float(mesh.area), 2) if total_faces_count > 1500 else round(float(part.area), 2),
        "bbox": {
            "min_mm": bmin,
            "max_mm": bmax,
            "size_mm": bsize,
            "center_mm": bcenter,
        },
        "major_planes": major_planes,
        "cylindrical_features": cylinders_out[:40],
    }
    return info, mesh


def _extract_mesh_planes_and_holes(mesh: trimesh.Trimesh) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Cluster planar surfaces and slice-detect circular holes on tessellated STL/3MF meshes."""
    if len(mesh.faces) == 0:
        return [], []

    normals = mesh.face_normals
    areas = mesh.area_faces
    centers = mesh.triangles_center

    plane_bins: dict[tuple[str, float], float] = {}
    z_levels: list[float] = []
    for n, a, c in zip(normals, areas, centers):
        if a < 1e-3:
            continue
        lbl = _axis_label(n)
        if lbl.startswith("("):
            continue
        dist = float(np.dot(n, c))
        coord = round(-dist if lbl.startswith("-") else dist, 2)
        # Snap within 0.05mm
        coord_snap = round(coord * 20.0) / 20.0
        key = (lbl, coord_snap)
        plane_bins[key] = plane_bins.get(key, 0.0) + float(a)

    major_planes = [
        {"orientation": k[0], "coord_mm": round(k[1], 3), "area_mm2": round(v, 2)}
        for k, v in sorted(plane_bins.items(), key=lambda kv: -kv[1])
        if v >= 2.0
    ][:18]

    for p in major_planes:
        if "Z" in p["orientation"]:
            z_levels.append(p["coord_mm"])
    z_levels = sorted(set(z_levels))

    # Slice along X, Y, Z planes to fit circles to interior polygon holes
    bmin, bmax = mesh.bounds
    detected_holes: list[dict[str, Any]] = []

    for axis_idx, axis_name in [(2, "Z"), (1, "Y"), (0, "X")]:
        lo, hi = float(bmin[axis_idx]), float(bmax[axis_idx])
        span = hi - lo
        if span < 0.5:
            continue
        sample_coords = [lo + span * frac for frac in (0.15, 0.35, 0.5, 0.65, 0.85)]
        if axis_idx == 2 and len(z_levels) >= 2:
            for i in range(len(z_levels) - 1):
                if z_levels[i + 1] - z_levels[i] > 0.4:
                    sample_coords.append(0.5 * (z_levels[i] + z_levels[i + 1]))

        normal = np.zeros(3)
        normal[axis_idx] = 1.0
        for val in sorted(set(round(x, 3) for x in sample_coords)):
            origin = np.zeros(3)
            origin[axis_idx] = val
            try:
                sec = mesh.section(plane_origin=origin, plane_normal=normal)
                if sec is None:
                    continue
                planar, to_3d = sec.to_2D()
                from shapely.geometry import Polygon as ShapelyPolygon
                for poly in planar.polygons_full:
                    for interior in poly.interiors:
                        ring_area = float(ShapelyPolygon(interior).area)
                        perim = float(interior.length)
                        if ring_area < 0.5 or perim < 1.0:
                            continue
                        circularity = 4.0 * math.pi * ring_area / (perim * perim)
                        if circularity < 0.86:
                            continue
                        # Least-squares circle fit on polygon vertices for accurate diameter on faceted STLs!
                        pts2d = np.asarray(interior.coords[:-1], dtype=float)
                        cx, cy = float(np.mean(pts2d[:, 0])), float(np.mean(pts2d[:, 1]))
                        radii = np.linalg.norm(pts2d - np.array([cx, cy]), axis=1)
                        # Circumscribed radius compensates polygon chord faceting in STL exports
                        r_est = float(np.percentile(radii, 85))
                        diam = round(2.0 * r_est, 3)
                        pt3 = (to_3d @ np.array([cx, cy, 0.0, 1.0]))[:3]

                        # Merge across adjacent slice heights along the same axis
                        merged = False
                        for h in detected_holes:
                            if h["axis"] == axis_name and abs(h["diameter_mm"] - diam) < 0.18:
                                other = np.array(h["center_mm"])
                                diff = pt3 - other
                                diff[axis_idx] = 0.0
                                if np.linalg.norm(diff) < 0.25:
                                    h["_coords"].append(val)
                                    merged = True
                                    break
                        if not merged:
                            detected_holes.append(
                                {
                                    "type": "hole (STL截面拟合内孔)",
                                    "axis": axis_name,
                                    "center_mm": [round(float(v), 3) for v in pt3],
                                    "diameter_mm": diam,
                                    "_coords": [val],
                                    "hint": _guess_fastener_hint(diam, True),
                                }
                            )
            except Exception:
                continue

    for h in detected_holes:
        vals = h.pop("_coords")
        ax_i = {"X": 0, "Y": 1, "Z": 2}[h["axis"]]
        h["center_mm"][ax_i] = round(0.5 * (min(vals) + max(vals)), 3)
        h["sampled_span_mm"] = round(max(vals) - min(vals), 3)

    return major_planes, detected_holes[:35]


def render_inspection_slices(mesh: trimesh.Trimesh, out_png: Path, title: str = "") -> Path:
    """Render a 4-panel 2D cross-section engineering measurement sheet with 1mm/5mm/10mm CAD grid."""
    _configure_cjk_font()
    bmin, bmax = mesh.bounds
    bsize = bmax - bmin
    bcenter = 0.5 * (bmin + bmax)

    specs = [
        ("XY 低位横截面 (Z = 25% 高度)", np.array([0, 0, bmin[2] + 0.25 * bsize[2]]), np.array([0, 0, 1]), ("X (左-右+) [mm]", "Y (前-后+) [mm]"), (0, 1)),
        ("XY 中位横截面 (Z = 50% 高度)", np.array([0, 0, bmin[2] + 0.50 * bsize[2]]), np.array([0, 0, 1]), ("X (左-右+) [mm]", "Y (前-后+) [mm]"), (0, 1)),
        ("XZ 中心正剖面 (Y = 中心)", np.array([0, bcenter[1], 0]), np.array([0, 1, 0]), ("X (左-右+) [mm]", "Z (下-上+) [mm]"), (0, 2)),
        ("YZ 中心侧剖面 (X = 中心)", np.array([bcenter[0], 0, 0]), np.array([1, 0, 0]), ("Y (前-后+) [mm]", "Z (下-上+) [mm]"), (1, 2)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=160)
    fig.patch.set_facecolor("#f8fafc")
    fig.suptitle(
        f"CAD 逆向截面测绘图: {title}  |  总包围盒: {bsize[0]:.2f} × {bsize[1]:.2f} × {bsize[2]:.2f} mm",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
        y=0.97,
    )

    for ax, (panel_title, origin, normal, (xlbl, ylbl), (ix, iy)) in zip(axes.flat, specs):
        ax.set_facecolor("#ffffff")
        coord_val = origin[int(np.argmax(np.abs(normal)))]
        ax.set_title(f"{panel_title}  (@ {coord_val:.2f} mm)", fontsize=10.5, fontweight="bold", color="#1e293b")

        # Project 3D section directly onto global (ix, iy) world coordinates so rulers match exact part coords!
        sec = mesh.section(plane_origin=origin, plane_normal=normal)
        if sec is not None:
            for entity in sec.entities:
                pts3 = sec.vertices[entity.points]
                ax.plot(pts3[:, ix], pts3[:, iy], color="#0f172a", linewidth=1.6, zorder=4)

            # Also shade solid cross-section via shapely in global (ix, iy)
            try:
                planar, to_3d = sec.to_2D()
                for poly in planar.polygons_full:
                    ext3 = (to_3d @ np.column_stack([np.asarray(poly.exterior.coords), np.zeros(len(poly.exterior.coords)), np.ones(len(poly.exterior.coords))]).T).T
                    ax.fill(ext3[:, ix], ext3[:, iy], facecolor="#3b82f6", alpha=0.22, zorder=2)
                    for hole in poly.interiors:
                        from shapely.geometry import Polygon as ShapelyPolygon
                        h_area = float(ShapelyPolygon(hole).area)
                        h3 = (to_3d @ np.column_stack([np.asarray(hole.coords), np.zeros(len(hole.coords)), np.ones(len(hole.coords))]).T).T
                        ax.fill(h3[:, ix], h3[:, iy], facecolor="#ffffff", alpha=1.0, zorder=3)
                        # Mark circular hole center
                        if h_area > 0.5 and (4 * math.pi * h_area / (hole.length ** 2)) > 0.85:
                            cx, cy = float(np.mean(h3[:-1, ix])), float(np.mean(h3[:-1, iy]))
                            r_est = float(np.percentile(np.hypot(h3[:-1, ix] - cx, h3[:-1, iy] - cy), 85))
                            ax.plot(cx, cy, marker="+", color="#dc2626", markersize=7, markeredgewidth=1.2, zorder=5)
                            ax.annotate(
                                f"Ø{2*r_est:.2f}\n({cx:.1f},{cy:.1f})",
                                (cx, cy),
                                textcoords="offset points",
                                xytext=(5, 5),
                                fontsize=7.5,
                                color="#b91c1c",
                                bbox=dict(boxstyle="round,pad=0.15", fc="#fef2f2", ec="#fca5a5", alpha=0.9),
                                zorder=6,
                            )
            except Exception:
                pass
        else:
            ax.text(0.5, 0.5, "该平面无截交实体", transform=ax.transAxes, ha="center", va="center", color="#94a3b8")

        margin_x = max(5.0, bsize[ix] * 0.15)
        margin_y = max(5.0, bsize[iy] * 0.15)
        x0, x1 = min(bmin[ix] - margin_x, -2.0), max(bmax[ix] + margin_x, 2.0)
        y0, y1 = min(bmin[iy] - margin_y, -2.0), max(bmax[iy] + margin_y, 2.0)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal", adjustable="box")

        # Choose sensible major/minor grid step (10mm / 2mm or 5mm / 1mm)
        max_span = max(x1 - x0, y1 - y0)
        major_step = 5.0 if max_span <= 50 else (10.0 if max_span <= 150 else 20.0)
        minor_step = 1.0 if major_step <= 10.0 else 5.0
        ax.xaxis.set_major_locator(plt.MultipleLocator(major_step))
        ax.yaxis.set_major_locator(plt.MultipleLocator(major_step))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_step))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(minor_step))
        ax.grid(which="major", color="#cbd5e1", linestyle="-", linewidth=0.8, zorder=0)
        ax.grid(which="minor", color="#e2e8f0", linestyle=":", linewidth=0.5, zorder=0)

        # Highlight coordinate origin (0, 0)
        ax.axhline(0, color="#ef4444", linewidth=0.9, linestyle="--", alpha=0.65, zorder=1)
        ax.axvline(0, color="#22c55e", linewidth=0.9, linestyle="--", alpha=0.65, zorder=1)
        ax.set_xlabel(xlbl, fontsize=9, color="#334155")
        ax.set_ylabel(ylbl, fontsize=9, color="#334155")
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return out_png


def inspect_cad_file(path: str | Path, out_dir: str | Path | None = None) -> dict[str, Any]:
    """Run full reverse-engineering inspection on a STEP / BREP / STL / 3MF / OBJ file."""
    src = Path(path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"CAD file not found: {src}")

    suffix = src.suffix.lower()
    if suffix in (".step", ".stp", ".brep"):
        info, mesh = _inspect_step_brep(src)
    elif suffix in (".stl", ".3mf", ".obj", ".ply"):
        from .large_cad import load_mesh_cached, simplify_mesh_watertight
        file_mb = round(src.stat().st_size / (1024 * 1024), 2)
        loaded = trimesh.load(str(src), force="mesh")
        raw_mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
        raw_tris = int(len(raw_mesh.faces))
        raw_wt = bool(raw_mesh.is_watertight) if raw_tris <= 400_000 else True
        # Use simplified work-LOD mesh for fast multi-plane slicing if input has >180k triangles!
        mesh = simplify_mesh_watertight(raw_mesh, tolerance_mm=0.04, max_faces=150_000) if raw_tris > 180_000 else raw_mesh
        bmin, bmax = raw_mesh.bounds
        bsize = bmax - bmin
        bcenter = 0.5 * (bmin + bmax)
        planes, holes = _extract_mesh_planes_and_holes(mesh)
        info = {
            "file": str(src),
            "file_size_mb": file_mb,
            "format": suffix.lstrip(".").upper(),
            "representation": "Triangle Mesh" + (f" (LOD加速分析: {len(mesh.faces)}/{raw_tris} tris)" if len(mesh.faces) < raw_tris else ""),
            "triangles_count": raw_tris,
            "is_watertight": raw_wt,
            "volume_mm3": round(float(abs(mesh.volume)), 2),
            "area_mm2": round(float(mesh.area), 2),
            "bbox": {
                "min_mm": [round(float(x), 3) for x in bmin],
                "max_mm": [round(float(x), 3) for x in bmax],
                "size_mm": [round(float(x), 3) for x in bsize],
                "center_mm": [round(float(x), 3) for x in bcenter],
            },
            "major_planes": planes,
            "cylindrical_features": holes,
        }
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Expected .step, .stp, .brep, .stl, .3mf, or .obj")

    # Add origin & alignment diagnostics (crucial when modifying external STLs/STEPs!)
    bb = info["bbox"]
    c = bb["center_mm"]
    mn = bb["min_mm"]
    origin_notes: list[str] = []
    if abs(c[0]) < 0.05 and abs(c[1]) < 0.05:
        origin_notes.append("XY 已居中对齐原点 (0, 0)")
    else:
        origin_notes.append(f"注意: XY 中心偏移在 ({c[0]}, {c[1]}) mm，修改/打孔时请以实际坐标为准或先平移对齐！")
    if abs(mn[2]) < 0.05:
        origin_notes.append("底面 (-Z) 已贴合 Z = 0 热床平面")
    elif abs(c[2]) < 0.05:
        origin_notes.append("Z 轴居中于 Z = 0 (底面位于负 Z)")
    else:
        origin_notes.append(f"注意: 底面位于 Z = {mn[2]} mm (非 Z=0)")
    info["origin_diagnostics"] = origin_notes

    if out_dir is not None:
        od = Path(out_dir).expanduser().resolve()
        od.mkdir(parents=True, exist_ok=True)
        slices_png = od / f"{src.stem}_inspect_slices.png"
        render_inspection_slices(mesh, slices_png, title=src.name)
        info["slices_png"] = str(slices_png)

        json_path = od / f"{src.stem}_inspect.json"
        json_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        info["report_json"] = str(json_path)

        md_path = od / f"{src.stem}_inspect.md"
        md_path.write_text(format_inspection_markdown(info), encoding="utf-8")
        info["report_md"] = str(md_path)

    return info


def format_inspection_markdown(info: dict[str, Any]) -> str:
    bb = info["bbox"]
    lines = [
        f"# CAD 文件逆向体检报告: `{Path(info['file']).name}`",
        "",
        f"- **格式类型**: `{info['format']}` ({info['representation']})",
        f"- **总尺寸 (X × Y × Z)**: `{bb['size_mm'][0]} × {bb['size_mm'][1]} × {bb['size_mm'][2]} mm`",
        f"- **坐标范围**: X `[{bb['min_mm'][0]}, {bb['max_mm'][0]}]` | Y `[{bb['min_mm'][1]}, {bb['max_mm'][1]}]` | Z `[{bb['min_mm'][2]}, {bb['max_mm'][2]}]` mm",
        f"- **包围盒中心**: `({bb['center_mm'][0]}, {bb['center_mm'][1]}, {bb['center_mm'][2]}) mm`",
        f"- **体积 / 表面积**: `{info['volume_mm3']} mm³` / `{info['area_mm2']} mm²`",
    ]
    if "solids_count" in info:
        lines.append(f"- **实体数量 / 有效性**: `{info['solids_count']}` solids (`is_valid={info['is_valid']}`)")
    if "is_watertight" in info:
        lines.append(f"- **三角面数 / 水密性**: `{info['triangles_count']}` tris (`watertight={info['is_watertight']}`)")

    lines.extend(["", "## 原点与对齐提示"])
    for note in info.get("origin_diagnostics", []):
        lines.append(f"- {note}")

    lines.extend(["", "## 主要基准平面 (按面积排序)", "| 朝向 | 坐标位置 (mm) | 面积 (mm²) |", "| :--- | ---: | ---: |"])
    for p in info.get("major_planes", []):
        lines.append(f"| {p['orientation']} | `{p['coord_mm']}` | `{p['area_mm2']}` |")

    lines.extend(["", "## 识别到的孔位与圆柱特征", "| 类型 | 轴向 | 中心坐标 (x, y, z) [mm] | 直径 Ø (mm) | 规格推测 |", "| :--- | :---: | :--- | ---: | :--- |"])
    for c in info.get("cylindrical_features", []):
        pt = c["center_mm"]
        lines.append(
            f"| {c['type']} | `{c['axis']}` | `({pt[0]}, {pt[1]}, {pt[2]})` | **`{c['diameter_mm']}`** | {c.get('hint', '')} |"
        )
    if not info.get("cylindrical_features"):
        lines.append("| (未识别到明显通孔/圆孔) | - | - | - | - |")

    return "\n".join(lines) + "\n"
