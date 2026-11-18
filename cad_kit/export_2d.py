"""1:1 Precision 2D DXF / SVG / Engineering Flat-Pattern Exporter for Laser Cutting & CNC.

Features:
  - Preserves TRUE analytic CAD `CIRCLE` and `ARC` entities in `.dxf` & `.svg` (never faceted polylines
    for B-Rep STEP/Part inputs, and upgrades circular mesh holes to true `CIRCLE` entities for STLs!).
  - Separates `INNER_HOLES` (cut first in laser/CNC CAM!) and `OUTER_CUT` onto standard CAD layers.
  - Smart face offset selection (`bottom`, `top`, `front`, `back`, `left`, `right`) that automatically
    clears 0.4mm bed chamfers to capture true full plate dimensions & shaft hole diameters.
  - Generates an accompanying `<name>_2d_drawing.png` blueprint with overall dimensions & hole table.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import ezdxf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
import trimesh

from .inspector import _configure_cjk_font, shape_to_trimesh
from .verify import load_any_as_trimesh

ViewFace = Literal["bottom", "top", "front", "back", "left", "right"]
CutPlaneName = Literal["XY", "XZ", "YZ"]


def _resolve_cut_plane(
    bmin: np.ndarray,
    bmax: np.ndarray,
    plane: CutPlaneName = "XY",
    face: ViewFace | None = None,
    offset: float | None = None,
) -> tuple[str, float, str]:
    """Determine canonical plane ('XY', 'XZ', 'YZ'), world coordinate offset, and tag name."""
    size = bmax - bmin
    if face is not None:
        f = face.lower()
        if f == "bottom":
            # Step 0.9mm (or 22% of thickness) above min Z to clear elephant-foot chamfer while staying below top counterbores
            z = float(bmin[2] + min(0.9, size[2] * 0.25)) if offset is None else float(offset)
            return "XY", z, "bottom_xy"
        if f == "top":
            z = float(bmax[2] - min(0.9, size[2] * 0.25)) if offset is None else float(offset)
            return "XY", z, "top_xy"
        if f == "front":
            y = float(bmin[1] + min(0.9, size[1] * 0.25)) if offset is None else float(offset)
            return "XZ", y, "front_xz"
        if f == "back":
            y = float(bmax[1] - min(0.9, size[1] * 0.25)) if offset is None else float(offset)
            return "XZ", y, "back_xz"
        if f == "left":
            x = float(bmin[0] + min(0.9, size[0] * 0.25)) if offset is None else float(offset)
            return "YZ", x, "left_yz"
        if f == "right":
            x = float(bmax[0] - min(0.9, size[0] * 0.25)) if offset is None else float(offset)
            return "YZ", x, "right_yz"
        raise ValueError(f"Unknown face '{face}'")

    p = plane.upper()
    if p == "XY":
        z = float(bmin[2] + min(0.9, size[2] * 0.25)) if offset is None else float(offset)
        return "XY", z, "xy"
    if p == "XZ":
        y = float(bmax[1] - min(1.2, size[1] * 0.25)) if offset is None else float(offset)
        return "XZ", y, "xz"
    if p == "YZ":
        x = float(0.5 * (bmin[0] + bmax[0])) if offset is None else float(offset)
        return "YZ", x, "yz"
    raise ValueError(f"Unknown plane '{plane}'. Choose XY, XZ, or YZ.")


def _plane_axes_indices(plane_name: str) -> tuple[int, int, str, str]:
    if plane_name == "XY":
        return 0, 1, "X [mm]", "Y [mm]"
    if plane_name == "XZ":
        return 0, 2, "X [mm]", "Z [mm]"
    if plane_name == "YZ":
        return 1, 2, "Y [mm]", "Z [mm]"
    raise ValueError(plane_name)


def _write_svg_from_polygons(
    outer_loops: list[np.ndarray],
    inner_loops: list[np.ndarray],
    circles: list[tuple[float, float, float]],
    out_svg: Path,
) -> None:
    """Write an exact 1:1 physical millimeter SVG file from 2D contours & analytic circles."""
    all_pts = np.vstack(outer_loops) if outer_loops else np.zeros((2, 2))
    min_xy = np.min(all_pts, axis=0) - 2.0
    max_xy = np.max(all_pts, axis=0) + 2.0
    w = float(max_xy[0] - min_xy[0])
    h = float(max_xy[1] - min_xy[1])

    # Note: SVG Y grows downward, so flip Y -> -Y and viewBox [min_x, -max_y, w, h]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.3f}mm" height="{h:.3f}mm" '
        f'viewBox="{min_xy[0]:.3f} {-max_xy[1]:.3f} {w:.3f} {h:.3f}">',
        '  <g fill="none" stroke-width="0.20">',
    ]
    for loop in outer_loops:
        pts_str = " ".join(f"{pt[0]:.4f},{-pt[1]:.4f}" for pt in loop)
        lines.append(f'    <polygon points="{pts_str}" stroke="#0f172a" />')
    for loop in inner_loops:
        pts_str = " ".join(f"{pt[0]:.4f},{-pt[1]:.4f}" for pt in loop)
        lines.append(f'    <polygon points="{pts_str}" stroke="#0284c7" />')
    for cx, cy, r in circles:
        lines.append(f'    <circle cx="{cx:.4f}" cy="{-cy:.4f}" r="{r:.4f}" stroke="#0284c7" />')
    lines.extend(["  </g>", "</svg>\n"])
    out_svg.write_text("\n".join(lines), encoding="utf-8")


def export_2d_profile(
    target: Any,
    out_dir: str | Path,
    name: str = "part",
    plane: CutPlaneName = "XY",
    face: ViewFace | None = None,
    offset: float | None = None,
) -> dict[str, Any]:
    """Export a 1:1 2D DXF + SVG + Dimensioned Blueprint PNG from a Part, STEP, or STL."""
    _configure_cjk_font()
    od = Path(out_dir).expanduser().resolve()
    od.mkdir(parents=True, exist_ok=True)

    is_brep = hasattr(target, "solids") and hasattr(target, "tessellate")
    brep_part = target if is_brep else None
    if isinstance(target, (str, Path)):
        src_p = Path(target).expanduser().resolve()
        if name == "part":
            name = src_p.stem
        if src_p.suffix.lower() in (".step", ".stp", ".brep"):
            import build123d as bd
            brep_part = bd.import_brep(str(src_p)) if src_p.suffix.lower() == ".brep" else bd.import_step(str(src_p))
            is_brep = True

    mesh = shape_to_trimesh(brep_part) if brep_part is not None else load_any_as_trimesh(target)
    if len(mesh.faces) == 0:
        raise ValueError("Cannot export 2D profile: empty geometry!")

    bmin, bmax = mesh.bounds
    plane_name, coord_val, tag = _resolve_cut_plane(bmin, bmax, plane=plane, face=face, offset=offset)
    ix, iy, x_lbl, y_lbl = _plane_axes_indices(plane_name)

    dxf_path = od / f"{name}_{tag}.dxf"
    svg_path = od / f"{name}_{tag}.svg"
    png_path = od / f"{name}_{tag}_2d.png"

    hole_records: list[dict[str, float | str]] = []
    outer_loops_2d: list[np.ndarray] = []
    inner_poly_loops_2d: list[np.ndarray] = []
    analytic_circles_2d: list[tuple[float, float, float]] = []

    # Extract 2D Shapely polygons in TRUE global (coord[ix], coord[iy]) coordinates for drawing & mesh fallback
    normal_3d = np.zeros(3)
    origin_3d = np.zeros(3)
    cut_axis = {"XY": 2, "XZ": 1, "YZ": 0}[plane_name]
    normal_3d[cut_axis] = 1.0
    origin_3d[cut_axis] = coord_val

    sec_mesh = mesh.section(plane_origin=origin_3d, plane_normal=normal_3d)
    if sec_mesh is None:
        raise ValueError(f"Plane {plane_name} @ {coord_val:.2f} mm does not intersect the part!")

    planar_m, to_3d = sec_mesh.to_2D()
    for poly in planar_m.polygons_full:
        ext_pts3 = (to_3d @ np.column_stack([np.asarray(poly.exterior.coords), np.zeros(len(poly.exterior.coords)), np.ones(len(poly.exterior.coords))]).T).T
        outer_loops_2d.append(ext_pts3[:, [ix, iy]])

        for hole_ring in poly.interiors:
            h_area = float(ShapelyPolygon(hole_ring).area)
            h_len = float(hole_ring.length)
            h_pts3 = (to_3d @ np.column_stack([np.asarray(hole_ring.coords), np.zeros(len(hole_ring.coords)), np.ones(len(hole_ring.coords))]).T).T
            pts2 = h_pts3[:-1, [ix, iy]]
            if h_area > 0.4 and h_len > 0.8 and (4.0 * math.pi * h_area / (h_len * h_len)) > 0.96:
                cx, cy = float(np.mean(pts2[:, 0])), float(np.mean(pts2[:, 1]))
                radii = np.hypot(pts2[:, 0] - cx, pts2[:, 1] - cy)
                if (float(np.max(radii)) - float(np.min(radii))) <= 0.08 * float(np.mean(radii)):
                    r_est = float(np.percentile(radii, 85))
                    analytic_circles_2d.append((round(cx, 4), round(cy, 4), round(r_est, 4)))
                else:
                    inner_poly_loops_2d.append(h_pts3[:, [ix, iy]])
            else:
                inner_poly_loops_2d.append(h_pts3[:, [ix, iy]])

    # 1. Export DXF: Use native build123d analytic curves when B-Rep is available!
    wrote_brep_dxf = False
    if brep_part is not None:
        try:
            import build123d as bd
            from OCP.BRepAdaptor import BRepAdaptor_Curve

            if plane_name == "XY":
                bd_plane = bd.Plane(origin=(0, 0, coord_val), x_dir=(1, 0, 0), z_dir=(0, 0, 1))
            elif plane_name == "XZ":
                bd_plane = bd.Plane(origin=(0, coord_val, 0), x_dir=(1, 0, 0), z_dir=(0, -1, 0))
            else:
                bd_plane = bd.Plane(origin=(coord_val, 0, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0))

            sec_sketch = bd.section(brep_part, bd_plane)
            faces_2d = sec_sketch.faces()
            if len(faces_2d) > 0:
                # Map each 2D face onto Plane.XY so ExportDXF writes exact global 2D coordinates!
                inv_loc = bd_plane.location.inverse()
                exp_dxf = bd.ExportDXF(unit=bd.Unit.MM)
                exp_dxf.add_layer("OUTER_CUT", color=bd.ColorIndex.BLACK)
                exp_dxf.add_layer("INNER_HOLES", color=bd.ColorIndex.CYAN)

                exact_circles: list[tuple[float, float, float]] = []
                for f2 in faces_2d:
                    f_flat = f2.moved(inv_loc)
                    exp_dxf.add_shape(f_flat.outer_wire(), layer="OUTER_CUT")
                    for iw in f_flat.inner_wires():
                        exp_dxf.add_shape(iw, layer="INNER_HOLES")
                        edges_iw = iw.edges()
                        if len(edges_iw) == 1 and edges_iw[0].geom_type == bd.GeomType.CIRCLE:
                            c_adapt = BRepAdaptor_Curve(edges_iw[0].wrapped).Circle()
                            loc_c = c_adapt.Location()
                            exact_circles.append((round(loc_c.X(), 3), round(loc_c.Y(), 3), round(c_adapt.Radius(), 3)))

                exp_dxf.write(str(dxf_path))
                wrote_brep_dxf = True
                if exact_circles:
                    analytic_circles_2d = exact_circles

                # Also export native analytic SVG via ExportSVG
                exp_svg = bd.ExportSVG(unit=bd.Unit.MM)
                exp_svg.add_layer("OUTER_CUT", line_color=(15, 23, 42), line_weight=0.20)
                exp_svg.add_layer("INNER_HOLES", line_color=(2, 132, 199), line_weight=0.18)
                for f2 in faces_2d:
                    f_flat = f2.moved(inv_loc)
                    exp_svg.add_shape(f_flat.outer_wire(), layer="OUTER_CUT")
                    for iw in f_flat.inner_wires():
                        exp_svg.add_shape(iw, layer="INNER_HOLES")
                exp_svg.write(str(svg_path))
        except Exception:
            wrote_brep_dxf = False

    if not wrote_brep_dxf:
        # Mesh / Shapely DXF writer with true CIRCLE entities for circular holes!
        doc = ezdxf.new("R2010", setup=True)
        doc.units = ezdxf.units.MM
        doc.layers.add("OUTER_CUT", color=7)
        doc.layers.add("INNER_HOLES", color=4)
        msp = doc.modelspace()
        for loop in outer_loops_2d:
            msp.add_lwpolyline([(float(p[0]), float(p[1])) for p in loop], close=True, dxfattribs={"layer": "OUTER_CUT"})
        for loop in inner_poly_loops_2d:
            msp.add_lwpolyline([(float(p[0]), float(p[1])) for p in loop], close=True, dxfattribs={"layer": "INNER_HOLES"})
        for cx, cy, r in analytic_circles_2d:
            msp.add_circle((cx, cy), radius=r, dxfattribs={"layer": "INNER_HOLES"})
        doc.saveas(str(dxf_path))
        _write_svg_from_polygons(outer_loops_2d, inner_poly_loops_2d, analytic_circles_2d, svg_path)

    # Sort circular holes cleanly (top-to-bottom, left-to-right) and build Hole Drilling Table
    analytic_circles_2d.sort(key=lambda c: (round(c[1], 1), round(c[0], 1)))
    for idx_h, (cx, cy, r) in enumerate(analytic_circles_2d, start=1):
        hole_records.append(
            {
                "id": f"C{idx_h}",
                "x_mm": round(cx, 2),
                "y_mm": round(cy, 2),
                "diameter_mm": round(2.0 * r, 2),
            }
        )

    # 2. Render 2D Engineering Flat-Pattern Blueprint (`*_2d.png`)
    all_xy = np.vstack(outer_loops_2d)
    xy_min = np.min(all_xy, axis=0)
    xy_max = np.max(all_xy, axis=0)
    xy_span = xy_max - xy_min

    fig, (ax_draw, ax_tbl) = plt.subplots(
        1, 2, figsize=(14.5, 7.8), dpi=160, gridspec_kw={"width_ratios": [2.2, 1.0]}
    )
    fig.patch.set_facecolor("#f8fafc")
    fig.suptitle(
        f"1:1 二维加工展开/截面工程图 (DXF/SVG 配套核对图): {name}  |  截面: {plane_name} @ {coord_val:.2f} mm",
        fontsize=12.5,
        fontweight="bold",
        color="#0f172a",
        y=0.97,
    )

    ax_draw.set_facecolor("#ffffff")
    for loop in outer_loops_2d:
        ax_draw.fill(loop[:, 0], loop[:, 1], facecolor="#e0f2fe", alpha=0.55, zorder=2)
        ax_draw.plot(loop[:, 0], loop[:, 1], color="#0f172a", linewidth=1.8, zorder=4)
    for loop in inner_poly_loops_2d:
        ax_draw.fill(loop[:, 0], loop[:, 1], facecolor="#ffffff", alpha=1.0, zorder=3)
        ax_draw.plot(loop[:, 0], loop[:, 1], color="#0284c7", linewidth=1.4, zorder=4)

    for rec in hole_records:
        cx, cy, d = float(rec["x_mm"]), float(rec["y_mm"]), float(rec["diameter_mm"])
        circ_patch = MplCircle((cx, cy), d * 0.5, facecolor="#ffffff", edgecolor="#0284c7", linewidth=1.5, zorder=4)
        ax_draw.add_patch(circ_patch)
        ax_draw.plot(cx, cy, marker="+", color="#dc2626", markersize=6, zorder=5)
        ax_draw.annotate(
            str(rec["id"]),
            (cx, cy),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            fontweight="bold",
            color="#991b1b",
            bbox=dict(boxstyle="round,pad=0.14", fc="#fef2f2", ec="#f87171", lw=0.7),
            zorder=6,
        )

    pad_x = max(6.0, float(xy_span[0]) * 0.16)
    pad_y = max(6.0, float(xy_span[1]) * 0.18)
    ax_draw.set_xlim(float(xy_min[0] - pad_x), float(xy_max[0] + pad_x))
    ax_draw.set_ylim(float(xy_min[1] - pad_y), float(xy_max[1] + pad_y))
    ax_draw.set_aspect("equal", adjustable="box")

    max_s = float(max(xy_span[0], xy_span[1]))
    maj = 5.0 if max_s <= 45 else (10.0 if max_s <= 150 else 20.0)
    ax_draw.xaxis.set_major_locator(plt.MultipleLocator(maj))
    ax_draw.yaxis.set_major_locator(plt.MultipleLocator(maj))
    ax_draw.xaxis.set_minor_locator(plt.MultipleLocator(1.0 if maj <= 10.0 else 5.0))
    ax_draw.yaxis.set_minor_locator(plt.MultipleLocator(1.0 if maj <= 10.0 else 5.0))
    ax_draw.grid(which="major", color="#cbd5e1", linewidth=0.8, zorder=0)
    ax_draw.grid(which="minor", color="#e2e8f0", linestyle=":", linewidth=0.5, zorder=0)
    ax_draw.axhline(0, color="#ef4444", linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
    ax_draw.axvline(0, color="#10b981", linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)

    # Dimension callouts
    dim_y = float(xy_min[1] - pad_y * 0.48)
    ax_draw.annotate("", xy=(xy_min[0], dim_y), xytext=(xy_max[0], dim_y), arrowprops=dict(arrowstyle="<->", color="#1d4ed8", lw=1.1))
    ax_draw.text(0.5 * (xy_min[0] + xy_max[0]), dim_y - pad_y * 0.08, f"{xy_span[0]:.2f} mm", ha="center", va="top", fontsize=8.5, fontweight="bold", color="#1e40af")

    dim_x = float(xy_max[0] + pad_x * 0.42)
    ax_draw.annotate("", xy=(dim_x, xy_min[1]), xytext=(dim_x, xy_max[1]), arrowprops=dict(arrowstyle="<->", color="#1d4ed8", lw=1.1))
    ax_draw.text(dim_x + pad_x * 0.08, 0.5 * (xy_min[1] + xy_max[1]), f"{xy_span[1]:.2f} mm", ha="left", va="center", rotation=90, fontsize=8.5, fontweight="bold", color="#1e40af")

    ax_draw.set_xlabel(x_lbl, fontsize=9)
    ax_draw.set_ylabel(y_lbl, fontsize=9)
    ax_draw.set_title(f"外轮廓尺寸: {xy_span[0]:.2f} × {xy_span[1]:.2f} mm  (黑线=OUTER_CUT, 蓝线=INNER_HOLES)", fontsize=10, fontweight="bold")

    # Right panel: Hole Coordinate Drilling Table & DXF Layer Summary
    ax_tbl.axis("off")
    tbl_lines = [
        f"【DXF / SVG 加工文件摘要】",
        f"• DXF 文件 : {dxf_path.name}",
        f"• SVG 文件 : {svg_path.name}",
        f"• 实体类型 : {'原生解析圆弧 CIRCLE/ARC' if wrote_brep_dxf else '网格截面圆拟合 CIRCLE'}",
        f"• 外轮廓框 : {xy_span[0]:.2f} × {xy_span[1]:.2f} mm",
        f"• 坐标极值 : [{xy_min[0]:.2f}..{xy_max[0]:.2f}, {xy_min[1]:.2f}..{xy_max[1]:.2f}]",
        "",
        "【圆孔坐标钻孔表 (孔心距原点 mm)】",
        f"{'编号':<5} {'横坐标':>8} {'纵坐标':>8} {'孔径 Ø':>8}",
        "-" * 36,
    ]
    if hole_records:
        for r in hole_records[:20]:
            tbl_lines.append(f"{str(r['id']):<5} {r['x_mm']:>8.2f} {r['y_mm']:>8.2f} {r['diameter_mm']:>8.2f}")
    else:
        tbl_lines.append("(当前截面无闭合圆孔)")

    ax_tbl.text(
        0.02,
        0.96,
        "\n".join(tbl_lines),
        transform=ax_tbl.transAxes,
        va="top",
        ha="left",
        fontsize=9.2,
        color="#1e293b",
        bbox=dict(boxstyle="round,pad=0.6", fc="#ffffff", ec="#cbd5e1", lw=1.0),
    )

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.93])
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    return {
        "plane": plane_name,
        "offset_mm": round(coord_val, 3),
        "outer_size_mm": [round(float(xy_span[0]), 3), round(float(xy_span[1]), 3)],
        "analytic_brep": wrote_brep_dxf,
        "holes": hole_records,
        "dxf": str(dxf_path),
        "svg": str(svg_path),
        "drawing_png": str(png_path),
    }
