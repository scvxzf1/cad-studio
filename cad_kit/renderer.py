"""High-speed 10-View CAD Engineering Contact Sheet & Volumetric Boolean Diff Renderer.

Generates:
  1. `preview_sheet.png`:
     - 6 True-Parallel Orthographic Views (Front -Y, Back +Y, Left -X, Right +X, Top +Z, Bottom -Z)
       with sub-pixel calibrated 1mm/5mm/10mm CAD grids, true global coordinate rulers, (0,0) origin
       crosshairs, overall dimension callouts, and projected Feature ID badges (H1, S1, B1...).
     - 2 Isometric Shaded + Crisp-Edge Views (Top-Front-Right & Bottom-Back-Left) with XYZ axis gizmo.
     - 1 X-Ray Semi-Transparent Internal Cavity View (highlights blind holes, steps, nut pockets).
     - 1 Solid-Capped Mid-Plane Section Cut View (reveals wall thickness & counterbore cross-sections).
  2. `diff_sheet.png`:
     - Exact 3D boolean volumetric diff via manifold3d:
       * Unchanged (A ∩ B) : Translucent cool slate + wireframe
       * Added     (B - A) : Solid emerald green (#10b981)
       * Removed   (A - B) : Translucent coral red (#ef4444) + red edges
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pyvista as pv
import trimesh

from .inspector import _configure_cjk_font
from .verify import load_any_as_trimesh, manifold_to_trimesh, trimesh_to_manifold

pv.OFF_SCREEN = True

# Persistent off-screen OpenGL Plotter pool keyed by window_size -> eliminates 14x context creation per build!
_SHARED_PLOTTERS: dict[tuple[int, int], pv.Plotter] = {}


def _get_shared_plotter(win_size: tuple[int, int]) -> pv.Plotter:
    key = (int(win_size[0]), int(win_size[1]))
    pl = _SHARED_PLOTTERS.get(key)
    if pl is None or getattr(pl, "_closed", False):
        pl = pv.Plotter(off_screen=True, window_size=key)
        pl.enable_lightkit()
        _SHARED_PLOTTERS[key] = pl
    else:
        pl.clear_actors()
        try:
            pl.hide_axes()
        except Exception:
            pass
    return pl


def trimesh_to_pv(mesh: trimesh.Trimesh) -> pv.PolyData:
    """Convert trimesh.Trimesh into PyVista PolyData with clean normals for sharp CAD shading."""
    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        return pv.PolyData()
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces_arr = np.asarray(mesh.faces, dtype=np.int64)
    padded = np.column_stack([np.full(len(faces_arr), 3, dtype=np.int64), faces_arr]).ravel()
    poly = pv.PolyData(verts, padded)
    poly = poly.compute_normals(cell_normals=True, point_normals=True, split_vertices=True, feature_angle=28.0)
    return poly


def extract_cad_edges(poly: pv.PolyData, angle: float = 24.0) -> pv.PolyData:
    """Extract sharp CAD crease & boundary lines from PolyData."""
    if poly.n_points == 0:
        return pv.PolyData()
    return poly.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=False,
        feature_edges=True,
        manifold_edges=False,
        feature_angle=angle,
    )


@dataclass(frozen=True)
class OrthoSpec:
    title: str
    cam_dir: tuple[float, float, float]   # Unit/direction vector from focal point TO camera
    up_dir: tuple[float, float, float]    # Camera world up vector
    h_axis_idx: int                       # Global axis index (0=X, 1=Y, 2=Z) along horizontal
    h_sign: int                           # +1 if screen-right is +axis, -1 if screen-right is -axis
    v_axis_idx: int                       # Global axis index along vertical
    v_sign: int                           # +1 if screen-up is +axis, -1 if screen-up is -axis
    h_label: str
    v_label: str


ORTHO_VIEWS: list[OrthoSpec] = [
    OrthoSpec("① 正面 Front (-Y 视向)", (0, -1, 0), (0, 0, 1), 0, +1, 2, +1, "X: 左(-X) → 右(+X) [mm]", "Z: 下(-Z) → 上(+Z) [mm]"),
    OrthoSpec("② 背面 Back (+Y 视向)",  (0, +1, 0), (0, 0, 1), 0, -1, 2, +1, "X: 右(+X) ← 左(-X) [mm]", "Z: 下(-Z) → 上(+Z) [mm]"),
    OrthoSpec("③ 顶视图 Top (+Z 俯视)",  (0, 0, +1), (0, 1, 0), 0, +1, 1, +1, "X: 左(-X) → 右(+X) [mm]", "Y: 前(-Y) → 后(+Y) [mm]"),
    OrthoSpec("④ 底视图 Bottom (-Z 仰视)", (0, 0, -1), (0, -1, 0), 0, +1, 1, -1, "X: 左(-X) → 右(+X) [mm]", "Y: 后(+Y) ← 前(-Y) [mm]"),
    OrthoSpec("⑤ 左视图 Left (-X 视向)", (-1, 0, 0), (0, 0, 1), 1, -1, 2, +1, "Y: 后(+Y) ← 前(-Y) [mm]", "Z: 下(-Z) → 上(+Z) [mm]"),
    OrthoSpec("⑥ 右视图 Right (+X 视向)", (+1, 0, 0), (0, 0, 1), 1, +1, 2, +1, "Y: 前(-Y) → 后(+Y) [mm]", "Z: 下(-Z) → 上(+Z) [mm]"),
]


def _render_pv_ortho_rgba(
    poly: pv.PolyData,
    edges: pv.PolyData,
    focal: np.ndarray,
    cam_dir: Sequence[float],
    up_dir: Sequence[float],
    parallel_scale: float,
    win_size: tuple[int, int] = (640, 520),
    ref_poly: pv.PolyData | None = None,
) -> np.ndarray:
    """Render an orthographic parallel-projected RGBA image with transparent background."""
    pl = _get_shared_plotter(win_size)
    pl.set_background("#ffffff")

    if ref_poly is not None and ref_poly.n_points > 0:
        pl.add_mesh(ref_poly, color="#94a3b8", opacity=0.22, smooth_shading=True)
        ref_edges = extract_cad_edges(ref_poly, 28.0)
        if ref_edges.n_points > 0:
            pl.add_mesh(ref_edges, color="#64748b", line_width=1.0, opacity=0.35)

    if poly.n_points > 0:
        pl.add_mesh(
            poly,
            color="#60a5fa",
            specular=0.28,
            specular_power=18,
            ambient=0.32,
            diffuse=0.72,
            smooth_shading=True,
        )
    if edges.n_points > 0:
        pl.add_mesh(edges, color="#0f172a", line_width=1.8)

    cdir = np.asarray(cam_dir, dtype=float)
    cdir = cdir / (np.linalg.norm(cdir) + 1e-12)
    dist = max(500.0, parallel_scale * 12.0)

    pl.camera.enable_parallel_projection()
    pl.camera.focal_point = tuple(float(x) for x in focal)
    pl.camera.position = tuple(float(x) for x in (focal + dist * cdir))
    pl.camera.up = tuple(float(x) for x in up_dir)
    pl.camera.parallel_scale = float(parallel_scale)
    pl.camera.clipping_range = (0.1, dist * 3.0)

    rgba = pl.screenshot(transparent_background=True, return_img=True)
    return rgba


def _render_pv_special(
    mesh: trimesh.Trimesh,
    poly: pv.PolyData,
    edges: pv.PolyData,
    mode: str,
    focal: np.ndarray,
    diag: float,
    win_size: tuple[int, int] = (640, 520),
    ref_poly: pv.PolyData | None = None,
) -> np.ndarray:
    """Render Isometric, X-Ray, or Capped Section 3D views."""
    pl = _get_shared_plotter(win_size)
    pl.set_background("#f8fafc")

    if ref_poly is not None and ref_poly.n_points > 0 and mode in ("iso_top", "section"):
        pl.add_mesh(ref_poly, color="#cbd5e1", opacity=0.28, smooth_shading=True)
        re = extract_cad_edges(ref_poly, 28.0)
        if re.n_points > 0:
            pl.add_mesh(re, color="#64748b", line_width=1.0, opacity=0.45)

    if mode == "iso_top":
        if poly.n_points > 0:
            pl.add_mesh(poly, color="#3b82f6", specular=0.35, ambient=0.30, smooth_shading=True)
        if edges.n_points > 0:
            pl.add_mesh(edges, color="#0f172a", line_width=1.8)
        cdir = np.array([1.05, -1.25, 0.92])
        up = (0, 0, 1)
    elif mode == "iso_bottom":
        if poly.n_points > 0:
            pl.add_mesh(poly, color="#38bdf8", specular=0.35, ambient=0.30, smooth_shading=True)
        if edges.n_points > 0:
            pl.add_mesh(edges, color="#0f172a", line_width=1.8)
        cdir = np.array([-1.05, 1.20, -0.88])
        up = (0, 0, 1)
    elif mode == "xray":
        if poly.n_points > 0:
            pl.add_mesh(poly, color="#60a5fa", opacity=0.27, smooth_shading=True)
        if edges.n_points > 0:
            pl.add_mesh(edges, color="#1e293b", line_width=2.0, opacity=0.92)
        cdir = np.array([-1.0, -1.25, 0.85])
        up = (0, 0, 1)
    elif mode == "section":
        # Perform watertight solid boolean half-space cut along Y = focal[1] using manifold3d!
        # Highlighting the cut cross-section face makes wall thickness & counterbores pop!
        try:
            bmin, bmax = mesh.bounds
            span = np.maximum(bmax - bmin, 10.0) * 3.0
            # Keep the +Y back half, slice open from -Y front so looking from -Y shows interior!
            cut_box = trimesh.creation.box(
                bounds=[
                    [focal[0] - span[0], bmin[1] - span[1], focal[2] - span[2]],
                    [focal[0] + span[0], focal[1], focal[2] + span[2]],
                ]
            )
            m_orig = trimesh_to_manifold(mesh)
            m_cut = trimesh_to_manifold(cut_box)
            sliced_mesh = manifold_to_trimesh(m_orig - m_cut)
            s_poly = trimesh_to_pv(sliced_mesh)
            s_edges = extract_cad_edges(s_poly, 24.0)
            if s_poly.n_points > 0:
                pl.add_mesh(s_poly, color="#3b82f6", specular=0.3, ambient=0.32, smooth_shading=True)
            # Overlay coral-orange cut cap at Y == focal[1]
            sec = mesh.section(plane_origin=[0, float(focal[1]), 0], plane_normal=[0, 1, 0])
            if sec is not None:
                planar, to_3d = sec.to_2D()
                for p2d in planar.polygons_full:
                    v2, f2 = trimesh.creation.triangulate_polygon(p2d)
                    v3 = (to_3d @ np.column_stack([v2, np.full(len(v2), -0.02), np.ones(len(v2))]).T).T[:, :3]
                    cap_pv = trimesh_to_pv(trimesh.Trimesh(vertices=v3, faces=f2))
                    if cap_pv.n_points > 0:
                        pl.add_mesh(cap_pv, color="#f97316", ambient=0.55, diffuse=0.6)
            if s_edges.n_points > 0:
                pl.add_mesh(s_edges, color="#0f172a", line_width=1.8)
        except Exception:
            clipped = poly.clip(normal=(0, -1, 0), origin=tuple(float(x) for x in focal))
            if clipped.n_points > 0:
                pl.add_mesh(clipped, color="#3b82f6", smooth_shading=True)
        cdir = np.array([0.65, -1.35, 0.65])
        up = (0, 0, 1)
    else:
        raise ValueError(mode)

    cdir = cdir / np.linalg.norm(cdir)
    dist = max(250.0, diag * 2.6)
    pl.camera.enable_parallel_projection()
    pl.camera.focal_point = tuple(float(x) for x in focal)
    pl.camera.position = tuple(float(x) for x in (focal + dist * cdir))
    pl.camera.up = up
    pl.camera.parallel_scale = max(diag * 0.56, 2.0)
    pl.add_axes(line_width=2, color="#0f172a", xlabel="+X", ylabel="+Y", zlabel="+Z")

    img = pl.screenshot(transparent_background=False, return_img=True)
    return img


def render_preview_sheet(
    target: Any,
    out_png: str | Path,
    title: str = "Part",
    mode: str = "fdm",
    anchors: list[dict[str, Any]] | None = None,
    ref_files: list[str | Path] | None = None,
) -> Path:
    """Generate the 10-panel Engineering Contact Sheet (`preview_sheet.png`)."""
    _configure_cjk_font()
    mesh = load_any_as_trimesh(target)
    if len(mesh.faces) == 0:
        raise ValueError("Cannot render preview sheet: mesh is empty!")

    poly = trimesh_to_pv(mesh)
    edges = extract_cad_edges(poly, angle=24.0)

    ref_poly: pv.PolyData | None = None
    if ref_files:
        ref_meshes = [load_any_as_trimesh(rf) for rf in ref_files if Path(rf).exists()]
        if ref_meshes:
            combined_ref = trimesh.util.concatenate(ref_meshes)
            ref_poly = trimesh_to_pv(combined_ref)

    bmin, bmax = mesh.bounds
    bsize = bmax - bmin
    focal = 0.5 * (bmin + bmax)
    diag = float(np.linalg.norm(bsize))

    win_w, win_h = 640, 500
    aspect = win_w / win_h

    # Layout: 2 rows x 5 columns (6 Orthographic + 4 3D/Diagnostic panels = 10 views!)
    fig, axes = plt.subplots(2, 5, figsize=(23, 10.2), dpi=150)
    fig.patch.set_facecolor("#f1f5f9")

    header_str = (
        f"工程十视图总览: {title}   |   工艺档位: [{mode.upper()}]   |   "
        f"总尺寸 X×Y×Z: {bsize[0]:.2f} × {bsize[1]:.2f} × {bsize[2]:.2f} mm   |   "
        f"坐标范围: X[{bmin[0]:.1f},{bmax[0]:.1f}] Y[{bmin[1]:.1f},{bmax[1]:.1f}] Z[{bmin[2]:.1f},{bmax[2]:.1f}]   |   "
        f"方位基准: 前=-Y  后=+Y  左=-X  右=+X  上=+Z  下=-Z"
    )
    fig.suptitle(header_str, fontsize=12.5, fontweight="bold", color="#0f172a", y=0.975)

    # Order panels intuitively:
    # Top row:    [1 Front -Y] [2 Back +Y]   [3 Top +Z]    [7 Iso Top]    [9 X-Ray]
    # Bottom row: [5 Left -X]  [6 Right +X]  [4 Bottom -Z] [8 Iso Bottom] [10 Section Cut]
    ortho_placement = [
        (axes[0, 0], ORTHO_VIEWS[0]),  # Front
        (axes[0, 1], ORTHO_VIEWS[1]),  # Back
        (axes[0, 2], ORTHO_VIEWS[2]),  # Top
        (axes[1, 0], ORTHO_VIEWS[4]),  # Left
        (axes[1, 1], ORTHO_VIEWS[5]),  # Right
        (axes[1, 2], ORTHO_VIEWS[3]),  # Bottom
    ]

    anchor_list = anchors or []

    # Upload scene geometry ONCE to the ortho Plotter and capture all 6 orthographic cameras!
    pl_ortho = _get_shared_plotter((win_w, win_h))
    pl_ortho.set_background("#ffffff")
    if ref_poly is not None and ref_poly.n_points > 0:
        pl_ortho.add_mesh(ref_poly, color="#94a3b8", opacity=0.22, smooth_shading=True)
        ref_edges = extract_cad_edges(ref_poly, 28.0)
        if ref_edges.n_points > 0:
            pl_ortho.add_mesh(ref_edges, color="#64748b", line_width=1.0, opacity=0.35)
    if poly.n_points > 0:
        pl_ortho.add_mesh(
            poly,
            color="#60a5fa",
            specular=0.28,
            specular_power=18,
            ambient=0.32,
            diffuse=0.72,
            smooth_shading=True,
        )
    if edges.n_points > 0:
        pl_ortho.add_mesh(edges, color="#0f172a", line_width=1.8)
    pl_ortho.camera.enable_parallel_projection()
    pl_ortho.camera.focal_point = tuple(float(x) for x in focal)

    for ax, spec in ortho_placement:
        ax.set_facecolor("#ffffff")
        span_h = float(bsize[spec.h_axis_idx])
        span_v = float(bsize[spec.v_axis_idx])
        half_v_needed = max(span_v * 0.68, (span_h / aspect) * 0.68, 4.0)

        cdir = np.asarray(spec.cam_dir, dtype=float)
        cdir = cdir / (np.linalg.norm(cdir) + 1e-12)
        dist = max(500.0, half_v_needed * 12.0)
        pl_ortho.camera.position = tuple(float(x) for x in (focal + dist * cdir))
        pl_ortho.camera.up = tuple(float(x) for x in spec.up_dir)
        pl_ortho.camera.parallel_scale = float(half_v_needed)
        pl_ortho.camera.clipping_range = (0.1, dist * 3.0)
        rgba = pl_ortho.screenshot(transparent_background=True, return_img=True)

        # Screen coordinates u = h_sign * coord[h_axis], v = v_sign * coord[v_axis]
        fu = spec.h_sign * float(focal[spec.h_axis_idx])
        fv = spec.v_sign * float(focal[spec.v_axis_idx])
        u0, u1 = fu - half_v_needed * aspect, fu + half_v_needed * aspect
        v0, v1 = fv - half_v_needed, fv + half_v_needed

        # Configure grid BEFORE overlay image (`zorder=3`) so gridlines stay behind the solid part!
        max_span = 2.0 * half_v_needed * aspect
        major_step = 5.0 if max_span <= 45 else (10.0 if max_span <= 140 else (20.0 if max_span <= 300 else 50.0))
        minor_step = 1.0 if major_step <= 10.0 else 5.0
        ax.xaxis.set_major_locator(MultipleLocator(major_step))
        ax.yaxis.set_major_locator(MultipleLocator(major_step))
        ax.xaxis.set_minor_locator(MultipleLocator(minor_step))
        ax.yaxis.set_minor_locator(MultipleLocator(minor_step))
        # Format tick labels so negative-directed screen axes still read TRUE global coordinates!
        if spec.h_sign < 0:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{-val:g}"))
        if spec.v_sign < 0:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{-val:g}"))

        ax.grid(which="major", color="#cbd5e1", linestyle="-", linewidth=0.75, zorder=1)
        ax.grid(which="minor", color="#e2e8f0", linestyle=":", linewidth=0.45, zorder=1)
        ax.axhline(0.0, color="#ef4444", linewidth=0.85, linestyle="--", alpha=0.55, zorder=2)
        ax.axvline(0.0, color="#10b981", linewidth=0.85, linestyle="--", alpha=0.55, zorder=2)

        ax.imshow(rgba, extent=[u0, u1, v0, v1], origin="upper", zorder=3)
        ax.set_xlim(u0, u1)
        ax.set_ylim(v0, v1)
        ax.set_aspect("equal", adjustable="box")

        # Overall horizontal & vertical dimension arrows
        pu_min = min(spec.h_sign * float(bmin[spec.h_axis_idx]), spec.h_sign * float(bmax[spec.h_axis_idx]))
        pu_max = max(spec.h_sign * float(bmin[spec.h_axis_idx]), spec.h_sign * float(bmax[spec.h_axis_idx]))
        pv_min = min(spec.v_sign * float(bmin[spec.v_axis_idx]), spec.v_sign * float(bmax[spec.v_axis_idx]))
        pv_max = max(spec.v_sign * float(bmin[spec.v_axis_idx]), spec.v_sign * float(bmax[spec.v_axis_idx]))

        dim_y = pv_min - 0.10 * half_v_needed
        if dim_y > v0 + 0.04 * half_v_needed and span_h > 0.2:
            ax.annotate(
                "",
                xy=(pu_min, dim_y),
                xytext=(pu_max, dim_y),
                arrowprops=dict(arrowstyle="<->", color="#1d4ed8", lw=1.0),
                zorder=5,
            )
            ax.text(
                0.5 * (pu_min + pu_max),
                dim_y - 0.02 * half_v_needed,
                f"{span_h:.2f} mm",
                ha="center",
                va="top",
                fontsize=7.5,
                fontweight="bold",
                color="#1e40af",
                bbox=dict(boxstyle="round,pad=0.12", fc="#eff6ff", ec="none", alpha=0.88),
                zorder=6,
            )

        # Project Feature ID badges onto primary views (Front, Top, Right) or matching normal views
        show_anchors = spec.title.startswith(("①", "③", "⑥"))
        for idx_a, anc in enumerate(anchor_list[:12]):
            nrm = anc.get("normal")
            if nrm is not None:
                # Show on view if feature normal faces camera or is orthogonal, skip opposing back-side clutter
                dot_cam = float(np.dot(nrm, spec.cam_dir))
                if dot_cam < -0.5:
                    continue
            elif not show_anchors:
                continue

            pos = anc["pos"]
            au = spec.h_sign * float(pos[spec.h_axis_idx])
            av = spec.v_sign * float(pos[spec.v_axis_idx])
            # Stagger callout offsets around clock positions so nearby holes don't overlap
            angle = math.radians(35 + (idx_a * 55) % 300)
            dx_pts = 18.0 * math.cos(angle)
            dy_pts = 16.0 * math.sin(angle)
            ax.plot(au, av, marker="o", markersize=3.5, color="#dc2626", zorder=7)
            ax.annotate(
                anc["id"],
                (au, av),
                textcoords="offset points",
                xytext=(dx_pts, dy_pts),
                ha="center",
                va="center",
                fontsize=7.2,
                fontweight="bold",
                color="#991b1b",
                bbox=dict(boxstyle="round,pad=0.16", fc="#fef2f2", ec="#f87171", lw=0.8, alpha=0.92),
                arrowprops=dict(arrowstyle="-", color="#ef4444", lw=0.75),
                zorder=8,
            )

        ax.set_title(spec.title, fontsize=9.8, fontweight="bold", color="#0f172a", pad=5)
        ax.set_xlabel(spec.h_label, fontsize=8, color="#334155")
        ax.set_ylabel(spec.v_label, fontsize=8, color="#334155")
        ax.tick_params(labelsize=7.2)

    # Render the 4 3D diagnostic views on columns 3 & 4
    special_specs = [
        (axes[0, 3], "iso_top", "⑦ 等轴测俯视 Iso (+X -Y +Z)"),
        (axes[1, 3], "iso_bottom", "⑧ 等轴测仰视 Iso (-X +Y -Z 防底面盲区)"),
        (axes[0, 4], "xray", "⑨ X-Ray 半透明内腔透视图"),
        (axes[1, 4], "section", f"⑩ 中心剖视图 Section (Y = {focal[1]:.1f} mm 橙色切面)"),
    ]
    for ax, smode, stitle in special_specs:
        s_img = _render_pv_special(
            mesh=mesh,
            poly=poly,
            edges=edges,
            mode=smode,
            focal=focal,
            diag=diag,
            win_size=(win_w, win_h),
            ref_poly=ref_poly,
        )
        ax.imshow(s_img)
        ax.set_title(stitle, fontsize=9.8, fontweight="bold", color="#0f172a", pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#cbd5e1")

    # Add Feature Legend bar along bottom if anchors exist (wrapped max 4 items per row)
    if anchor_list:
        legend_items = [
            f"[{a['id']}] {a['desc']}" + (f" ({a['spec']})" if a.get("spec") else "") + f" @({a['pos'][0]:.1f},{a['pos'][1]:.1f},{a['pos'][2]:.1f})"
            for a in anchor_list[:12]
        ]
        rows = ["    |    ".join(legend_items[i : i + 4]) for i in range(0, len(legend_items), 4)]
        legend_text = "特征锚点索引 (可直接点名修改):   " + "\n".join(rows)
        fig.text(
            0.5,
            0.008,
            legend_text,
            ha="center",
            va="bottom",
            fontsize=8.3,
            color="#1e293b",
            bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec="#cbd5e1", lw=0.9),
        )

    plt.tight_layout(rect=[0.005, 0.055 if len(anchor_list) > 4 else (0.038 if anchor_list else 0.01), 0.995, 0.952])
    out_path = Path(out_png).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, pil_kwargs={"compress_level": 2})
    plt.close(fig)
    return out_path


def _dilate_manifold(m: Any, eps: float = 0.025) -> Any:
    """Fast, guaranteed-valid 3D dilation via 6-axis + diagonal translations to suppress coplanar sliver noise."""
    acc = m
    for dx, dy, dz in (
        (eps, 0, 0), (-eps, 0, 0),
        (0, eps, 0), (0, -eps, 0),
        (0, 0, eps), (0, 0, -eps),
    ):
        acc = acc + m.translate((dx, dy, dz))
    return acc


def compute_volumetric_diff(old_target: Any, new_target: Any) -> dict[str, Any]:
    """Compute exact boolean Added (New - Old), Removed (Old - New), and Unchanged (Old ∩ New) meshes."""
    old_mesh = load_any_as_trimesh(old_target)
    new_mesh = load_any_as_trimesh(new_target)

    m_old = trimesh_to_manifold(old_mesh)
    m_new = trimesh_to_manifold(new_mesh)

    # Exact volumetric differences for accurate mm³ reporting
    exact_added = manifold_to_trimesh(m_new - m_old)
    exact_removed = manifold_to_trimesh(m_old - m_new)
    unchanged_mesh = manifold_to_trimesh(m_old ^ m_new)

    # Clean visual diff meshes free of zero-thickness coplanar re-triangulation slivers
    added_mesh = manifold_to_trimesh(m_new - _dilate_manifold(m_old, 0.025))
    removed_mesh = manifold_to_trimesh(m_old - _dilate_manifold(m_new, 0.025))

    old_vol = float(old_mesh.volume) if len(old_mesh.faces) > 0 else 0.0
    new_vol = float(new_mesh.volume) if len(new_mesh.faces) > 0 else 0.0
    added_vol = float(exact_added.volume) if len(exact_added.faces) > 0 else 0.0
    removed_vol = float(exact_removed.volume) if len(exact_removed.faces) > 0 else 0.0

    return {
        "old_mesh": old_mesh,
        "new_mesh": new_mesh,
        "added_mesh": added_mesh,
        "removed_mesh": removed_mesh,
        "unchanged_mesh": unchanged_mesh,
        "old_volume_mm3": round(old_vol, 2),
        "new_volume_mm3": round(new_vol, 2),
        "added_volume_mm3": round(added_vol, 2),
        "removed_volume_mm3": round(removed_vol, 2),
        "old_size_mm": [round(float(x), 2) for x in (old_mesh.bounds[1] - old_mesh.bounds[0])],
        "new_size_mm": [round(float(x), 2) for x in (new_mesh.bounds[1] - new_mesh.bounds[0])],
    }


def render_diff_sheet(
    old_target: Any,
    new_target: Any,
    out_png: str | Path,
    title: str = "修改前后体积对比 (Diff)",
    precomputed_diff: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Render a 4-panel Red/Green Volumetric Boolean Diff Sheet (`diff_sheet.png`)."""
    _configure_cjk_font()
    diff = precomputed_diff if precomputed_diff is not None else compute_volumetric_diff(old_target, new_target)

    pv_unchanged = trimesh_to_pv(diff["unchanged_mesh"])
    pv_added = trimesh_to_pv(diff["added_mesh"])
    pv_removed = trimesh_to_pv(diff["removed_mesh"])
    edges_added = extract_cad_edges(pv_added, 24.0)
    edges_removed = extract_cad_edges(pv_removed, 24.0)
    edges_unchanged = extract_cad_edges(pv_unchanged, 28.0)

    all_bounds = np.vstack([diff["old_mesh"].bounds, diff["new_mesh"].bounds])
    bmin = np.min(all_bounds, axis=0)
    bmax = np.max(all_bounds, axis=0)
    focal = 0.5 * (bmin + bmax)
    diag = max(float(np.linalg.norm(bmax - bmin)), 10.0)

    cam_views = [
        ("① 等轴测俯视 Diff (+X -Y +Z)", (1.05, -1.25, 0.90), (0, 0, 1)),
        ("② 正面 Front Diff (-Y 视向)",   (0.0, -1.0, 0.0),    (0, 0, 1)),
        ("③ 顶视图 Top Diff (+Z 俯视)",   (0.0, 0.0, 1.0),     (0, 1, 0)),
        ("④ 等轴测仰视 Diff (-X +Y -Z)", (-1.05, 1.20, -0.85), (0, 0, 1)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=150)
    fig.patch.set_facecolor("#f8fafc")

    summary = (
        f"{title}   |   "
        f"■ 绿色 = 新增结构 (+{diff['added_volume_mm3']} mm³)   |   "
        f"■ 红色 = 切除部分 (-{diff['removed_volume_mm3']} mm³)   |   "
        f"□ 灰透 = 未改动保留区\n"
        f"旧尺寸 {diff['old_size_mm']} mm  →  新尺寸 {diff['new_size_mm']} mm   |   "
        f"净体积变化: {diff['new_volume_mm3'] - diff['old_volume_mm3']:+.2f} mm³"
    )
    fig.suptitle(summary, fontsize=11.5, fontweight="bold", color="#0f172a", y=0.975)

    # Upload diff meshes ONCE to shared Plotter and capture all 4 camera angles!
    pl_diff = _get_shared_plotter((720, 540))
    pl_diff.set_background("#ffffff")
    if pv_unchanged.n_points > 0:
        pl_diff.add_mesh(pv_unchanged, color="#94a3b8", opacity=0.22, smooth_shading=True)
    if edges_unchanged.n_points > 0:
        pl_diff.add_mesh(edges_unchanged, color="#94a3b8", line_width=0.9, opacity=0.35)
    if pv_added.n_points > 0:
        pl_diff.add_mesh(pv_added, color="#10b981", opacity=0.92, specular=0.3, smooth_shading=True)
    if edges_added.n_points > 0:
        pl_diff.add_mesh(edges_added, color="#064e3b", line_width=1.8)
    if pv_removed.n_points > 0:
        pl_diff.add_mesh(pv_removed, color="#ef4444", opacity=0.45, smooth_shading=True)
    if edges_removed.n_points > 0:
        pl_diff.add_mesh(edges_removed, color="#991b1b", line_width=1.6)
    pl_diff.camera.enable_parallel_projection()
    pl_diff.camera.focal_point = tuple(float(x) for x in focal)
    pl_diff.camera.parallel_scale = diag * 0.56
    pl_diff.add_axes(line_width=2, color="#0f172a", xlabel="+X", ylabel="+Y", zlabel="+Z")

    for ax, (vtitle, cdir_raw, up) in zip(axes.flat, cam_views):
        cdir = np.asarray(cdir_raw, dtype=float)
        cdir = cdir / np.linalg.norm(cdir)
        dist = max(250.0, diag * 2.8)
        pl_diff.camera.position = tuple(float(x) for x in (focal + dist * cdir))
        pl_diff.camera.up = up
        img = pl_diff.screenshot(transparent_background=False, return_img=True)

        ax.imshow(img)
        ax.set_title(vtitle, fontsize=10.5, fontweight="bold", color="#0f172a")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#cbd5e1")

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.92])
    out_path = Path(out_png).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, pil_kwargs={"compress_level": 2})
    plt.close(fig)

    stats = {k: v for k, v in diff.items() if not k.endswith("_mesh")}
    return out_path, stats
