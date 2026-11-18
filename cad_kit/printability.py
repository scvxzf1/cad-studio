"""3D Printability DFM Scanner, 6-Axis Optimal Auto-Orienter & Native Multi-Object .3mf Exporter.

Analyzes:
  1. First-layer bed contact area (`mm²`), footprint slenderness (tipping / Brim warning).
  2. Face overhang angles relative to -Z build plate:
     - Safe walls & self-supporting chamfers (<= 45 deg)
     - Moderate overhangs (45 ~ 55 deg)
     - Severe unsupported overhangs (> 55 deg)
     - Horizontal bridges vs True Mid-Air Floating Islands (graph local Z-minima above bed -> guaranteed spaghetti risk!).
  3. 6-Orientation Smart Ranking (-Z, +Z, -Y, +Y, -X, +X on bed) + 1-click optimal rotation.
  4. Native named & colored `.3mf` export via `build123d.Mesher` / `lib3mf`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import trimesh

from .inspector import _configure_cjk_font, shape_to_trimesh
from .renderer import extract_cad_edges
from .verify import load_any_as_trimesh

pv.OFF_SCREEN = True


@dataclass(frozen=True)
class OrientationCandidate:
    name: str
    desc_cn: str
    rot_xyz_deg: tuple[float, float, float]


SIX_ORIENTATIONS: list[OrientationCandidate] = [
    OrientationCandidate("-Z_bed", "默认底面贴床 (-Z朝下)", (0.0, 0.0, 0.0)),
    OrientationCandidate("+Z_bed", "顶面翻转贴床 (+Z朝下)", (180.0, 0.0, 0.0)),
    OrientationCandidate("-Y_bed", "正面朝下平放 (-Y朝下)", (-90.0, 0.0, 0.0)),
    OrientationCandidate("+Y_bed", "背面朝下平放 (+Y朝下)", (90.0, 0.0, 0.0)),
    OrientationCandidate("-X_bed", "左侧面贴床侧放 (-X朝下)", (0.0, -90.0, 0.0)),
    OrientationCandidate("+X_bed", "右侧面贴床侧放 (+X朝下)", (0.0, 90.0, 0.0)),
]


def _rotate_mesh_xyz_and_drop_to_bed(mesh: trimesh.Trimesh, rot_deg: tuple[float, float, float]) -> trimesh.Trimesh:
    """Apply intrinsic/extrinsic Rx -> Ry -> Rz rotation matching build123d Rot(rx, ry, rz) and drop min-Z to 0."""
    m = mesh.copy()
    rx, ry, rz = [math.radians(a) for a in rot_deg]
    mx = trimesh.transformations.rotation_matrix(rx, [1, 0, 0])
    my = trimesh.transformations.rotation_matrix(ry, [0, 1, 0])
    mz = trimesh.transformations.rotation_matrix(rz, [0, 0, 1])
    # build123d Rot(rx, ry, rz) applies intrinsic XYZ => matrix M = Rx @ Ry @ Rz? Let's keep exact consistency:
    # For single-axis 90/180 rotations in SIX_ORIENTATIONS, at most one angle is non-zero so order is identical!
    mat = mz @ my @ mx
    m.apply_transform(mat)
    bmin = m.bounds[0]
    cx = 0.5 * (m.bounds[0][0] + m.bounds[1][0])
    cy = 0.5 * (m.bounds[0][1] + m.bounds[1][1])
    m.apply_translation([-cx, -cy, -bmin[2]])
    return m


def _detect_floating_islands(mesh: trimesh.Trimesh, z_bed: float, z_tol: float = 0.28) -> list[dict[str, Any]]:
    """Detect true mid-air ungrounded local Z-minima plateaus above the build plate.

    Theorem: When slicing bottom-up along +Z, a disconnected mid-air island starts at z* > z_bed
    iff a connected plateau of vertices near z* has NO mesh edge leading to any lower vertex (z < z* - eps).
    """
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return []

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    n_v = len(verts)
    z = verts[:, 2]

    # Snap Z into 0.05mm bins to group flat horizontal undersides cleanly
    z_bin = np.round(z / 0.05) * 0.05

    # Build adjacency of vertices within the same Z plateau (|Δz| <= 0.06mm)
    same_mask = np.abs(z[edges[:, 0]] - z[edges[:, 1]]) <= 0.06
    same_edges = edges[same_mask]
    if len(same_edges) > 0:
        row = np.concatenate([same_edges[:, 0], same_edges[:, 1]])
        col = np.concatenate([same_edges[:, 1], same_edges[:, 0]])
        adj = coo_matrix((np.ones(len(row), dtype=np.int8), (row, col)), shape=(n_v, n_v))
        n_comp, labels = connected_components(adj, directed=False)
    else:
        n_comp = n_v
        labels = np.arange(n_v)

    # Mark any vertex that has a strictly lower neighbor (z_nbr < z_self - 0.06mm)
    has_lower = np.zeros(n_v, dtype=bool)
    u, v = edges[:, 0], edges[:, 1]
    has_lower[u] |= (z[v] < z[u] - 0.06)
    has_lower[v] |= (z[u] < z[v] - 0.06)

    # Propagate has_lower across each flat plateau component:
    # If ANY vertex in the plateau connects downward (like a bridge resting on two pillars!),
    # the entire bridge plateau is grounded, NOT a floating island!
    comp_grounded = np.zeros(n_comp, dtype=bool)
    np.logical_or.at(comp_grounded, labels, has_lower)

    # Compute area-weighted outward vertex normals so concave hole floors (normal_z > 0, solid below!)
    # are never mistaken for downward-hanging mid-air islands (normal_z < 0, air below!)
    v_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)

    islands: list[dict[str, Any]] = []
    for cid in np.where(~comp_grounded)[0]:
        v_ids = np.where(labels == cid)[0]
        z_mean = float(np.mean(z[v_ids]))
        if z_mean <= z_bed + z_tol:
            continue
        # True mid-air bottom must face downward into empty air (mean outward normal_z < -0.25)
        if float(np.mean(v_normals[v_ids, 2])) > -0.25:
            continue
        pts = verts[v_ids]
        span_xy = np.ptp(pts[:, :2], axis=0)
        center = np.mean(pts, axis=0)
        islands.append(
            {
                "center_mm": [round(float(center[0]), 2), round(float(center[1]), 2), round(z_mean, 2)],
                "span_xy_mm": [round(float(span_xy[0]), 2), round(float(span_xy[1]), 2)],
                "vertex_count": int(len(v_ids)),
            }
        )

    islands.sort(key=lambda item: item["center_mm"][2])
    return islands


def analyze_mesh_printability(mesh: trimesh.Trimesh) -> dict[str, Any]:
    """Compute comprehensive FDM printability metrics and per-face classification codes.

    Face category codes (`face_classes` int array of length len(mesh.faces)):
      0 = Safe wall / upward surface (overhang <= 45 deg)
      1 = First-layer build plate adhesion surface (at Z = z_bed, facing -Z)
      2 = Moderate overhang or short bridge (45 ~ 55 deg, or horizontal bridge <= 20mm span)
      3 = Severe overhang (> 55 deg above bed) or long unsupported ceiling
    """
    if len(mesh.faces) == 0:
        raise ValueError("Empty mesh in analyze_mesh_printability")

    bmin, bmax = mesh.bounds
    bsize = bmax - bmin
    z_bed = float(bmin[2])

    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    tri_centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    nz = normals[:, 2]

    n_faces = len(mesh.faces)
    face_classes = np.zeros(n_faces, dtype=np.int32)

    # 1. Bed contact faces (facing straight down at z ≈ z_bed)
    on_bed = (nz < -0.97) & (np.abs(tri_centers[:, 2] - z_bed) <= 0.12)
    face_classes[on_bed] = 1
    bed_area = float(np.sum(areas[on_bed]))

    # 2. Above-bed downward-sloping faces:
    # Overhang angle from vertical wall: sin(theta_from_vertical) = -nz (for nz < 0)
    #   theta = 45 deg -> nz = -0.7071
    #   theta = 55 deg -> nz = -0.8192
    above_bed = ~on_bed & (tri_centers[:, 2] > z_bed + 0.22)
    moderate_mask = above_bed & (nz < -0.715) & (nz >= -0.825)
    severe_slope_mask = above_bed & (nz < -0.825) & (nz > -0.985)
    flat_ceiling_mask = above_bed & (nz <= -0.985)

    face_classes[moderate_mask] = 2
    face_classes[severe_slope_mask] = 3

    # 3. Group horizontal downward ceilings (`flat_ceiling_mask`) into connected face patches:
    # Small horizontal overhangs/bridges (e.g. counterbore sacrificial layer, M3/M4 horizontal holes,
    # USB-C slot roof <= 18mm wide with area <= 220 mm²) bridge cleanly on modern FDM printers -> class 2!
    # Larger flat ceilings (> 220 mm² or > 22mm min-width) need support -> class 3!
    bridge_area = 0.0
    severe_ceiling_area = 0.0
    if np.any(flat_ceiling_mask):
        ceil_idx = np.where(flat_ceiling_mask)[0]
        ceil_set = set(int(i) for i in ceil_idx)
        f_adj = mesh.face_adjacency
        mask_adj = np.array([(int(a) in ceil_set and int(b) in ceil_set) for a, b in f_adj], dtype=bool)
        sub_pairs = f_adj[mask_adj]
        remap = {int(orig): k for k, orig in enumerate(ceil_idx)}
        if len(sub_pairs) > 0:
            r = [remap[int(a)] for a, _ in sub_pairs] + [remap[int(b)] for _, b in sub_pairs]
            c = [remap[int(b)] for _, b in sub_pairs] + [remap[int(a)] for a, _ in sub_pairs]
            g = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(len(ceil_idx), len(ceil_idx)))
            nc, clabels = connected_components(g, directed=False)
        else:
            nc = len(ceil_idx)
            clabels = np.arange(len(ceil_idx))

        for cid in range(nc):
            patch_faces = ceil_idx[clabels == cid]
            patch_area = float(np.sum(areas[patch_faces]))
            patch_verts = mesh.vertices[mesh.faces[patch_faces].ravel()]
            span_xy = np.ptp(patch_verts[:, :2], axis=0)
            min_bridge_span = float(np.min(span_xy))
            if min_bridge_span <= 18.5 and patch_area <= 240.0:
                face_classes[patch_faces] = 2
                bridge_area += patch_area
            else:
                face_classes[patch_faces] = 3
                severe_ceiling_area += patch_area

    mod_area = float(np.sum(areas[face_classes == 2]))
    sev_area = float(np.sum(areas[face_classes == 3]))
    total_area = max(float(np.sum(areas)), 1e-3)

    islands = _detect_floating_islands(mesh, z_bed=z_bed)

    # Slenderness / tipping check
    min_base_dim = max(float(min(bsize[0], bsize[1])), 1.0)
    slenderness = float(bsize[2]) / min_base_dim
    bed_ratio = bed_area / max(float(bsize[0] * bsize[1]), 1.0)

    # Printability Score (0 ~ 100): higher is easier & cleaner to print support-free!
    penalty = (
        (sev_area / total_area) * 220.0
        + (mod_area / total_area) * 35.0
        + len(islands) * 22.0
        + (12.0 if bed_area < 25.0 else 0.0)
        + max(0.0, (slenderness - 2.2) * 8.0)
    )
    bonus = min(10.0, bed_ratio * 12.0)
    score = round(max(0.0, min(100.0, 92.0 + bonus - penalty)), 1)

    recommendations: list[str] = []
    if len(islands) > 0:
        recommendations.append(
            f"检测到 {len(islands)} 处半空无支撑孤岛 (最低位于 Z={islands[0]['center_mm'][2]}mm)，直接打印必炒面，必须开启支撑或更换摆放朝向！"
        )
    if sev_area > 15.0:
        recommendations.append(
            f"存在 {sev_area:.1f} mm² 大角度严重悬垂区 (>55°)，建议在切片软件开启局部树状/普通支撑，或参考最佳摆放朝向旋转零件。"
        )
    elif mod_area > 10.0:
        recommendations.append(
            f"含 {bridge_area:.1f} mm² 水平短桥接 / 45°~55° 悬垂，Bambu 等打印机可免支撑直接架桥（建议开启降速悬垂冷却）。"
        )
    if bed_area < 35.0 or slenderness > 2.2:
        recommendations.append(
            f"首层着床面积较小 ({bed_area:.1f} mm²) 或高宽比偏大 ({slenderness:.2f})，切片时强烈建议开启 5~8mm 外沿 Brim 裙边防翘边倒塌！"
        )
    if not recommendations:
        recommendations.append("极佳免支撑 FDM 结构：首层附着充足、无危险悬垂与半空孤岛，可直接平放裸打！")

    return {
        "score": score,
        "needs_support": bool(len(islands) > 0 or sev_area > 20.0),
        "needs_brim": bool(bed_area < 35.0 or slenderness > 2.2),
        "bbox_size_mm": [round(float(x), 2) for x in bsize],
        "bed_contact_area_mm2": round(bed_area, 2),
        "moderate_overhang_or_bridge_mm2": round(mod_area, 2),
        "severe_overhang_mm2": round(sev_area, 2),
        "floating_islands_count": len(islands),
        "floating_islands": islands[:10],
        "slenderness_ratio": round(slenderness, 2),
        "recommendations": recommendations,
        "_face_classes": face_classes,
    }


def rank_six_orientations(target: Any) -> list[dict[str, Any]]:
    """Evaluate all 6 principal build-plate orientations and sort best-first."""
    base_mesh = load_any_as_trimesh(target)
    results: list[dict[str, Any]] = []
    for cand in SIX_ORIENTATIONS:
        m_rot = _rotate_mesh_xyz_and_drop_to_bed(base_mesh, cand.rot_xyz_deg)
        rep = analyze_mesh_printability(m_rot)
        rep.pop("_face_classes", None)
        results.append(
            {
                "orientation": cand.name,
                "desc": cand.desc_cn,
                "rot_xyz_deg": list(cand.rot_xyz_deg),
                **rep,
            }
        )
    # Sort by highest score, then fewest islands, then lowest severe overhang, then largest bed area
    results.sort(
        key=lambda r: (
            -r["score"],
            r["floating_islands_count"],
            r["severe_overhang_mm2"],
            -r["bed_contact_area_mm2"],
        )
    )
    return results


def apply_optimal_orientation(part_or_mesh: Any, rot_xyz_deg: Sequence[float]) -> Any:
    """Rotate a build123d Part or trimesh.Trimesh to the optimal bed orientation with Z_min = 0 and XY centered."""
    rx, ry, rz = float(rot_xyz_deg[0]), float(rot_xyz_deg[1]), float(rot_xyz_deg[2])
    if hasattr(part_or_mesh, "solids") and hasattr(part_or_mesh, "bounding_box"):
        import build123d as bd
        rotated = bd.Rot(rx, ry, rz) * part_or_mesh
        bb = rotated.bounding_box()
        cx = 0.5 * (bb.min.X + bb.max.X)
        cy = 0.5 * (bb.min.Y + bb.max.Y)
        return bd.Pos(-cx, -cy, -bb.min.Z) * rotated
    return _rotate_mesh_xyz_and_drop_to_bed(load_any_as_trimesh(part_or_mesh), (rx, ry, rz))


def render_printability_sheet(
    target: Any,
    out_png: str | Path,
    title: str = "Part",
    ranking: list[dict[str, Any]] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Render a 4-panel FDM Printability & Overhang Heatmap Blueprint (`printability_sheet.png`)."""
    _configure_cjk_font()
    mesh = load_any_as_trimesh(target)
    report = analyze_mesh_printability(mesh)
    f_cls = report.pop("_face_classes")
    orient_ranks = ranking if ranking is not None else rank_six_orientations(mesh)

    # Build colored PyVista PolyData using per-cell RGB
    palette = np.array(
        [
            [96, 165, 250],   # 0: Safe wall/top (#60a5fa)
            [16, 185, 129],   # 1: Bed contact (#10b981)
            [245, 158, 11],   # 2: Moderate overhang / bridge (#f59e0b)
            [239, 68, 68],    # 3: Severe overhang (>55 deg) (#ef4444)
        ],
        dtype=np.uint8,
    )
    cell_rgb = palette[f_cls]

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    padded = np.column_stack([np.full(len(faces), 3, dtype=np.int64), faces]).ravel()
    poly = pv.PolyData(verts, padded)
    poly.cell_data["RGB"] = cell_rgb
    edges = extract_cad_edges(poly, angle=25.0)

    bmin, bmax = mesh.bounds
    focal = 0.5 * (bmin + bmax)
    diag = max(float(np.linalg.norm(bmax - bmin)), 10.0)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.2), dpi=150)
    fig.patch.set_facecolor("#f8fafc")

    best = orient_ranks[0]
    status_tag = "免支撑直接打印 OK" if not report["needs_support"] else "建议加支撑或换向 [!]"
    hdr = (
        f"3D 打印可制造性 (DFM) 体检报告: {title}   |   当前评分: {report['score']}/100 ({status_tag})\n"
        f"■ 绿色 = 首层着床面 ({report['bed_contact_area_mm2']} mm²)   |   "
        f"■ 黄色 = 架桥/轻度悬垂 ({report['moderate_overhang_or_bridge_mm2']} mm²)   |   "
        f"■ 红色 = >55°重度悬垂 ({report['severe_overhang_mm2']} mm²)   |   "
        f"半空孤岛: {report['floating_islands_count']} 处"
    )
    fig.suptitle(hdr, fontsize=11.5, fontweight="bold", color="#0f172a", y=0.975)

    cam_specs = [
        (axes[0, 0], "① 等轴俯视 DFM 热力图 (+X -Y +Z)", (1.05, -1.25, 0.88), (0, 0, 1)),
        (axes[0, 1], "② 等轴仰视悬垂与孤岛检查 (-X +Y -Z)", (-1.05, 1.20, -0.85), (0, 0, 1)),
        (axes[1, 0], "③ 热床贴合底视图 Bottom (-Z 仰视: 绿色越大越稳)", (0.0, 0.0, -1.0), (0, -1, 0)),
    ]

    for ax, vtitle, cdir_raw, up in cam_specs:
        pl = pv.Plotter(off_screen=True, window_size=(700, 520))
        pl.set_background("#ffffff")
        pl.enable_lightkit()
        pl.add_mesh(poly, scalars="RGB", rgb=True, preference="cell", ambient=0.42, diffuse=0.65)
        if edges.n_points > 0:
            pl.add_mesh(edges, color="#0f172a", line_width=1.6)
        # Mark floating islands with bright magenta spheres if any!
        for isl in report["floating_islands"]:
            sp = pv.Sphere(radius=max(diag * 0.028, 1.2), center=isl["center_mm"])
            pl.add_mesh(sp, color="#d946ef")

        cdir = np.asarray(cdir_raw, dtype=float)
        cdir = cdir / np.linalg.norm(cdir)
        dist = max(250.0, diag * 2.7)
        pl.camera.enable_parallel_projection()
        pl.camera.focal_point = tuple(float(x) for x in focal)
        pl.camera.position = tuple(float(x) for x in (focal + dist * cdir))
        pl.camera.up = up
        pl.camera.parallel_scale = diag * 0.56
        pl.add_axes(line_width=2, color="#0f172a", xlabel="+X", ylabel="+Y", zlabel="+Z")
        img = pl.screenshot(return_img=True)
        pl.close()

        ax.imshow(img)
        ax.set_title(vtitle, fontsize=10.2, fontweight="bold", color="#0f172a")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#cbd5e1")

    # Panel 4: 6-Orientation Smart Ranking Horizontal Bar Chart + Actionable Advice
    ax_bar = axes[1, 1]
    ax_bar.set_facecolor("#ffffff")
    labels_y = [
        f"{'★推荐 ' if idx == 0 else ''}{r['desc']}\n(着床{r['bed_contact_area_mm2']:.0f}mm² | 重悬垂{r['severe_overhang_mm2']:.0f}mm² | 孤岛{r['floating_islands_count']})"
        for idx, r in enumerate(orient_ranks)
    ][::-1]
    scores_x = [r["score"] for r in orient_ranks][::-1]
    bar_cols = ["#10b981" if s >= 85 else ("#f59e0b" if s >= 65 else "#ef4444") for s in scores_x]

    bars = ax_bar.barh(range(len(labels_y)), scores_x, color=bar_cols, height=0.58, edgecolor="#0f172a", linewidth=0.7)
    ax_bar.set_yticks(range(len(labels_y)))
    ax_bar.set_yticklabels(labels_y, fontsize=8.3, color="#1e293b")
    ax_bar.set_xlim(0, 108)
    ax_bar.set_xlabel("可打印性综合评分 (0~100, 越高越免支撑稳固)", fontsize=9)
    ax_bar.set_title(f"④ 六向热床摆放智能寻优对比 (最优: {best['desc']})", fontsize=10.2, fontweight="bold", color="#0f172a")
    ax_bar.grid(axis="x", color="#e2e8f0", linestyle="--", linewidth=0.7)
    for b, s in zip(bars, scores_x):
        ax_bar.text(s + 1.5, b.get_y() + b.get_height() * 0.5, f"{s:.1f}", va="center", fontsize=8.5, fontweight="bold", color="#0f172a")

    rec_str = "工艺建议: " + "  ".join(report["recommendations"])
    fig.text(
        0.5,
        0.008,
        rec_str,
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec="#cbd5e1", lw=0.9),
    )

    plt.tight_layout(rect=[0.01, 0.04, 0.99, 0.925])
    out_path = Path(out_png).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    report["orientation_ranking"] = orient_ranks
    report["best_orientation"] = best
    return out_path, report


def export_3mf(
    items: Sequence[dict[str, Any]] | Any,
    out_3mf: str | Path,
) -> Path:
    """Export one or multiple named & colored parts to a BambuStudio-ready native `.3mf` package!

    `items` can be:
      - A single build123d Part or trimesh.Trimesh
      - Or a sequence of dicts: `[{"name": "bottom_box", "shape": part, "color": "#3b82f6"}, ...]`
    """
    import build123d as bd
    import lib3mf

    out_path = Path(out_3mf).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(items, (list, tuple)):
        items = [{"name": out_path.stem, "shape": items, "color": "#3b82f6"}]

    # Check if all shapes are B-Rep shapes supported directly by build123d.Mesher
    all_brep = all(hasattr(it["shape"], "solids") and hasattr(it["shape"], "tessellate") for it in items)
    if all_brep:
        mesher = bd.Mesher()
        for it in items:
            solids = it["shape"].solids()
            col_hex = it.get("color", "#3b82f6")
            base_name = str(it.get("name", "part"))
            for idx, s in enumerate(solids):
                s_copy = s.moved(bd.Location())
                s_copy.label = base_name if len(solids) == 1 else f"{base_name}_{idx+1}"
                s_copy.color = bd.Color(col_hex)
                mesher.add_shape(s_copy, angular_deflection=0.12, linear_deflection=0.03)
        mesher.write(str(out_path))
        return out_path

    # Mixed B-Rep / trimesh direct lib3mf writer preserving object names!
    wrapper = lib3mf.get_wrapper()
    model = wrapper.CreateModel()
    model.SetUnit(lib3mf.ModelUnit.MilliMeter)

    for it in items:
        mesh = load_any_as_trimesh(it["shape"])
        if len(mesh.faces) == 0:
            continue
        mesh_obj = model.AddMeshObject()
        mesh_obj.SetName(str(it.get("name", "part")))

        pos_list = []
        for v in np.asarray(mesh.vertices, dtype=np.float32):
            p = lib3mf.Position()
            p.Coordinates[0] = float(v[0])
            p.Coordinates[1] = float(v[1])
            p.Coordinates[2] = float(v[2])
            pos_list.append(p)

        tri_list = []
        for f in np.asarray(mesh.faces, dtype=np.uint32):
            t = lib3mf.Triangle()
            t.Indices[0] = int(f[0])
            t.Indices[1] = int(f[1])
            t.Indices[2] = int(f[2])
            tri_list.append(t)

        mesh_obj.SetGeometry(pos_list, tri_list)
        model.AddBuildItem(mesh_obj, wrapper.GetIdentityTransform())

    writer = model.QueryWriter("3mf")
    writer.WriteToFile(str(out_path))
    return out_path
