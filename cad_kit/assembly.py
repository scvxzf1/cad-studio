"""Multi-Part CAD Assembly System: N×N Collision/Clearance Matrix, Exploded View, BOM & Print-Plate .3mf.

Capabilities:
  1. Compose multiple printable/machinable parts + COTS reference parts (PCBs, rails, motors)
     with distinct colors and explicit or automatic 3D explosion vectors.
  2. N×N Pairwise Interference & Minimum Clearance Matrix (`verify_assembly`) using `manifold3d`
     boolean intersections + `scipy.spatial.cKDTree` surface-to-surface distance checks.
  3. 6-Panel Multi-Color Assembly & Exploded Engineering Blueprint (`assembly_sheet.png`):
     Assembled Iso, Exploded Iso with part callout numbers, Front, Right, Top, and Multi-Color Section Cut!
  4. Automatic Bill of Materials (`bom.md` + `bom.json`): manufactured part dimensions, volumes,
     estimated filament gram weights, plus aggregated fastener & COTS hardware list.
  5. Exports `<name>_assembly.step`, `<name>_assembly.3mf`, `<name>_print_plate.3mf` (non-reference
     parts auto-oriented & packed onto Z=0!), and `viewer.html` with live 💥 Explode Slider!
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
import trimesh

from .features import FeatureRegistry
from .inspector import _configure_cjk_font
from .printability import apply_optimal_orientation, export_3mf, rank_six_orientations
from .renderer import extract_cad_edges, trimesh_to_pv
from .tolerances import ManufacturingMode
from .verify import load_any_as_trimesh, manifold_to_trimesh, trimesh_to_manifold
from .viewer_html import export_viewer_html

pv.OFF_SCREEN = True

ASSEMBLY_PALETTE: list[tuple[str, tuple[float, float, float]]] = [
    ("#3b82f6", (0.23, 0.51, 0.96)),  # Royal Blue
    ("#10b981", (0.06, 0.73, 0.51)),  # Emerald Green
    ("#f59e0b", (0.96, 0.62, 0.04)),  # Amber Gold
    ("#8b5cf6", (0.55, 0.36, 0.96)),  # Violet
    ("#ec4899", (0.93, 0.28, 0.60)),  # Rose Pink
    ("#06b6d4", (0.02, 0.71, 0.83)),  # Cyan
    ("#f97316", (0.98, 0.45, 0.09)),  # Orange
]


def _hex_to_rgb01(h: str) -> tuple[float, float, float]:
    s = h.strip().lstrip("#")
    if len(s) == 6:
        return (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0)
    return (0.23, 0.51, 0.96)


@dataclass
class AssemblyNode:
    part_id: str
    label: str
    shape: Any
    color_hex: str
    rgb: tuple[float, float, float]
    is_reference: bool = False
    explode_vec: tuple[float, float, float] | None = None
    material: str = "PETG / PLA"
    density_g_cm3: float = 1.25
    infill_factor: float = 0.42       # Effective shell+infill solid fraction for realistic FDM weight estimate
    qty: int = 1
    registry: FeatureRegistry | None = None
    auto_orient_for_plate: bool = True


@dataclass
class HardwareBOMItem:
    spec: str
    qty: int
    category: str = "紧固件"
    note: str = ""


@dataclass
class Assembly:
    """Container & orchestrator for multi-part mechanical assemblies."""

    name: str
    description: str = ""
    mode: ManufacturingMode = "fdm"
    nodes: list[AssemblyNode] = field(default_factory=list)
    hardware: list[HardwareBOMItem] = field(default_factory=list)

    def add_part(
        self,
        part_id: str,
        shape: Any,
        label: str = "",
        pos: Sequence[float] = (0.0, 0.0, 0.0),
        rot: Sequence[float] = (0.0, 0.0, 0.0),
        explode_vec: Sequence[float] | None = None,
        color_hex: str | None = None,
        is_reference: bool = False,
        material: str | None = None,
        qty: int = 1,
        registry: FeatureRegistry | None = None,
        auto_orient_for_plate: bool = True,
    ) -> AssemblyNode:
        """Add a positioned sub-part or reference COTS component to the assembly."""
        # Apply optional rigid placement (rot -> pos)
        placed = shape
        has_rot = any(abs(float(a)) > 1e-6 for a in rot)
        has_pos = any(abs(float(x)) > 1e-6 for x in pos)
        if isinstance(shape, (str, Path)):
            p = Path(shape).expanduser().resolve()
            if p.suffix.lower() in (".step", ".stp", ".brep"):
                import build123d as bd
                placed = bd.import_brep(str(p)) if p.suffix.lower() == ".brep" else bd.import_step(str(p))
            else:
                placed = load_any_as_trimesh(p)

        if hasattr(placed, "solids") and (has_rot or has_pos):
            import build123d as bd
            if has_rot:
                placed = bd.Rot(float(rot[0]), float(rot[1]), float(rot[2])) * placed
            if has_pos:
                placed = bd.Pos(float(pos[0]), float(pos[1]), float(pos[2])) * placed
        elif isinstance(placed, trimesh.Trimesh) and (has_rot or has_pos):
            placed = placed.copy()
            if has_rot:
                import math
                rx, ry, rz = [math.radians(float(a)) for a in rot]
                mat = (
                    trimesh.transformations.rotation_matrix(rz, [0, 0, 1])
                    @ trimesh.transformations.rotation_matrix(ry, [0, 1, 0])
                    @ trimesh.transformations.rotation_matrix(rx, [1, 0, 0])
                )
                placed.apply_transform(mat)
            if has_pos:
                placed.apply_translation([float(pos[0]), float(pos[1]), float(pos[2])])

        if color_hex is None:
            if is_reference:
                chex, crgb = "#94a3b8", (0.58, 0.64, 0.72)
            else:
                non_ref_idx = sum(1 for n in self.nodes if not n.is_reference)
                chex, crgb = ASSEMBLY_PALETTE[non_ref_idx % len(ASSEMBLY_PALETTE)]
        else:
            chex, crgb = color_hex, _hex_to_rgb01(color_hex)

        ev = (float(explode_vec[0]), float(explode_vec[1]), float(explode_vec[2])) if explode_vec is not None else None
        mat_str = material or ("外购标准/参考件" if is_reference else ("PETG / PLA" if self.mode == "fdm" else "6061 铝合金"))

        node = AssemblyNode(
            part_id=part_id,
            label=label or part_id,
            shape=placed,
            color_hex=chex,
            rgb=crgb,
            is_reference=is_reference,
            explode_vec=ev,
            material=mat_str,
            qty=qty,
            registry=registry,
            auto_orient_for_plate=auto_orient_for_plate,
        )
        self.nodes.append(node)
        return node

    def add_hardware(self, spec: str, qty: int = 1, category: str = "紧固件", note: str = "") -> None:
        self.hardware.append(HardwareBOMItem(spec=spec, qty=qty, category=category, note=note))


def _resolve_explode_vectors(nodes: list[AssemblyNode], meshes: list[trimesh.Trimesh]) -> list[tuple[float, float, float]]:
    """Fill in sensible outward explosion vectors for any node where explode_vec is None."""
    if not meshes:
        return []
    all_bounds = np.array([0.5 * (m.bounds[0] + m.bounds[1]) for m in meshes])
    asm_center = np.mean(all_bounds, axis=0)
    asm_span = np.ptp(np.vstack([m.bounds for m in meshes]), axis=0)
    scale = max(float(np.max(asm_span)) * 0.42, 18.0)

    out: list[tuple[float, float, float]] = []
    for n, m in zip(nodes, meshes):
        if n.explode_vec is not None:
            out.append(n.explode_vec)
            continue
        c = 0.5 * (m.bounds[0] + m.bounds[1])
        delta = c - asm_center
        norm = float(np.linalg.norm(delta))
        if norm < 1.0:
            out.append((0.0, 0.0, 0.0))
        else:
            # Emphasize primary dominant axis for clean engineering axial explosion!
            dom = int(np.argmax(np.abs(delta)))
            vec = np.zeros(3)
            vec[dom] = math.copysign(scale, float(delta[dom])) if "math" in globals() else (scale if delta[dom] >= 0 else -scale)
            out.append((round(float(vec[0]), 2), round(float(vec[1]), 2), round(float(vec[2]), 2)))
    return out


def verify_assembly(asm: Assembly) -> dict[str, Any]:
    """Run N×N pairwise 3D boolean interference & minimum clearance matrix checks."""
    meshes = [load_any_as_trimesh(n.shape) for n in asm.nodes]
    manifolds = [trimesh_to_manifold(m) for m in meshes]
    n_cnt = len(asm.nodes)

    pair_reports: list[dict[str, Any]] = []
    issues: list[str] = []

    for i in range(n_cnt):
        bi_min, bi_max = meshes[i].bounds
        for j in range(i + 1, n_cnt):
            bj_min, bj_max = meshes[j].bounds
            # AABB separation gap along each axis
            axis_gaps = np.maximum(0.0, np.maximum(bi_min - bj_max, bj_min - bi_max))
            aabb_dist = float(np.linalg.norm(axis_gaps))

            ni, nj = asm.nodes[i], asm.nodes[j]
            inter_vol = 0.0
            collision_bbox = None
            min_clearance_mm: float | None = None

            if aabb_dist <= 0.01:
                inter_m = manifold_to_trimesh(manifolds[i] ^ manifolds[j])
                inter_vol = float(inter_m.volume) if len(inter_m.faces) > 0 else 0.0
                if inter_vol > 0.015:
                    ib0, ib1 = inter_m.bounds
                    collision_bbox = {
                        "min_mm": [round(float(x), 2) for x in ib0],
                        "max_mm": [round(float(x), 2) for x in ib1],
                    }
                    issues.append(
                        f"零件碰撞: [{ni.label}] 与 [{nj.label}] 干涉体积 = {inter_vol:.2f} mm³ "
                        f"@ X[{ib0[0]:.1f},{ib1[0]:.1f}] Y[{ib0[1]:.1f},{ib1[1]:.1f}] Z[{ib0[2]:.1f},{ib1[2]:.1f}]"
                    )

            if inter_vol <= 0.015 and aabb_dist < 10.0 and len(meshes[i].faces) > 0 and len(meshes[j].faces) > 0:
                pts_i, _ = trimesh.sample.sample_surface(meshes[i], 2200)
                pts_j, _ = trimesh.sample.sample_surface(meshes[j], 2200)
                dists, _ = cKDTree(pts_i).query(pts_j, k=1)
                min_clearance_mm = round(float(np.min(dists)), 3)

            if inter_vol > 0.015 or (min_clearance_mm is not None and min_clearance_mm < 6.0):
                status = (
                    "[!] 干涉碰撞"
                    if inter_vol > 0.015
                    else ("OK 贴合接触" if (min_clearance_mm is not None and min_clearance_mm <= 0.08) else "OK 安全间隙")
                )
                pair_reports.append(
                    {
                        "part_a": ni.label,
                        "part_b": nj.label,
                        "collision_volume_mm3": round(inter_vol, 3),
                        "min_clearance_mm": 0.0 if inter_vol > 0.015 else min_clearance_mm,
                        "status": status,
                        "collision_bbox": collision_bbox,
                    }
                )

    all_bounds = np.vstack([m.bounds for m in meshes])
    asm_min = np.min(all_bounds, axis=0)
    asm_max = np.max(all_bounds, axis=0)
    return {
        "ok": len(issues) == 0,
        "parts_count": n_cnt,
        "printable_parts_count": sum(1 for n in asm.nodes if not n.is_reference),
        "bbox": {
            "min_mm": [round(float(x), 2) for x in asm_min],
            "max_mm": [round(float(x), 2) for x in asm_max],
            "size_mm": [round(float(x), 2) for x in (asm_max - asm_min)],
        },
        "pairs": pair_reports,
        "issues": issues,
    }


def render_assembly_sheet(asm: Assembly, out_png: str | Path) -> Path:
    """Render a 6-panel Multi-Color Assembly + Exploded View + Multi-Color Section Blueprint."""
    _configure_cjk_font()
    meshes = [load_any_as_trimesh(n.shape) for n in asm.nodes]
    exp_vecs = _resolve_explode_vectors(asm.nodes, meshes)

    polys = [trimesh_to_pv(m) for m in meshes]
    edge_polys = [extract_cad_edges(p, 25.0) for p in polys]

    all_b = np.vstack([m.bounds for m in meshes])
    bmin, bmax = np.min(all_b, axis=0), np.max(all_b, axis=0)
    focal = 0.5 * (bmin + bmax)
    diag = max(float(np.linalg.norm(bmax - bmin)), 15.0)

    # Compute exploded bounds for Panel 2 camera framing
    exp_bounds = []
    for m, ev in zip(meshes, exp_vecs):
        off = np.asarray(ev, dtype=float)
        exp_bounds.append(m.bounds[0] + off)
        exp_bounds.append(m.bounds[1] + off)
    eb_arr = np.vstack(exp_bounds)
    exp_focal = 0.5 * (np.min(eb_arr, axis=0) + np.max(eb_arr, axis=0))
    exp_diag = max(float(np.linalg.norm(np.max(eb_arr, axis=0) - np.min(eb_arr, axis=0))), diag)

    fig, axes = plt.subplots(2, 3, figsize=(19.5, 11.2), dpi=150)
    fig.patch.set_facecolor("#f1f5f9")
    bsize = bmax - bmin
    fig.suptitle(
        f"多零件总装与爆炸拆解工程图: {asm.name}   |   "
        f"总装包络尺寸: {bsize[0]:.1f} × {bsize[1]:.1f} × {bsize[2]:.1f} mm   |   "
        f"组件总数: {len(asm.nodes)} 件 ({sum(1 for n in asm.nodes if not n.is_reference)} 自制件 + {sum(1 for n in asm.nodes if n.is_reference)} 参考件)",
        fontsize=12.5,
        fontweight="bold",
        color="#0f172a",
        y=0.975,
    )

    def _add_assembly_actors(pl: pv.Plotter, explode_scale: float = 0.0, section_y: float | None = None) -> None:
        for idx, (node, m, poly, edg, ev) in enumerate(zip(asm.nodes, meshes, polys, edge_polys, exp_vecs)):
            off = np.asarray(ev, dtype=float) * explode_scale
            op = 0.42 if node.is_reference else 1.0
            if section_y is not None:
                # Boolean half-space cut along Y <= section_y to reveal multi-color internal mating cross-sections!
                try:
                    span = np.maximum(bmax - bmin, 20.0) * 3.0
                    cut_box = trimesh.creation.box(
                        bounds=[
                            [focal[0] - span[0], bmin[1] - span[1], focal[2] - span[2]],
                            [focal[0] + span[0], section_y, focal[2] + span[2]],
                        ]
                    )
                    sliced_m = manifold_to_trimesh(trimesh_to_manifold(m) - trimesh_to_manifold(cut_box))
                    s_pv = trimesh_to_pv(sliced_m)
                    s_edg = extract_cad_edges(s_pv, 25.0)
                    if s_pv.n_points > 0:
                        pl.add_mesh(s_pv, color=node.color_hex, opacity=op, specular=0.3, ambient=0.34, smooth_shading=True)
                    if s_edg.n_points > 0:
                        pl.add_mesh(s_edg, color="#0f172a", line_width=1.6, opacity=0.85)
                except Exception:
                    pass
                continue

            p_disp = poly.translate(off, inplace=False) if np.linalg.norm(off) > 1e-4 else poly
            e_disp = edg.translate(off, inplace=False) if (edg.n_points > 0 and np.linalg.norm(off) > 1e-4) else edg
            if p_disp.n_points > 0:
                pl.add_mesh(p_disp, color=node.color_hex, opacity=op, specular=0.32, ambient=0.32, smooth_shading=True)
            if e_disp.n_points > 0:
                pl.add_mesh(e_disp, color="#0f172a", line_width=1.6 if not node.is_reference else 1.0, opacity=0.88 if not node.is_reference else 0.45)

            if explode_scale > 0.01 and np.linalg.norm(off) > 0.5:
                c_home = 0.5 * (m.bounds[0] + m.bounds[1])
                c_exp = c_home + off
                line = pv.Line(c_home, c_exp)
                pl.add_mesh(line, color=node.color_hex, line_width=2.2)

    panel_configs = [
        (axes[0, 0], "① 装配闭合态等轴测 (Assembled Iso)", (1.05, -1.25, 0.88), (0, 0, 1), focal, diag * 0.55, 0.0, None),
        (axes[0, 1], "② 3D 爆炸拆解图 (Exploded View — 沿装配轴展开)", (1.08, -1.28, 0.86), (0, 0, 1), exp_focal, exp_diag * 0.54, 1.0, None),
        (axes[0, 2], f"③ 总装中心剖视图 Section (Y = {focal[1]:.1f} mm 内部配合核验)", (0.58, -1.38, 0.62), (0, 0, 1), focal, diag * 0.52, 0.0, float(focal[1])),
        (axes[1, 0], "④ 总装正视图 Front (-Y 视向)", (0.0, -1.0, 0.0), (0, 0, 1), focal, diag * 0.52, 0.0, None),
        (axes[1, 1], "⑤ 总装右视图 Right (+X 视向)", (1.0, 0.0, 0.0), (0, 0, 1), focal, diag * 0.52, 0.0, None),
        (axes[1, 2], "⑥ 总装顶视图 Top (+Z 俯视)", (0.0, 0.0, 1.0), (0, 1, 0), focal, diag * 0.52, 0.0, None),
    ]

    for ax, ptitle, cdir_raw, up, fc, pscale, escale, sec_y in panel_configs:
        pl = pv.Plotter(off_screen=True, window_size=(720, 540))
        pl.set_background("#ffffff")
        pl.enable_lightkit()
        _add_assembly_actors(pl, explode_scale=escale, section_y=sec_y)

        cdir = np.asarray(cdir_raw, dtype=float)
        cdir = cdir / np.linalg.norm(cdir)
        dist = max(300.0, pscale * 6.0)
        pl.camera.enable_parallel_projection()
        pl.camera.focal_point = tuple(float(x) for x in fc)
        pl.camera.position = tuple(float(x) for x in (fc + dist * cdir))
        pl.camera.up = up
        pl.camera.parallel_scale = pscale
        pl.add_axes(line_width=2, color="#0f172a", xlabel="+X", ylabel="+Y", zlabel="+Z")

        img = pl.screenshot(return_img=True)
        pl.close()

        ax.imshow(img)
        ax.set_title(ptitle, fontsize=10.2, fontweight="bold", color="#0f172a")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#cbd5e1")

    # Color-coded Assembly Component Legend Bar along bottom
    legend_parts = [
        f"■ [{i+1}] {n.label} ({'参考件' if n.is_reference else n.part_id}, {n.color_hex})"
        for i, n in enumerate(asm.nodes)
    ]
    fig.text(
        0.5,
        0.008,
        "总装零件图例:   " + "    |    ".join(legend_parts),
        ha="center",
        va="bottom",
        fontsize=8.8,
        fontweight="bold",
        color="#1e293b",
        bbox=dict(boxstyle="round,pad=0.38", fc="#ffffff", ec="#cbd5e1", lw=0.9),
    )

    plt.tight_layout(rect=[0.008, 0.042, 0.992, 0.945])
    out_path = Path(out_png).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def build_bom_data(asm: Assembly, check_res: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Generate structured BOM dictionary and formatted Markdown BOM & Clearance report."""
    part_rows: list[dict[str, Any]] = []
    total_print_grams = 0.0

    for idx, n in enumerate(asm.nodes, start=1):
        m = load_any_as_trimesh(n.shape)
        sz = m.bounds[1] - m.bounds[0]
        vol_cm3 = float(m.volume) / 1000.0 if len(m.faces) > 0 else 0.0
        est_g = 0.0 if n.is_reference else round(vol_cm3 * n.density_g_cm3 * n.infill_factor * n.qty, 1)
        total_print_grams += est_g
        part_rows.append(
            {
                "index": idx,
                "part_id": n.part_id,
                "label": n.label,
                "type": "外购/参考件" if n.is_reference else "自制结构件",
                "qty": n.qty,
                "size_mm": [round(float(x), 2) for x in sz],
                "volume_cm3": round(vol_cm3, 2),
                "est_filament_g": est_g,
                "material": n.material,
            }
        )

    hw_rows = [
        {"spec": h.spec, "category": h.category, "qty": h.qty, "note": h.note}
        for h in asm.hardware
    ]

    bom_dict = {
        "assembly_name": asm.name,
        "description": asm.description,
        "mode": asm.mode,
        "total_estimated_print_weight_g": round(total_print_grams, 1),
        "parts": part_rows,
        "hardware": hw_rows,
        "clearance_matrix": check_res.get("pairs", []),
    }

    md = [
        f"# 装配体 BOM 物料清单与配合体检报告: `{asm.name}`",
        "",
        f"> {asm.description or '多零件机械装配总成'}  ｜  工艺档位: `{asm.mode.upper()}`  ｜  预估打印总耗材: **约 `{total_print_grams:.1f} g`**",
        "",
        "## 一、 结构件与模块明细表 (Parts List)",
        "",
        "| # | 代号 | 名称 | 属性 | 数量 | 包络尺寸 X×Y×Z (mm) | 实体体积 (cm³) | 预估打印克重 (g) | 材料 |",
        "| :-: | :--- | :--- | :---: | :-: | :--- | ---: | ---: | :--- |",
    ]
    for r in part_rows:
        sz_s = f"{r['size_mm'][0]} × {r['size_mm'][1]} × {r['size_mm'][2]}"
        g_s = "-" if r["type"] == "外购/参考件" else f"{r['est_filament_g']} g"
        md.append(
            f"| {r['index']} | `{r['part_id']}` | **{r['label']}** | {r['type']} | {r['qty']} | `{sz_s}` | `{r['volume_cm3']}` | {g_s} | {r['material']} |"
        )

    md.extend(
        [
            "",
            "## 二、 紧固件与外购标准件采购/备料清单 (Hardware BOM)",
            "",
            "| 规格型号 | 类别 | 数量 | 装配用途说明 |",
            "| :--- | :---: | :-: | :--- |",
        ]
    )
    if hw_rows:
        for h in hw_rows:
            md.append(f"| **{h['spec']}** | {h['category']} | `{h['qty']}` | {h['note']} |")
    else:
        md.append("| (未声明额外紧固件) | - | - | - |")

    md.extend(
        [
            "",
            "## 三、 N×N 零件两两干涉与最小间隙体检表",
            "",
            "| 零件 A | 零件 B | 状态 | 干涉体积 (mm³) | 实测最小表面间隙 (mm) |",
            "| :--- | :--- | :---: | ---: | ---: |",
        ]
    )
    for pr in check_res.get("pairs", []):
        cl_str = f"`{pr['min_clearance_mm']}`" if pr["min_clearance_mm"] is not None else "-"
        md.append(f"| {pr['part_a']} | {pr['part_b']} | {pr['status']} | `{pr['collision_volume_mm3']}` | {cl_str} |")

    return bom_dict, "\n".join(md) + "\n"


def export_print_plate_3mf(asm: Assembly, out_3mf: str | Path, spacing_mm: float = 10.0) -> Path:
    """Auto-orient non-reference printable parts onto Z=0 and pack them neatly into a ready-to-slice `.3mf`!"""
    import build123d as bd

    printable = [n for n in asm.nodes if not n.is_reference]
    cursor_x = 0.0
    packed_items: list[dict[str, Any]] = []

    for n in printable:
        shape = n.shape
        if n.auto_orient_for_plate:
            ranks = rank_six_orientations(shape)
            best_rot = ranks[0]["rot_xyz_deg"]
            shape = apply_optimal_orientation(shape, best_rot)
        else:
            shape = apply_optimal_orientation(shape, (0.0, 0.0, 0.0))

        m = load_any_as_trimesh(shape)
        bmin, bmax = m.bounds
        w_x = float(bmax[0] - bmin[0])
        shift_x = cursor_x + w_x * 0.5
        if hasattr(shape, "solids"):
            placed = bd.Pos(shift_x, 0.0, 0.0) * shape
        else:
            placed = m.copy()
            placed.apply_translation([shift_x, 0.0, 0.0])

        packed_items.append({"name": n.part_id, "shape": placed, "color": n.color_hex})
        cursor_x += w_x + spacing_mm

    return export_3mf(packed_items, out_3mf)


def run_assembly_build(asm: Assembly, out_dir: str | Path) -> dict[str, Any]:
    """Execute full assembly pipeline: N×N verify, STEP + Assembled/Plate .3mf, 6-view sheet, BOM & Explode HTML."""
    import build123d as bd

    od = Path(out_dir).expanduser().resolve()
    od.mkdir(parents=True, exist_ok=True)

    check = verify_assembly(asm)
    meshes = [load_any_as_trimesh(n.shape) for n in asm.nodes]
    exp_vecs = _resolve_explode_vectors(asm.nodes, meshes)

    # 1. Export combined STEP if all shapes are B-Rep
    step_path: Path | None = od / f"{asm.name}_assembly.step"
    if all(hasattr(n.shape, "solids") for n in asm.nodes):
        comp = bd.Compound(children=[n.shape for n in asm.nodes])
        bd.export_step(comp, str(step_path))
    else:
        step_path = None

    # 2. Export Assembled colored .3mf + Auto-Oriented Print-Plate .3mf!
    asm_3mf = od / f"{asm.name}_assembly.3mf"
    export_3mf(
        [{"name": n.part_id, "shape": n.shape, "color": n.color_hex} for n in asm.nodes],
        asm_3mf,
    )
    plate_3mf = od / f"{asm.name}_print_plate.3mf"
    export_print_plate_3mf(asm, plate_3mf)

    # 3. Render 6-panel Multi-Color Assembly + Exploded Sheet
    sheet_png = od / "assembly_sheet.png"
    render_assembly_sheet(asm, sheet_png)

    # 4. Generate BOM Markdown & JSON
    bom_dict, bom_md_str = build_bom_data(asm, check)
    bom_md_path = od / "bom.md"
    bom_md_path.write_text(bom_md_str, encoding="utf-8")

    # 5. Export Interactive WebGL 3D Viewer with 💥 Explode Slider + 🎯 Picker + 📏 3D Caliper!
    all_anchors: list[dict[str, Any]] = []
    for n in asm.nodes:
        if n.registry:
            for a in n.registry.to_dict().get("anchors", []):
                all_anchors.append({**a, "id": f"{n.part_id}:{a['id']}"})

    viewer_items = [
        {
            "id": n.part_id,
            "label": f"[{i+1}] {n.label}",
            "shape": n.shape,
            "rgb": n.rgb,
            "opacity": 0.42 if n.is_reference else 1.0,
            "explode_vec": ev,
        }
        for i, (n, ev) in enumerate(zip(asm.nodes, exp_vecs))
    ]
    viewer_path = od / "viewer.html"
    export_viewer_html(
        target=None,
        out_html=viewer_path,
        title=f"{asm.name} (总装)",
        mode=asm.mode,
        params={"组件数量": len(asm.nodes), "预估总克重": f"{bom_dict['total_estimated_print_weight_g']} g"},
        anchors=all_anchors[:18],
        assembly_items=viewer_items,
    )

    manifest = {
        "assembly_name": asm.name,
        "mode": asm.mode,
        "verification": check,
        "bom": bom_dict,
        "files": {
            "assembly_step": str(step_path) if step_path else None,
            "assembly_3mf": str(asm_3mf),
            "print_plate_3mf": str(plate_3mf),
            "assembly_sheet_png": str(sheet_png),
            "bom_md": str(bom_md_path),
            "viewer_html": str(viewer_path),
        },
    }
    return manifest
