"""Unified CLI (`cad-cli`) for CAD Studio.

Subcommands:
  - cad-cli init-part <name> [--template basic|enclosure|extrusion-bracket|din-clip|gear] [--mode fdm|cnc|exact] [--ref <file>]
  - cad-cli build <part.py> [--mode fdm|cnc|exact] [--dxf [face]] [--no-html] [--no-diff]
  - cad-cli export-2d <file.step|file.stl|part.py> [--face bottom|top|front|back|left|right] [--plane XY|XZ|YZ] [--offset <mm>] [-o <out_dir>]
  - cad-cli inspect <file.step|file.stl> [-o <out_dir>] [--preview]
  - cad-cli diff <old_file> <new_file> [-o <diff_sheet.png>]
  - cad-cli render <file.step|file.stl> [-o <preview_sheet.png>] [--html <viewer.html>]
  - cad-cli open-slicer <file.step|file.stl>
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

from .assembly import run_assembly_build
from .export_2d import export_2d_profile
from .inspector import format_inspection_markdown, inspect_cad_file
from .printability import apply_optimal_orientation, export_3mf, rank_six_orientations, render_printability_sheet
from .renderer import compute_volumetric_diff, render_diff_sheet, render_preview_sheet
from .verify import verify_part_geometry
from .viewer_html import export_viewer_html


def _portable_path(p: str | Path | None) -> str | None:
    """Format paths relative to CWD when inside workspace so committed JSON manifests stay portable."""
    if p is None:
        return None
    pp = Path(p).resolve()
    try:
        return str(pp.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(pp)


TEMPLATES_SCAFFOLD: dict[str, str] = {
    "basic": '''"""Parametric structural part: __NAME__

Orientation Convention (CAD Studio Standard):
  前 Front = -Y | 后 Back = +Y | 左 Left = -X | 右 Right = +X | 上 Top = +Z | 下 Bottom = -Z (Z=0 热床)
"""

from __future__ import annotations

from dataclasses import dataclass
from build123d import Align, Axis, Box, Part, Pos

from cad_kit import (
    FeatureRegistry,
    ManufacturingMode,
    ToleranceProfile,
    make_counterbore_cutter,
    safe_chamfer,
    safe_fillet,
)


@dataclass
class Params:
    base_len_x: float = 60.0
    base_wid_y: float = 40.0
    base_thick_z: float = 5.0
    corner_fillet: float = 4.0
    hole_span_x: float = 46.0
    hole_span_y: float = 26.0
    mount_screw: str = "M3"


def build(
    params: Params | None = None,
    mode: ManufacturingMode = "__MODE__",
) -> tuple[Part, FeatureRegistry, Params]:
    p = params or Params()
    tol = ToleranceProfile.for_mode(mode)
    reg = FeatureRegistry(part_name="__NAME__", mode=mode, description="__NAME__ 基础结构安装板")

    part = Box(p.base_len_x, p.base_wid_y, p.base_thick_z, align=(Align.CENTER, Align.CENTER, Align.MIN))
    part = safe_fillet(part, part.edges().filter_by(Axis.Z), p.corner_fillet, reg, "四角圆角")

    if tol.bed_chamfer > 0:
        bottom_face = part.faces().sort_by(Axis.Z)[0]
        part = safe_chamfer(part, bottom_face.outer_wire().edges(), tol.bed_chamfer, reg, "底边防象脚倒角")

    cb_cutter = make_counterbore_cutter(tol, size=p.mount_screw, depth=p.base_thick_z + 5.0)
    hx, hy = p.hole_span_x / 2.0, p.hole_span_y / 2.0
    thru_d, cb_d, _ = tol.counterbore_dims(p.mount_screw)
    for hid, pos, desc in [
        ("H1", (-hx, -hy, p.base_thick_z), "左前安装孔"),
        ("H2", (+hx, -hy, p.base_thick_z), "右前安装孔"),
        ("H3", (-hx, +hy, p.base_thick_z), "左后安装孔"),
        ("H4", (+hx, +hy, p.base_thick_z), "右后安装孔"),
    ]:
        reg.mark(hid, pos, desc, kind="hole", spec=f"{p.mount_screw}沉头 Ø{thru_d}/Ø{cb_d}", normal=(0, 0, 1))
        part = part - Pos(*pos) * cb_cutter

    return part, reg, p


if __name__ == "__main__":
    from cad_kit.cli import run_part_build
    run_part_build(__file__)
''',
    "enclosure": '''"""Parametric Split Electronics Enclosure: __NAME__ (上下盖子母止口分体电子盒)"""

from __future__ import annotations

from dataclasses import dataclass
from build123d import Part

from cad_kit import (
    EnclosureLayout,
    EnclosureParams,
    FeatureRegistry,
    ManufacturingMode,
    ToleranceProfile,
    cut_panel_port,
    make_split_enclosure,
)


@dataclass
class Params:
    outer_x: float = 86.0
    outer_y: float = 58.0
    bottom_h: float = 22.0
    lid_h: float = 8.0
    wall_t: float = 2.8
    corner_r: float = 5.0
    flange_ears: bool = True
    vent_slots: int = 5
    usb_c_z: float = 11.0
    layout: EnclosureLayout = "print_plate"  # 'print_plate' | 'assembled' | 'bottom' | 'lid'


def build(
    params: Params | None = None,
    mode: ManufacturingMode = "__MODE__",
) -> tuple[Part, FeatureRegistry, Params]:
    p = params or Params()
    tol = ToleranceProfile.for_mode(mode)
    reg = FeatureRegistry(part_name="__NAME__", mode=mode, description="__NAME__ 上下分体电子控制盒")

    enc_cfg = EnclosureParams(
        outer_x=p.outer_x,
        outer_y=p.outer_y,
        bottom_h=p.bottom_h,
        lid_h=p.lid_h,
        wall_t=p.wall_t,
        corner_r=p.corner_r,
        flange_ears=p.flange_ears,
        vent_slots=p.vent_slots,
        pcb_standoffs=[(-28.0, -18.0), (28.0, -18.0), (-28.0, 18.0), (28.0, 18.0)],
        pcb_standoff_h=5.0,
    )
    part = make_split_enclosure(enc_cfg, tol=tol, reg=reg, layout=p.layout)

    # 在下壳正面 (-Y = -outer_y/2) 开带插头避让沉槽的 USB-C 接口
    if p.layout != "lid":
        part = cut_panel_port(
            part,
            wall="-Y",
            port_type="usb_c",
            offset_along_wall=0.0,
            center_z=p.usb_c_z,
            wall_coord=-p.outer_y * 0.5,
            tol=tol,
            reg=reg,
            anchor_id="USB_C",
            desc="正面Type-C接口",
        )
    return part, reg, p


if __name__ == "__main__":
    from cad_kit.cli import run_part_build
    run_part_build(__file__)
''',
    "extrusion-bracket": '''"""2020/3030/4040 Aluminum Extrusion Corner Gusset Bracket: __NAME__"""

from __future__ import annotations

from dataclasses import dataclass
from build123d import Part

from cad_kit import (
    ExtrusionSeries,
    FeatureRegistry,
    ManufacturingMode,
    ToleranceProfile,
    make_extrusion_bracket,
)


@dataclass
class Params:
    series: ExtrusionSeries = "2020"
    arm_len: float = 44.0
    wall_t: float = 5.0
    gusset_t: float = 3.8
    with_tongues: bool = True


def build(
    params: Params | None = None,
    mode: ManufacturingMode = "__MODE__",
) -> tuple[Part, FeatureRegistry, Params]:
    p = params or Params()
    tol = ToleranceProfile.for_mode(mode)
    reg = FeatureRegistry(part_name="__NAME__", mode=mode, description=f"{p.series} 欧标铝型材防转直角加强角码")
    part = make_extrusion_bracket(
        series=p.series,
        arm_len=p.arm_len,
        wall_t=p.wall_t,
        gusset_t=p.gusset_t,
        with_tongues=p.with_tongues,
        tol=tol,
        reg=reg,
    )
    return part, reg, p
''',
    "din-clip": '''"""35mm Standard TS35 DIN Rail Snap-On Clip: __NAME__"""

from __future__ import annotations

from dataclasses import dataclass
from build123d import Part

from cad_kit import FeatureRegistry, ManufacturingMode, ToleranceProfile, make_din_rail_clip


@dataclass
class Params:
    width_x: float = 18.0
    top_deck_y: float = 48.0
    deck_thick_z: float = 4.5
    mount_hole_span_y: float = 30.0
    mount_screw: str = "M3"


def build(
    params: Params | None = None,
    mode: ManufacturingMode = "__MODE__",
) -> tuple[Part, FeatureRegistry, Params]:
    p = params or Params()
    tol = ToleranceProfile.for_mode(mode)
    reg = FeatureRegistry(part_name="__NAME__", mode=mode, description="35mm 欧标 DIN 导轨快拆弹性卡扣底座")
    part = make_din_rail_clip(
        width_x=p.width_x,
        top_deck_y=p.top_deck_y,
        deck_thick_z=p.deck_thick_z,
        mount_hole_span_y=p.mount_hole_span_y,
        mount_screw=p.mount_screw,
        tol=tol,
        reg=reg,
    )
    return part, reg, p
''',
    "gear": '''"""Parametric Involute Spur Gear with Hub & D-Shaft Bore: __NAME__"""

from __future__ import annotations

from dataclasses import dataclass
from build123d import Part

from cad_kit import FeatureRegistry, ManufacturingMode, ToleranceProfile, make_spur_gear


@dataclass
class Params:
    module: float = 1.5
    teeth: int = 24
    thickness: float = 6.0
    bore_d: float = 5.0
    d_shaft_flat: float = 4.5
    hub_d: float = 16.0
    hub_h: float = 7.0
    set_screw: str = "M3"


def build(
    params: Params | None = None,
    mode: ManufacturingMode = "__MODE__",
) -> tuple[Part, FeatureRegistry, Params]:
    p = params or Params()
    tol = ToleranceProfile.for_mode(mode)
    reg = FeatureRegistry(part_name="__NAME__", mode=mode, description=f"渐开线正齿轮 m={p.module} z={p.teeth}")
    part = make_spur_gear(
        module=p.module,
        teeth=p.teeth,
        thickness=p.thickness,
        bore_d=p.bore_d,
        d_shaft_flat=p.d_shaft_flat,
        hub_d=p.hub_d,
        hub_h=p.hub_h,
        set_screw=p.set_screw,
        tol=tol,
        reg=reg,
    )
    return part, reg, p
''',
}


def _load_part_module(script_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"part_mod_{script_path.stem}", str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load Python module from {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_part_build(
    script_path: str | Path,
    mode: str | None = None,
    out_dir: str | Path | None = None,
    gen_html: bool = True,
    gen_diff: bool = True,
    save_snapshot: bool = True,
    expected_solids: int | None = None,
    dxf_face: str | None = None,
    print_check: bool = False,
    auto_orient: bool = False,
) -> dict[str, Any]:
    """Execute a part.py script, export STEP/STL/3MF (+ optional DXF/DFM), verify geometry, and render views."""
    t0 = time.perf_counter()
    sp = Path(script_path).expanduser().resolve()
    if not sp.exists():
        raise FileNotFoundError(f"Part script not found: {sp}")

    od = Path(out_dir).expanduser().resolve() if out_dir else (sp.parent / "out")
    od.mkdir(parents=True, exist_ok=True)

    mod = _load_part_module(sp)
    if not hasattr(mod, "build"):
        raise AttributeError(f"{sp} must define a `build(params=None, mode=...)` function.")

    build_kwargs: dict[str, Any] = {}
    if mode is not None:
        build_kwargs["mode"] = mode

    res = mod.build(**build_kwargs)
    if isinstance(res, tuple) and len(res) == 3:
        part, reg, params_obj = res
    elif isinstance(res, tuple) and len(res) == 2:
        part, reg = res
        params_obj = None
    else:
        part, reg, params_obj = res, None, None

    active_mode = (mode or (reg.mode if reg else "fdm")).lower()
    part_name = reg.part_name if reg else sp.stem
    params_dict = asdict(params_obj) if (params_obj is not None and is_dataclass(params_obj)) else {}
    reg_dict = reg.to_dict() if reg else {}
    if "reference_files" in reg_dict:
        reg_dict["reference_files"] = [_portable_path(rf) for rf in reg_dict["reference_files"]]
    anchors_list = reg_dict.get("anchors", [])
    ref_files = reg.reference_files if reg else []
    exp_solids = expected_solids if expected_solids is not None else (reg.expected_solids if reg else 1)

    orient_applied = None
    if auto_orient:
        ranks = rank_six_orientations(part)
        best = ranks[0]
        if best["orientation"] != "-Z_bed":
            part = apply_optimal_orientation(part, best["rot_xyz_deg"])
            orient_applied = best
            # Clear static anchors / refs when part is globally re-oriented to avoid coordinate mismatch
            anchors_list = []
            ref_files = []

    is_brep = hasattr(part, "solids") and hasattr(part, "tessellate")
    step_path: Path | None = od / f"{part_name}.step"
    stl_path = od / f"{part_name}.stl"
    threemf_path = od / f"{part_name}.3mf"
    prev_dir = od / ".prev"
    prev_stl = prev_dir / f"{part_name}_prev.stl"

    staging_old = prev_dir / f"{part_name}_staging.stl"
    if stl_path.exists():
        prev_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stl_path, staging_old)

    if is_brep:
        import build123d as bd
        assert step_path is not None
        bd.export_step(part, str(step_path))
        bd.export_stl(part, str(stl_path), tolerance=0.02, angular_tolerance=0.08)
    else:
        part.export(str(stl_path))
        step_path = None

    export_3mf([{"name": part_name, "shape": part, "color": "#3b82f6"}], threemf_path)

    check = verify_part_geometry(part, expected_solids=exp_solids, ref_files=ref_files)
    for item in check.get("interferences", []):
        if "ref_file" in item:
            item["ref_file"] = _portable_path(item["ref_file"])

    preview_png = od / "preview_sheet.png"
    render_preview_sheet(
        target=part,
        out_png=preview_png,
        title=part_name,
        mode=active_mode,
        anchors=anchors_list,
        ref_files=ref_files,
    )

    diff_png: Path | None = None
    diff_stats: dict[str, Any] | None = None
    diff_meshes: dict[str, Any] | None = None
    if staging_old.exists():
        cand_diff = compute_volumetric_diff(staging_old, stl_path)
        if cand_diff["added_volume_mm3"] > 0.01 or cand_diff["removed_volume_mm3"] > 0.01:
            if save_snapshot:
                shutil.copy2(staging_old, prev_stl)
        staging_old.unlink(missing_ok=True)

    if gen_diff and prev_stl.exists():
        raw_diff = compute_volumetric_diff(prev_stl, stl_path)
        if raw_diff["added_volume_mm3"] > 0.01 or raw_diff["removed_volume_mm3"] > 0.01:
            diff_png = od / "diff_sheet.png"
            render_diff_sheet(prev_stl, stl_path, diff_png, title=f"{part_name} 修改前后对比")
            diff_stats = {k: v for k, v in raw_diff.items() if not k.endswith("_mesh")}
            diff_meshes = raw_diff

    viewer_path: Path | None = None
    if gen_html:
        viewer_path = od / "viewer.html"
        export_viewer_html(
            target=part,
            out_html=viewer_path,
            title=part_name,
            mode=active_mode,
            params=params_dict,
            anchors=anchors_list,
            ref_files=ref_files,
            diff_meshes=diff_meshes,
        )

    export2d_info: dict[str, Any] | None = None
    if dxf_face:
        face_arg = dxf_face.lower() if dxf_face.lower() in ("bottom", "top", "front", "back", "left", "right") else None
        plane_arg = dxf_face.upper() if dxf_face.upper() in ("XY", "XZ", "YZ") else "XY"
        export2d_info = export_2d_profile(
            part,
            out_dir=od,
            name=part_name,
            plane=plane_arg,  # type: ignore[arg-type]
            face=face_arg,    # type: ignore[arg-type]
        )
        for k2 in ("dxf", "svg", "drawing_png"):
            if k2 in export2d_info:
                export2d_info[k2] = _portable_path(export2d_info[k2])

    dfm_png: Path | None = None
    dfm_report: dict[str, Any] | None = None
    if print_check:
        dfm_png = od / "printability_sheet.png"
        _, dfm_report = render_printability_sheet(part, dfm_png, title=part_name)
        if orient_applied is not None:
            dfm_report["applied_auto_orientation"] = orient_applied

    elapsed = round(time.perf_counter() - t0, 2)
    manifest = {
        "part_name": part_name,
        "mode": active_mode,
        "elapsed_sec": elapsed,
        "files": {
            "script": _portable_path(sp),
            "step": _portable_path(step_path),
            "stl": _portable_path(stl_path),
            "threemf": _portable_path(threemf_path),
            "preview_sheet_png": _portable_path(preview_png),
            "diff_sheet_png": _portable_path(diff_png),
            "printability_sheet_png": _portable_path(dfm_png),
            "viewer_html": _portable_path(viewer_path),
            "dxf": export2d_info["dxf"] if export2d_info else None,
            "svg": export2d_info["svg"] if export2d_info else None,
            "drawing_2d_png": export2d_info["drawing_png"] if export2d_info else None,
        },
        "params": params_dict,
        "registry": reg_dict,
        "verification": check,
        "diff": diff_stats,
        "printability": dfm_report,
        "export_2d": export2d_info,
    }
    manifest_path = od / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["files"]["manifest_json"] = _portable_path(manifest_path)
    return manifest


def _cmd_init_part(args: argparse.Namespace) -> int:
    root = Path.cwd()
    part_dir = root / "projects" / args.name
    (part_dir / "refs").mkdir(parents=True, exist_ok=True)
    (part_dir / "out").mkdir(parents=True, exist_ok=True)
    script_file = part_dir / "part.py"
    if script_file.exists() and not args.force:
        print(f"[!] {script_file} already exists (use --force to overwrite).")
        return 1
    tpl = TEMPLATES_SCAFFOLD[args.template]
    code = tpl.replace("__NAME__", args.name).replace("__MODE__", args.mode)
    script_file.write_text(code, encoding="utf-8")
    if args.ref:
        ref_src = Path(args.ref).expanduser().resolve()
        if ref_src.exists():
            dst = part_dir / "refs" / ref_src.name
            shutil.copy2(ref_src, dst)
            print(f"[+] Copied reference file -> {dst}")
    print(f"[+] Initialized part project [{args.template}] at: {part_dir}")
    print(f"    Edit:  {script_file}")
    print(f"    Build: uv run cad-cli build {script_file}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    sp = Path(args.script).expanduser().resolve()
    mod = _load_part_module(sp)
    if hasattr(mod, "build_assembly") and not hasattr(mod, "build"):
        return _run_assembly_cli(sp, args.out_dir)

    manifest = run_part_build(
        script_path=sp,
        mode=args.mode,
        out_dir=args.out_dir,
        gen_html=not args.no_html,
        gen_diff=not args.no_diff,
        save_snapshot=not args.no_snapshot,
        expected_solids=args.expected_solids,
        dxf_face=args.dxf,
        print_check=args.print_check,
        auto_orient=args.auto_orient,
    )
    v = manifest["verification"]
    bb = v["bbox"]
    print(f"[✓] Built '{manifest['part_name']}' [mode={manifest['mode'].upper()}] in {manifest['elapsed_sec']}s")
    print(f"    BBox Size : {bb['size_mm'][0]} × {bb['size_mm'][1]} × {bb['size_mm'][2]} mm")
    print(f"    BBox Range: X{bb['min_mm'][0]}..{bb['max_mm'][0]} | Y{bb['min_mm'][1]}..{bb['max_mm'][1]} | Z{bb['min_mm'][2]}..{bb['max_mm'][2]}")
    print(f"    Volume    : {v['volume_mm3']} mm³  |  Solids: {v['solids_count']}  |  Watertight: {v['is_watertight']}")
    for k, fp in manifest["files"].items():
        if fp:
            print(f"    {k:22s}: {fp}")
    if manifest.get("diff"):
        d = manifest["diff"]
        print(f"    [Diff] +Added {d['added_volume_mm3']} mm³  |  -Removed {d['removed_volume_mm3']} mm³")
    if manifest.get("printability"):
        pr = manifest["printability"]
        best = pr["best_orientation"]
        print(f"    [DFM]  Score {pr['score']}/100  |  Bed {pr['bed_contact_area_mm2']} mm²  |  Islands {pr['floating_islands_count']}  |  Best: {best['desc']} ({best['score']}/100)")
    for w in v.get("warnings", []):
        print(f"    [WARN] {w}")
    for err in v.get("issues", []):
        print(f"    [ISSUE] {err}")
    return 0 if v["ok"] else 2


def _cmd_export_2d(args: argparse.Namespace) -> int:
    src = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (src.parent / "out" if src.suffix == ".py" else src.parent)
    target: Any = src
    name = src.stem
    if src.suffix.lower() == ".py":
        mod = _load_part_module(src)
        res = mod.build()
        target = res[0] if isinstance(res, tuple) else res
        if isinstance(res, tuple) and len(res) >= 2 and res[1] is not None:
            name = res[1].part_name
    info = export_2d_profile(
        target=target,
        out_dir=out_dir,
        name=name,
        plane=args.plane,
        face=args.face,
        offset=args.offset,
    )
    print(f"[✓] Exported 1:1 2D Laser/CNC files ({info['plane']} @ {info['offset_mm']} mm):")
    print(f"    Outer Size  : {info['outer_size_mm'][0]} × {info['outer_size_mm'][1]} mm  (Analytic B-Rep: {info['analytic_brep']})")
    print(f"    DXF File    : {info['dxf']}")
    print(f"    SVG File    : {info['svg']}")
    print(f"    2D Blueprint: {info['drawing_png']}")
    print(f"    Holes Count : {len(info['holes'])}")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    src = Path(args.file).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (src.parent / f"inspect_{src.stem}")
    info = inspect_cad_file(src, out_dir=out_dir)
    if args.preview:
        prev_png = out_dir / f"{src.stem}_preview_sheet.png"
        render_preview_sheet(src, prev_png, title=f"参考件: {src.name}", mode="exact")
        info["preview_sheet_png"] = _portable_path(prev_png)
    print(format_inspection_markdown(info))
    print(f"[✓] Saved slices PNG : {info.get('slices_png')}")
    if info.get("preview_sheet_png"):
        print(f"[✓] Saved 10-view PNG: {info['preview_sheet_png']}")
    print(f"[✓] Saved JSON report: {info.get('report_json')}")
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    out_png = Path(args.output).expanduser().resolve()
    path, stats = render_diff_sheet(args.old_file, args.new_file, out_png, title=args.title)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[✓] Diff sheet saved to: {path}")
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    src = Path(args.file).expanduser().resolve()
    out_png = Path(args.output).expanduser().resolve() if args.output else src.with_name(f"{src.stem}_preview_sheet.png")
    render_preview_sheet(src, out_png, title=src.name, mode="exact")
    print(f"[✓] 10-view sheet saved to: {out_png}")
    if args.html:
        html_path = Path(args.html).expanduser().resolve()
        export_viewer_html(src, html_path, title=src.name, mode="exact")
        print(f"[✓] Interactive 3D viewer saved to: {html_path}")
    return 0


ASSEMBLY_SCAFFOLD = '''"""Multi-Part CAD Assembly: __NAME__"""

from __future__ import annotations

from build123d import Align, Box, Pos, Rot

from cad_kit import (
    Assembly,
    EnclosureParams,
    FeatureRegistry,
    ToleranceProfile,
    cut_panel_port,
    make_din_rail_clip,
    make_split_enclosure,
)


def build_assembly(mode: str = "__MODE__") -> Assembly:
    tol = ToleranceProfile.for_mode(mode)  # type: ignore[arg-type]
    asm = Assembly(
        name="__NAME__",
        description="DIN导轨安装式工业传感器分体控制盒总装 (含下壳、上盖、导轨快拆卡座与内部PCB模块)",
        mode=mode,  # type: ignore[arg-type]
    )

    enc_cfg = EnclosureParams(
        outer_x=78.0,
        outer_y=54.0,
        bottom_h=20.0,
        lid_h=7.5,
        wall_t=2.8,
        corner_r=4.5,
        flange_ears=False,
        vent_slots=4,
        pcb_standoffs=[(-24.0, -16.0), (24.0, -16.0), (-24.0, 16.0), (24.0, 16.0)],
        pcb_standoff_h=4.5,
        pcb_screw="M2.5",
    )

    # 1. 下壳 (Z = 0..22.2mm)
    reg_bot = FeatureRegistry("bottom_box", mode=mode)  # type: ignore[arg-type]
    bot = make_split_enclosure(enc_cfg, tol=tol, reg=reg_bot, layout="bottom")
    bot = cut_panel_port(bot, wall="-Y", port_type="usb_c", center_z=10.5, wall_coord=-27.0, tol=tol, reg=reg_bot)
    asm.add_part("bottom_box", bot, label="控制盒下壳", explode_vec=(0, 0, 0), registry=reg_bot)

    # 2. 内部定制传感器 PCB 模块 (参考件: 四角做避让圆弧切角以绕开外壳四角螺丝柱，带4个M2.5安装孔!)
    from build123d import Cylinder
    pcb = Pos(0, 0, 6.9 + 0.05) * Box(62.0, 40.0, 1.6, align=(Align.CENTER, Align.CENTER, Align.MIN))
    for cx, cy in [(-31.3, -20.3), (31.3, -20.3), (-31.3, 20.3), (31.3, 20.3)]:
        pcb = pcb - Pos(cx, cy, 5.0) * Cylinder(radius=5.2, height=6.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
    for px, py in [(-24.0, -16.0), (24.0, -16.0), (-24.0, 16.0), (24.0, 16.0)]:
        pcb = pcb - Pos(px, py, 5.0) * Cylinder(radius=1.35, height=6.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
    asm.add_part("ctrl_pcb", pcb, label="主控传感器PCB板", is_reference=True, explode_vec=(0, 0, 22.0))

    # 3. 分体上盖 (装配闭合态扣在 Z = 20.0..27.5mm；打印排盘时自动翻平免支撑!)
    lid_flat = make_split_enclosure(enc_cfg, tol=tol, layout="lid")
    lid_assembled = Pos(0, 0, enc_cfg.bottom_h + enc_cfg.lid_h) * Rot(180, 0, 0) * lid_flat
    asm.add_part("top_lid", lid_assembled, label="防尘子母扣上盖", explode_vec=(0, 0, 46.0))

    # 4. 底部 35mm DIN 导轨快拆卡扣座 (贴合在下壳底面 Z = 0 下方)
    reg_din = FeatureRegistry("din_clip", mode=mode)  # type: ignore[arg-type]
    din = Pos(0, 0, -4.5 - 0.02) * make_din_rail_clip(width_x=22.0, top_deck_y=50.0, deck_thick_z=4.5, tol=tol, reg=reg_din)
    asm.add_part("din_clip", din, label="35mm导轨弹性卡扣", explode_vec=(0, 0, -28.0), registry=reg_din)

    # 5. BOM 紧固件清单
    asm.add_hardware("M3×8 内六角圆柱头螺丝 (ISO 4762)", qty=4, category="紧固件", note="上盖锁紧至下壳四角")
    asm.add_hardware("M3×D4.5×L5 滚花热熔铜螺母", qty=6, category="嵌件", note="下壳四角 ×4 + 导轨卡扣 ×2")
    asm.add_hardware("M2.5×5 圆头自攻/机丝螺丝", qty=4, category="紧固件", note="固定内部传感器PCB板")
    asm.add_hardware("M3×6 平头沉头螺丝", qty=2, category="紧固件", note="下壳底部固定至DIN导轨卡座")

    return asm


if __name__ == "__main__":
    from pathlib import Path
    from cad_kit.assembly import run_assembly_build
    run_assembly_build(build_assembly(), Path(__file__).parent / "out")
'''


def _run_assembly_cli(sp: Path, out_dir: str | None) -> int:
    t0 = time.perf_counter()
    mod = _load_part_module(sp)
    if not hasattr(mod, "build_assembly"):
        raise AttributeError(f"{sp} must define `build_assembly()` returning an `Assembly`.")
    asm = mod.build_assembly()
    od = Path(out_dir).expanduser().resolve() if out_dir else (sp.parent / "out")
    manifest = run_assembly_build(asm, od)
    for k, v in list(manifest["files"].items()):
        manifest["files"][k] = _portable_path(v)
    mf_path = od / "assembly_manifest.json"
    mf_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["files"]["manifest_json"] = _portable_path(mf_path)

    chk = manifest["verification"]
    elapsed = round(time.perf_counter() - t0, 2)
    print(f"[✓] Built Assembly '{asm.name}' ({chk['parts_count']} parts) in {elapsed}s")
    print(f"    Envelope  : {chk['bbox']['size_mm'][0]} × {chk['bbox']['size_mm'][1]} × {chk['bbox']['size_mm'][2]} mm")
    print(f"    Est Weight: ~{manifest['bom']['total_estimated_print_weight_g']} g (printable parts)")
    for k, fp in manifest["files"].items():
        if fp:
            print(f"    {k:22s}: {fp}")
    for err in chk.get("issues", []):
        print(f"    [ISSUE] {err}")
    return 0 if chk["ok"] else 2


def _cmd_init_assembly(args: argparse.Namespace) -> int:
    asm_dir = Path.cwd() / "projects" / args.name
    (asm_dir / "out").mkdir(parents=True, exist_ok=True)
    script_file = asm_dir / "assembly.py"
    if script_file.exists() and not args.force:
        print(f"[!] {script_file} already exists (use --force to overwrite).")
        return 1
    script_file.write_text(
        ASSEMBLY_SCAFFOLD.replace("__NAME__", args.name).replace("__MODE__", args.mode),
        encoding="utf-8",
    )
    print(f"[+] Initialized multi-part assembly project at: {asm_dir}")
    print(f"    Build: uv run cad-cli build-assembly {script_file}")
    return 0


def _cmd_build_assembly(args: argparse.Namespace) -> int:
    return _run_assembly_cli(Path(args.script).expanduser().resolve(), args.out_dir)


def _cmd_print_check(args: argparse.Namespace) -> int:
    src = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (src.parent / "out" if src.suffix == ".py" else src.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    target: Any = src
    name = src.stem
    if src.suffix.lower() == ".py":
        mod = _load_part_module(src)
        res = mod.build()
        target = res[0] if isinstance(res, tuple) else res
        if isinstance(res, tuple) and len(res) >= 2 and res[1] is not None:
            name = res[1].part_name

    png_path = out_dir / f"{name}_printability_sheet.png"
    _, report = render_printability_sheet(target, png_path, title=name)
    best = report["best_orientation"]

    print(f"[✓] 3D Printability DFM Report for '{name}':")
    print(f"    Current Score : {report['score']}/100  (Needs Support: {report['needs_support']}, Needs Brim: {report['needs_brim']})")
    print(f"    Bed Contact   : {report['bed_contact_area_mm2']} mm²  |  Severe Overhang (>55°): {report['severe_overhang_mm2']} mm²")
    print(f"    Mid-Air Island: {report['floating_islands_count']} 处")
    print(f"    Best Orient   : {best['desc']} -> Score {best['score']}/100 (rot={best['rot_xyz_deg']})")
    for r in report["recommendations"]:
        print(f"    • {r}")
    print(f"    Heatmap Sheet : {_portable_path(png_path)}")

    if args.auto_orient:
        oriented = apply_optimal_orientation(target, best["rot_xyz_deg"])
        out_3mf = out_dir / f"{name}_oriented.3mf"
        export_3mf([{"name": f"{name}_oriented", "shape": oriented, "color": "#10b981"}], out_3mf)
        if hasattr(oriented, "solids"):
            import build123d as bd
            out_step = out_dir / f"{name}_oriented.step"
            bd.export_step(oriented, str(out_step))
            print(f"    Oriented STEP : {_portable_path(out_step)}")
        print(f"    Oriented 3MF  : {_portable_path(out_3mf)}")
    return 0


def _cmd_open_slicer(args: argparse.Namespace) -> int:
    target = Path(args.file).expanduser().resolve()
    if not target.exists():
        print(f"[!] File not found: {target}")
        return 1
    candidates: list[Path] = []
    env_bin = os.environ.get("BAMBU_STUDIO_BIN")
    if env_bin:
        candidates.append(Path(env_bin).expanduser())
    for cmd_name in ("bambu-studio", "BambuStudio"):
        found = shutil.which(cmd_name)
        if found:
            candidates.append(Path(found))
    sibling_dir = Path(__file__).resolve().parents[2] / "3d-mark"
    if sibling_dir.is_dir():
        candidates.extend(sorted(sibling_dir.glob("启动BambuStudio.sh")))
        candidates.extend(sorted(sibling_dir.glob("BambuStudio*.AppImage"), reverse=True))
    for c in candidates:
        if c.exists():
            subprocess.Popen([str(c), str(target)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[✓] Launched BambuStudio with: {target}")
            return 0
    print("[!] BambuStudio launcher not found (set BAMBU_STUDIO_BIN=/path/to/BambuStudio.AppImage).")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(prog="cad-cli", description="CAD Studio Unified Agentic Workflow CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init-part", help="Create a new parametric part workspace under projects/<name>")
    p_init.add_argument("name", help="Part folder/identifier name")
    p_init.add_argument(
        "--template",
        choices=list(TEMPLATES_SCAFFOLD.keys()),
        default="basic",
        help="Starter structural template (basic, enclosure, extrusion-bracket, din-clip, gear)",
    )
    p_init.add_argument("--mode", choices=["fdm", "cnc", "exact"], default="fdm")
    p_init.add_argument("--ref", help="Optional existing STEP/STL file to copy into refs/")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing part.py")
    p_init.set_defaults(func=_cmd_init_part)

    p_build = sub.add_parser("build", help="Build part.py -> STEP + STL + Verify + 10-View PNG + Diff + 3D HTML")
    p_build.add_argument("script", help="Path to part.py")
    p_build.add_argument("--mode", choices=["fdm", "cnc", "exact"], default=None, help="Override manufacturing mode")
    p_build.add_argument("-o", "--out-dir", default=None, help="Custom output directory")
    p_build.add_argument(
        "--dxf",
        nargs="?",
        const="bottom",
        default=None,
        help="Also export 1:1 2D DXF/SVG (optional face/plane: bottom, top, front, back, left, right, XY, XZ, YZ)",
    )
    p_build.add_argument("--print-check", action="store_true", help="Also run FDM overhang/island DFM check & heatmap")
    p_build.add_argument("--auto-orient", action="store_true", help="Auto-rotate part to #1 printability orientation")
    p_build.add_argument("--no-html", action="store_true", help="Skip generating viewer.html")
    p_build.add_argument("--no-diff", action="store_true", help="Skip generating diff_sheet.png")
    p_build.add_argument("--no-snapshot", action="store_true", help="Do not overwrite .prev baseline STL")
    p_build.add_argument("--expected-solids", type=int, default=None, help="Expected connected solid count (0 to disable)")
    p_build.set_defaults(func=_cmd_build)

    p_dfm = sub.add_parser("print-check", help="Run FDM overhang/island DFM scan, 6-axis ranking & optional auto-orient")
    p_dfm.add_argument("input", help="Input .step / .stl / part.py")
    p_dfm.add_argument("--auto-orient", action="store_true", help="Export optimally oriented .3mf/.step")
    p_dfm.add_argument("-o", "--out-dir", default=None, help="Output folder")
    p_dfm.set_defaults(func=_cmd_print_check)

    p_iasm = sub.add_parser("init-assembly", help="Create a multi-part assembly project under projects/<name>")
    p_iasm.add_argument("name", help="Assembly folder name")
    p_iasm.add_argument("--mode", choices=["fdm", "cnc", "exact"], default="fdm")
    p_iasm.add_argument("--force", action="store_true", help="Overwrite existing assembly.py")
    p_iasm.set_defaults(func=_cmd_init_assembly)

    p_basm = sub.add_parser("build-assembly", help="Build multi-part assembly -> N×N check + Exploded Sheet + BOM + .3mf + HTML")
    p_basm.add_argument("script", help="Path to assembly.py")
    p_basm.add_argument("-o", "--out-dir", default=None, help="Output folder")
    p_basm.set_defaults(func=_cmd_build_assembly)

    p_2d = sub.add_parser("export-2d", help="Export 1:1 laser/CNC DXF + SVG + Hole Drilling Sheet from STEP/STL/part.py")
    p_2d.add_argument("input", help="Input .step / .stl / part.py")
    p_2d.add_argument("--face", choices=["bottom", "top", "front", "back", "left", "right"], default=None)
    p_2d.add_argument("--plane", choices=["XY", "XZ", "YZ"], default="XY")
    p_2d.add_argument("--offset", type=float, default=None, help="Exact section coordinate (mm)")
    p_2d.add_argument("-o", "--out-dir", default=None, help="Output folder")
    p_2d.set_defaults(func=_cmd_export_2d)

    p_insp = sub.add_parser("inspect", help="Reverse-engineer dimensions, planes, holes & 2D slices of STEP/STL")
    p_insp.add_argument("file", help="Path to .step / .stp / .stl / .3mf file")
    p_insp.add_argument("-o", "--out-dir", default=None, help="Output folder for slices PNG & JSON/MD report")
    p_insp.add_argument("--preview", action="store_true", help="Also generate 10-view preview_sheet.png")
    p_insp.set_defaults(func=_cmd_inspect)

    p_diff = sub.add_parser("diff", help="Render 3D boolean volumetric diff (Added green / Removed red)")
    p_diff.add_argument("old_file", help="Old .step / .stl")
    p_diff.add_argument("new_file", help="New .step / .stl")
    p_diff.add_argument("-o", "--output", default="diff_sheet.png", help="Output PNG path")
    p_diff.add_argument("--title", default="修改前后体积对比 (Diff)")
    p_diff.set_defaults(func=_cmd_diff)

    p_rend = sub.add_parser("render", help="Render 10-view engineering sheet (and optional 3D HTML) for any STEP/STL")
    p_rend.add_argument("file", help="Input .step / .stl / .3mf")
    p_rend.add_argument("-o", "--output", default=None, help="Output preview PNG path")
    p_rend.add_argument("--html", default=None, help="Optional interactive viewer.html output path")
    p_rend.set_defaults(func=_cmd_render)

    p_slicer = sub.add_parser("open-slicer", help="Open STEP/STL directly in BambuStudio")
    p_slicer.add_argument("file", help="Path to .step or .stl")
    p_slicer.set_defaults(func=_cmd_open_slicer)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
