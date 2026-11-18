"""Unified CLI (`cad-cli`) for CAD Studio.

Subcommands:
  - cad-cli init-part <name> [--mode fdm|cnc|exact] [--ref <file>]
  - cad-cli build <part.py> [--mode fdm|cnc|exact] [--html] [--no-diff] [--no-snapshot]
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
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

from .inspector import format_inspection_markdown, inspect_cad_file
from .renderer import compute_volumetric_diff, render_diff_sheet, render_preview_sheet
from .verify import verify_part_geometry
from .viewer_html import export_viewer_html


PART_SCAFFOLD = '''"""Parametric structural part: __NAME__

Orientation Convention (CAD Studio Standard):
  前 Front = -Y | 后 Back = +Y | 左 Left = -X | 右 Right = +X | 上 Top = +Z | 下 Bottom = -Z (Z=0 热床)
"""

from __future__ import annotations

from dataclasses import dataclass
from build123d import Align, Box, Axis, Part, Pos, Rot

from cad_kit import (
    FeatureRegistry,
    ManufacturingMode,
    ToleranceProfile,
    make_counterbore_cutter,
    make_heatset_cutter,
    make_rib,
    make_slot_cutter,
    safe_chamfer,
    safe_fillet,
)


@dataclass
class Params:
    """All tunable dimensions in millimeters (mm) — avoid magic numbers in build()."""

    base_len_x: float = 60.0       # 左右总长 (X)
    base_wid_y: float = 40.0       # 前后总宽 (Y)
    base_thick_z: float = 5.0      # 底板厚度 (Z)
    corner_fillet: float = 4.0     # 四角外圆角半径
    hole_span_x: float = 46.0      # 左右安装孔中心距
    hole_span_y: float = 26.0      # 前后安装孔中心距
    mount_screw: str = "M3"        # 安装孔规格


def build(
    params: Params | None = None,
    mode: ManufacturingMode = "__MODE__",
) -> tuple[Part, FeatureRegistry, Params]:
    p = params or Params()
    tol = ToleranceProfile.for_mode(mode)
    reg = FeatureRegistry(
        part_name="__NAME__",
        mode=mode,
        description="__NAME__ 结构件",
    )

    # 1. 基础毛坯底板 (XY 居中于原点，底面贴合 Z=0)
    part = Box(
        p.base_len_x,
        p.base_wid_y,
        p.base_thick_z,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )

    # 2. 外轮廓竖直边倒圆角 (先圆角外轮廓，再打孔，更稳健)
    part = safe_fillet(part, part.edges().filter_by(Axis.Z), p.corner_fillet, reg, "四角圆角")

    # 3. 底面 (-Z) 防象脚/去毛刺微倒角
    if tol.bed_chamfer > 0:
        bottom_face = part.faces().sort_by(Axis.Z)[0]
        part = safe_chamfer(part, bottom_face.outer_wire().edges(), tol.bed_chamfer, reg, "底边防象脚倒角")

    # 4. 四角 M3 沉头安装孔 (注册语义锚点 H1..H4，自动投影到六视图与3D预览)
    cb_cutter = make_counterbore_cutter(tol, size=p.mount_screw, depth=p.base_thick_z + 5.0)
    hx, hy = p.hole_span_x / 2.0, p.hole_span_y / 2.0
    hole_defs = [
        ("H1", (-hx, -hy, p.base_thick_z), "左前安装孔"),
        ("H2", (+hx, -hy, p.base_thick_z), "右前安装孔"),
        ("H3", (-hx, +hy, p.base_thick_z), "左后安装孔"),
        ("H4", (+hx, +hy, p.base_thick_z), "右后安装孔"),
    ]
    thru_d, cb_d, _ = tol.counterbore_dims(p.mount_screw)
    for hid, pos, desc in hole_defs:
        reg.mark(hid, pos, desc, kind="hole", spec=f"{p.mount_screw}沉头 Ø{thru_d}/Ø{cb_d}", normal=(0, 0, 1))
        part = part - Pos(*pos) * cb_cutter

    return part, reg, p


if __name__ == "__main__":
    from cad_kit.cli import run_part_build
    run_part_build(__file__)
'''


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
    expected_solids: int = 1,
) -> dict[str, Any]:
    """Execute a part.py script, export STEP/STL, verify geometry/interference, and render views."""
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
    anchors_list = reg.to_dict()["anchors"] if reg else []
    ref_files = reg.reference_files if reg else []

    # Export STEP & STL
    is_brep = hasattr(part, "solids") and hasattr(part, "tessellate")
    step_path = od / f"{part_name}.step"
    stl_path = od / f"{part_name}.stl"
    prev_dir = od / ".prev"
    prev_stl = prev_dir / f"{part_name}_prev.stl"

    # Snapshot existing STL to a staging file BEFORE overwriting so identical rebuilds keep last real diff!
    staging_old = prev_dir / f"{part_name}_staging.stl"
    if stl_path.exists():
        prev_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stl_path, staging_old)

    if is_brep:
        import build123d as bd
        bd.export_step(part, str(step_path))
        bd.export_stl(part, str(stl_path), tolerance=0.02, angular_tolerance=0.08)
    else:
        # trimesh output from direct STL modification
        part.export(str(stl_path))
        step_path = None  # type: ignore[assignment]

    # Run geometry & interference verification
    check = verify_part_geometry(part, expected_solids=expected_solids, ref_files=ref_files)

    # Render 10-view Engineering Sheet
    preview_png = od / "preview_sheet.png"
    render_preview_sheet(
        target=part,
        out_png=preview_png,
        title=part_name,
        mode=active_mode,
        anchors=anchors_list,
        ref_files=ref_files,
    )

    # Determine whether geometry changed vs staging_old; if so, promote staging_old -> prev_stl
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

    # Export Interactive 3D WebGL Viewer HTML
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

    elapsed = round(time.perf_counter() - t0, 2)
    manifest = {
        "part_name": part_name,
        "mode": active_mode,
        "elapsed_sec": elapsed,
        "files": {
            "script": str(sp),
            "step": str(step_path) if step_path else None,
            "stl": str(stl_path),
            "preview_sheet_png": str(preview_png),
            "diff_sheet_png": str(diff_png) if diff_png else None,
            "viewer_html": str(viewer_path) if viewer_path else None,
        },
        "params": params_dict,
        "registry": reg.to_dict() if reg else {},
        "verification": check,
        "diff": diff_stats,
    }
    manifest_path = od / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["files"]["manifest_json"] = str(manifest_path)
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
    code = PART_SCAFFOLD.replace("__NAME__", args.name).replace("__MODE__", args.mode)
    script_file.write_text(code, encoding="utf-8")
    if args.ref:
        ref_src = Path(args.ref).expanduser().resolve()
        if ref_src.exists():
            dst = part_dir / "refs" / ref_src.name
            shutil.copy2(ref_src, dst)
            print(f"[+] Copied reference file -> {dst}")
    print(f"[+] Initialized part project at: {part_dir}")
    print(f"    Edit:  {script_file}")
    print(f"    Build: uv run cad-cli build {script_file}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    manifest = run_part_build(
        script_path=args.script,
        mode=args.mode,
        out_dir=args.out_dir,
        gen_html=not args.no_html,
        gen_diff=not args.no_diff,
        save_snapshot=not args.no_snapshot,
        expected_solids=args.expected_solids,
    )
    v = manifest["verification"]
    bb = v["bbox"]
    print(f"[✓] Built '{manifest['part_name']}' [mode={manifest['mode'].upper()}] in {manifest['elapsed_sec']}s")
    print(f"    BBox Size : {bb['size_mm'][0]} × {bb['size_mm'][1]} × {bb['size_mm'][2]} mm")
    print(f"    BBox Range: X{bb['min_mm'][0]}..{bb['max_mm'][0]} | Y{bb['min_mm'][1]}..{bb['max_mm'][1]} | Z{bb['min_mm'][2]}..{bb['max_mm'][2]}")
    print(f"    Volume    : {v['volume_mm3']} mm³  |  Solids: {v['solids_count']}  |  Watertight: {v['is_watertight']}")
    for k, fp in manifest["files"].items():
        if fp:
            print(f"    {k:18s}: {fp}")
    if manifest.get("diff"):
        d = manifest["diff"]
        print(f"    [Diff] +Added {d['added_volume_mm3']} mm³  |  -Removed {d['removed_volume_mm3']} mm³")
    for w in v.get("warnings", []):
        print(f"    [WARN] {w}")
    for err in v.get("issues", []):
        print(f"    [ISSUE] {err}")
    return 0 if v["ok"] else 2


def _cmd_inspect(args: argparse.Namespace) -> int:
    src = Path(args.file).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (src.parent / f"inspect_{src.stem}")
    info = inspect_cad_file(src, out_dir=out_dir)
    if args.preview:
        prev_png = out_dir / f"{src.stem}_preview_sheet.png"
        render_preview_sheet(src, prev_png, title=f"参考件: {src.name}", mode="exact")
        info["preview_sheet_png"] = str(prev_png)
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


def _cmd_open_slicer(args: argparse.Namespace) -> int:
    import os
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
    # Auto-discover sibling 3d-mark directory if present locally
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
    p_init.add_argument("--mode", choices=["fdm", "cnc", "exact"], default="fdm")
    p_init.add_argument("--ref", help="Optional existing STEP/STL file to copy into refs/")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing part.py")
    p_init.set_defaults(func=_cmd_init_part)

    p_build = sub.add_parser("build", help="Build part.py -> STEP + STL + Verify + 10-View PNG + Diff + 3D HTML")
    p_build.add_argument("script", help="Path to part.py")
    p_build.add_argument("--mode", choices=["fdm", "cnc", "exact"], default=None, help="Override manufacturing mode")
    p_build.add_argument("-o", "--out-dir", default=None, help="Custom output directory")
    p_build.add_argument("--no-html", action="store_true", help="Skip generating viewer.html")
    p_build.add_argument("--no-diff", action="store_true", help="Skip generating diff_sheet.png")
    p_build.add_argument("--no-snapshot", action="store_true", help="Do not overwrite .prev baseline STL")
    p_build.add_argument("--expected-solids", type=int, default=1, help="Expected connected solid count (0 to disable)")
    p_build.set_defaults(func=_cmd_build)

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
