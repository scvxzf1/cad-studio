"""Hot-Resident Stdio MCP Server for CAD Studio (`cad-studio-mcp`).

Keeps OpenCASCADE (`build123d`), VTK/OpenGL (`PyVista`), `trimesh`, and `manifold3d`
pre-warmed in RAM across the entire conversation so iterative builds & parameter tweaks
complete in ~1.5-2.5 seconds with zero Python cold-import or OpenGL window teardown overhead!
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
import shutil
import sys
from typing import Any

# Redirect any stray library stdout during import to stderr so Stdio JSON-RPC remains 100% pure!
with contextlib.redirect_stdout(sys.stderr):
    import build123d as _bd  # noqa: F401
    from mcp.server.mcpserver import MCPServer

    from .assembly import run_assembly_build
    from .cli import ASSEMBLY_SCAFFOLD, TEMPLATES_SCAFFOLD, _load_part_module, run_part_build
    from .ergonomics import (
        diagnose_cad_exception,
        get_workspace_status,
        lookup_engineering_spec,
        set_part_params,
    )
    from .export_2d import export_2d_profile
    from .inspector import inspect_cad_file
    from .large_cad import optimize_cad_file
    from .printability import apply_optimal_orientation, export_3mf, render_printability_sheet
    from .renderer import render_preview_sheet

DEFAULT_STUDIO_ROOT = Path(os.environ.get("CAD_STUDIO_ROOT", Path(__file__).resolve().parents[1])).resolve()

mcp = MCPServer(
    "cad-studio-mcp",
    instructions=(
        "CAD Studio 结构件建模与测绘热驻留工具集。坐标系铁律: 前=-Y, 后=+Y, 左=-X, 右=+X, 上=+Z, 下=-Z(Z=0热床)。"
        "修改参数优先调用 cad_set_params；构建完成必须调用 read_file 查看返回的 preview_sheet_png / diff_sheet_png 视觉自检！"
    ),
)


def _resolve_target_path(p: str, studio_root: Path | None = None) -> Path:
    pp = Path(p).expanduser()
    if pp.is_absolute():
        return pp.resolve()
    base = studio_root or DEFAULT_STUDIO_ROOT
    cand = (base / pp).resolve()
    if cand.exists():
        return cand
    return (Path.cwd() / pp).resolve()


@mcp.tool(
    description="快速查看 CAD Studio 中所有零件与装配体项目索引、可调参数 (Params 含中文注释)、特征锚点 (H1/S1) 及输出文件路径。"
)
def cad_status(workspace_dir: str = "") -> dict[str, Any]:
    with contextlib.redirect_stdout(sys.stderr):
        root = Path(workspace_dir).expanduser().resolve() if workspace_dir else DEFAULT_STUDIO_ROOT
        return get_workspace_status(root)


@mcp.tool(
    description="按关键词即时查询紧固件尺寸 (M1.6-M8 在 FDM/CNC 下的通孔/沉头/热熔螺母/螺母槽)、轴承 (608/625等)、铝型材 (2020-4040) 及结构母版函数签名。"
)
def cad_lookup_spec(query: str = "") -> dict[str, Any]:
    with contextlib.redirect_stdout(sys.stderr):
        return lookup_engineering_spec(query)


@mcp.tool(
    description="新建零件或装配体脚手架目录。template 可选: basic | enclosure | extrusion-bracket | din-clip | gear；is_assembly=True 时创建多零件总装项目。"
)
def cad_init_project(
    name: str,
    template: str = "basic",
    mode: str = "fdm",
    is_assembly: bool = False,
    ref_file: str = "",
    workspace_dir: str = "",
    overwrite: bool = False,
) -> dict[str, Any]:
    with contextlib.redirect_stdout(sys.stderr):
        root = Path(workspace_dir).expanduser().resolve() if workspace_dir else DEFAULT_STUDIO_ROOT
        proj_dir = root / "projects" / name
        (proj_dir / "out").mkdir(parents=True, exist_ok=True)
        if is_assembly:
            script = proj_dir / "assembly.py"
            if script.exists() and not overwrite:
                return {"ok": False, "error": f"{script} 已存在 (如需覆盖传 overwrite=True)"}
            script.write_text(ASSEMBLY_SCAFFOLD.replace("__NAME__", name).replace("__MODE__", mode), encoding="utf-8")
            return {"ok": True, "kind": "assembly", "script": str(script)}

        (proj_dir / "refs").mkdir(parents=True, exist_ok=True)
        script = proj_dir / "part.py"
        if script.exists() and not overwrite:
            return {"ok": False, "error": f"{script} 已存在 (如需覆盖传 overwrite=True)"}
        if template not in TEMPLATES_SCAFFOLD:
            return {"ok": False, "error": f"未知模板 '{template}'，可选: {list(TEMPLATES_SCAFFOLD.keys())}"}
        script.write_text(TEMPLATES_SCAFFOLD[template].replace("__NAME__", name).replace("__MODE__", mode), encoding="utf-8")
        copied_ref = None
        if ref_file:
            rf = _resolve_target_path(ref_file, root)
            if rf.exists():
                dst = proj_dir / "refs" / rf.name
                shutil.copy2(rf, dst)
                copied_ref = str(dst)
        return {"ok": True, "kind": "part", "template": template, "script": str(script), "copied_ref": copied_ref}


@mcp.tool(
    description="原子级安全修改 part.py 中 Params 的字段默认值 (保留源码注释、零语法错误风险)，并直接在热驻留内存中极速重构建出 STEP/STL/3MF、10视图与红绿 Diff 图！"
)
def cad_set_params(
    script_path: str,
    updates: dict[str, Any],
    mode: str = "",
    fast: bool = False,
    build: bool = True,
) -> dict[str, Any]:
    with contextlib.redirect_stdout(sys.stderr):
        sp = _resolve_target_path(script_path)
        try:
            patch_res = set_part_params(sp, updates)
            if not build:
                return {"ok": True, **patch_res}
            manifest = run_part_build(
                script_path=sp,
                mode=mode or None,
                gen_html=not fast,
            )
            return {"ok": manifest["verification"]["ok"], "param_updates": patch_res["updated"], "manifest": manifest}
        except Exception as exc:
            return {"ok": False, "diagnostics": diagnose_cad_exception(exc, sp)}


@mcp.tool(
    description="热驻留极速构建 part.py 或 assembly.py：导出 STEP/STL/3MF、运行干涉碰撞检查、生成 10视图 preview_sheet.png、修改对比 diff_sheet.png 与交互 3D viewer.html。"
)
def cad_build(
    script_path: str,
    mode: str = "",
    dxf_face: str = "",
    print_check: bool = False,
    auto_orient: bool = False,
    fast: bool = False,
    out_dir: str = "",
) -> dict[str, Any]:
    with contextlib.redirect_stdout(sys.stderr):
        sp = _resolve_target_path(script_path)
        od = Path(out_dir).expanduser().resolve() if out_dir else None
        try:
            mod = _load_part_module(sp)
            if hasattr(mod, "build_assembly") and not hasattr(mod, "build"):
                asm = mod.build_assembly(mode=mode) if mode else mod.build_assembly()
                target_od = od or (sp.parent / "out")
                res = run_assembly_build(asm, target_od)
                return {"ok": res["verification"]["ok"], "kind": "assembly", "manifest": res}

            manifest = run_part_build(
                script_path=sp,
                mode=mode or None,
                out_dir=od,
                gen_html=not fast,
                dxf_face=dxf_face or None,
                print_check=print_check,
                auto_orient=auto_orient,
            )
            return {"ok": manifest["verification"]["ok"], "kind": "part", "manifest": manifest}
        except Exception as exc:
            return {"ok": False, "diagnostics": diagnose_cad_exception(exc, sp)}


@mcp.tool(
    description="逆向体检任意现有 .step / .stp / .stl / .3mf 模型文件：提取包围盒、原点偏移警告、各台阶平面高度、圆孔坐标与直径，并生成带毫米标尺的 4 截面测绘图。"
)
def cad_inspect(file_path: str, preview: bool = False, out_dir: str = "") -> dict[str, Any]:
    with contextlib.redirect_stdout(sys.stderr):
        src = _resolve_target_path(file_path)
        od = Path(out_dir).expanduser().resolve() if out_dir else (src.parent / f"inspect_{src.stem}")
        info = inspect_cad_file(src, out_dir=od)
        if preview:
            prev_png = od / f"{src.stem}_preview_sheet.png"
            render_preview_sheet(src, prev_png, title=f"参考件: {src.name}", mode="exact")
            info["preview_sheet_png"] = str(prev_png)
        return info


@mcp.tool(
    description="运行 3D 打印可制造性 (DFM) 扫描：检测重度悬垂面、半空炒面孤岛、首层着床面积，对 ±X/±Y/±Z 六向摆放评分排行；auto_orient=True 时自动翻转至最优朝向导出 .3mf/.step。"
)
def cad_print_check(input_path: str, auto_orient: bool = False, out_dir: str = "") -> dict[str, Any]:
    with contextlib.redirect_stdout(sys.stderr):
        src = _resolve_target_path(input_path)
        od = Path(out_dir).expanduser().resolve() if out_dir else (src.parent / "out" if src.suffix == ".py" else src.parent)
        od.mkdir(parents=True, exist_ok=True)
        target: Any = src
        name = src.stem
        if src.suffix.lower() == ".py":
            mod = _load_part_module(src)
            res = mod.build()
            target = res[0] if isinstance(res, tuple) else res
            if isinstance(res, tuple) and len(res) >= 2 and res[1] is not None:
                name = res[1].part_name

        png_path = od / f"{name}_printability_sheet.png"
        _, report = render_printability_sheet(target, png_path, title=name)
        report["printability_sheet_png"] = str(png_path)
        if auto_orient:
            best = report["best_orientation"]
            oriented = apply_optimal_orientation(target, best["rot_xyz_deg"])
            out_3mf = od / f"{name}_oriented.3mf"
            export_3mf([{"name": f"{name}_oriented", "shape": oriented, "color": "#10b981"}], out_3mf)
            report["oriented_3mf"] = str(out_3mf)
            if hasattr(oriented, "solids"):
                import build123d as bd
                out_step = od / f"{name}_oriented.step"
                bd.export_step(oriented, str(out_step))
                report["oriented_step"] = str(out_step)
        return report


@mcp.tool(
    description="从 STEP / STL / part.py 导出 1:1 激光切割/CNC 二维加工文件 (.dxf + .svg + 圆孔坐标钻孔核对图)，圆孔保留原生 CIRCLE/ARC 解析实体。"
)
def cad_export_2d(
    input_path: str,
    face: str = "bottom",
    plane: str = "XY",
    offset: float | None = None,
    out_dir: str = "",
) -> dict[str, Any]:
    with contextlib.redirect_stdout(sys.stderr):
        src = _resolve_target_path(input_path)
        od = Path(out_dir).expanduser().resolve() if out_dir else (src.parent / "out" if src.suffix == ".py" else src.parent)
        target: Any = src
        name = src.stem
        if src.suffix.lower() == ".py":
            mod = _load_part_module(src)
            res = mod.build()
            target = res[0] if isinstance(res, tuple) else res
            if isinstance(res, tuple) and len(res) >= 2 and res[1] is not None:
                name = res[1].part_name
        face_arg = face if face in ("bottom", "top", "front", "back", "left", "right") else None
        return export_2d_profile(target, out_dir=od, name=name, plane=plane, face=face_arg, offset=offset)  # type: ignore[arg-type]


@mcp.tool(
    description="针对超大 .step / .stl / .3mf 文件 (几十~几百MB) 进行水密误差受控减面瘦身、剔除微小电子元件/螺纹碎块 (min_solid_diag_mm)、局部包围盒裁剪 (roi_bbox) 或转存高压缩比 .3mf / 二进制 .stl。"
)
def cad_optimize_file(
    input_path: str,
    out_path: str = "",
    tolerance_mm: float = 0.025,
    max_faces: int = 0,
    min_solid_diag_mm: float = 0.0,
    roi_bbox: list[float] | None = None,
    out_format: str = "3mf",
) -> dict[str, Any]:
    with contextlib.redirect_stdout(sys.stderr):
        src = _resolve_target_path(input_path)
        dst = Path(out_path).expanduser().resolve() if out_path else None
        roi_tuple = tuple(float(x) for x in roi_bbox) if (roi_bbox and len(roi_bbox) == 6) else None
        return optimize_cad_file(
            input_path=src,
            out_path=dst,
            tolerance_mm=tolerance_mm,
            max_faces=max_faces if max_faces > 0 else None,
            min_solid_diag_mm=min_solid_diag_mm,
            roi_bbox=roi_tuple,  # type: ignore[arg-type]
            out_format=out_format,  # type: ignore[arg-type]
        )


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
