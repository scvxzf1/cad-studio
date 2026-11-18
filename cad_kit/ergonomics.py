"""Agent Ergonomics & Performance Helpers:
  1. `set_part_params`: AST-verified, comment-preserving atomic parameter updater for `part.py`.
  2. `get_workspace_status`: Instant project/Params/Anchor indexer (extracts inline Chinese comments!).
  3. `diagnose_cad_exception`: Translates cryptic OpenCASCADE / Python tracebacks into actionable fixes.
  4. `lookup_engineering_spec`: Zero-token-bloat query over fasteners, bearings, extrusions & templates.
"""

from __future__ import annotations

import ast
import difflib
import inspect
import json
from pathlib import Path
import traceback
from typing import Any

from .templates import (
    BEARING_TABLE,
    EXTRUSION_SPECS,
    EnclosureParams,
    cut_panel_port,
    make_bearing_block,
    make_din_rail_clip,
    make_extrusion_bracket,
    make_extrusion_mount_plate,
    make_split_enclosure,
    make_spur_gear,
)
from .tolerances import FASTENER_TABLE, ToleranceProfile


def extract_params_schema(script_path: str | Path) -> dict[str, dict[str, Any]]:
    """Parse `@dataclass class Params` from `part.py` via AST, including inline `# comments`!"""
    sp = Path(script_path).expanduser().resolve()
    if not sp.exists():
        return {}
    src = sp.read_text(encoding="utf-8")
    lines = src.splitlines()
    try:
        tree = ast.parse(src, filename=str(sp))
    except SyntaxError:
        return {}

    schema: dict[str, dict[str, Any]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Params":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fname = stmt.target.id
                    type_str = ast.get_source_segment(src, stmt.annotation) or "Any"
                    val_obj: Any = None
                    val_src = ""
                    if stmt.value is not None:
                        val_src = ast.get_source_segment(src, stmt.value) or ""
                        try:
                            val_obj = ast.literal_eval(stmt.value)
                        except Exception:
                            val_obj = val_src
                    raw_line = lines[stmt.lineno - 1] if 1 <= stmt.lineno <= len(lines) else ""
                    comment = ""
                    if "#" in raw_line:
                        comment = raw_line.split("#", 1)[1].strip()
                    schema[fname] = {
                        "value": val_obj,
                        "type": type_str,
                        "comment": comment,
                        "lineno": stmt.lineno,
                    }
    return schema


def _coerce_param_value(raw_val: Any, type_hint: str, current_val: Any) -> Any:
    """Coerce string/JSON input value to match the declared Python field type."""
    if isinstance(raw_val, str):
        s = raw_val.strip()
        if "bool" in type_hint or isinstance(current_val, bool):
            if s.lower() in ("true", "1", "yes", "on"):
                return True
            if s.lower() in ("false", "0", "no", "off"):
                return False
        if type_hint == "int" or (isinstance(current_val, int) and not isinstance(current_val, bool)):
            return int(float(s))
        if type_hint == "float" or isinstance(current_val, float):
            return float(s)
        if s.startswith(("[", "(", "{")):
            try:
                return ast.literal_eval(s)
            except Exception:
                pass
        # Strip surrounding quotes if passed via CLI like '"M4"'
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            return s[1:-1]
        return s
    if isinstance(current_val, float) and isinstance(raw_val, int) and not isinstance(raw_val, bool):
        return float(raw_val)
    return raw_val


def set_part_params(script_path: str | Path, updates: dict[str, Any]) -> dict[str, Any]:
    """Atomically update `Params` default values in `part.py` while preserving comments & formatting."""
    sp = Path(script_path).expanduser().resolve()
    if not sp.exists():
        raise FileNotFoundError(f"Part script not found: {sp}")

    src = sp.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(sp))

    params_cls: ast.ClassDef | None = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Params":
            params_cls = node
            break
    if params_cls is None:
        raise ValueError(f"No `class Params` found in {sp}")

    field_nodes: dict[str, ast.AnnAssign] = {}
    for stmt in params_cls.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            field_nodes[stmt.target.id] = stmt

    available = list(field_nodes.keys())
    unknown = [k for k in updates if k not in field_nodes]
    if unknown:
        hints = {
            u: difflib.get_close_matches(u, available, n=3, cutoff=0.45)
            for u in unknown
        }
        raise KeyError(
            f"未知参数 {unknown}！`{sp.parent.name}` 可调参数列表: {available} (相近建议: {hints})"
        )

    # Compute exact byte/character spans of stmt.value in source and replace back-to-front!
    line_offsets: list[int] = [0]
    for line in src.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(line))

    def _node_char_span(n: ast.AST) -> tuple[int, int]:
        start = line_offsets[n.lineno - 1] + n.col_offset  # type: ignore[attr-defined]
        end = line_offsets[n.end_lineno - 1] + n.end_col_offset  # type: ignore[attr-defined]
        return start, end

    edits: list[tuple[int, int, str, str, Any, Any]] = []
    for k, raw_new in updates.items():
        stmt = field_nodes[k]
        assert stmt.value is not None
        type_str = ast.get_source_segment(src, stmt.annotation) or ""
        try:
            old_val = ast.literal_eval(stmt.value)
        except Exception:
            old_val = ast.get_source_segment(src, stmt.value)
        coerced = _coerce_param_value(raw_new, type_str, old_val)
        new_literal = repr(coerced)
        s_idx, e_idx = _node_char_span(stmt.value)
        edits.append((s_idx, e_idx, new_literal, k, old_val, coerced))

    edits.sort(key=lambda item: item[0], reverse=True)
    patched = src
    applied_changes: dict[str, dict[str, Any]] = {}
    for s_idx, e_idx, new_lit, k, old_val, coerced in edits:
        patched = patched[:s_idx] + new_lit + patched[e_idx:]
        applied_changes[k] = {"old": old_val, "new": coerced}

    # Verify patched code still parses as valid Python AST before writing to disk!
    ast.parse(patched, filename=str(sp))
    sp.write_text(patched, encoding="utf-8")
    return {
        "script": str(sp),
        "updated": applied_changes,
        "all_params": {k: v["value"] for k, v in extract_params_schema(sp).items()},
    }


def diagnose_cad_exception(exc: BaseException, script_path: str | Path) -> dict[str, Any]:
    """Extract exact failing user line in `part.py`/`assembly.py` and provide Chinese CAD fix guidance."""
    sp = Path(script_path).expanduser().resolve()
    tb_list = traceback.extract_tb(exc.__traceback__)
    user_frame = None
    for fr in reversed(tb_list):
        if Path(fr.filename).resolve() == sp:
            user_frame = fr
            break
    if user_frame is None and tb_list:
        user_frame = tb_list[-1]

    err_type = type(exc).__name__
    err_msg = str(exc)
    line_info = (
        {"file": user_frame.filename, "lineno": user_frame.lineno, "func": user_frame.name, "code": user_frame.line}
        if user_frame
        else None
    )
    code_txt = (user_frame.line or "") if user_frame else ""

    advice = "请检查报错行的尺寸正负号、旋转轴向或布尔体坐标。"
    if "StdFail_NotDone" in err_type or "StdFail_NotDone" in err_msg or "BRep_API" in err_msg:
        if "fillet" in code_txt or "fillet" in err_msg.lower():
            advice = "圆角半径超出了相邻面/边的几何极限！请改用 `safe_fillet(part, edges, radius, reg)` 或减小圆角半径，并确保先导大外圆角、最后打孔。"
        elif "chamfer" in code_txt or "chamfer" in err_msg.lower():
            advice = "倒角宽度过大撞到了相邻孔或台阶边！请改用 `safe_chamfer(...)` 或减小倒角尺寸。"
        else:
            advice = "OpenCASCADE 布尔/造型内核执行失败：请避免两个实体做零厚度擦边共面相切，相加时留 `0.2mm` 嵌入量，相减切割体留 `+1~2mm` 越界余量 (overcut)。"
    elif isinstance(exc, IndexError):
        advice = "拓扑筛选结果为空 (`faces()`/`edges()` 列表越界)！请检查 `filter_by` 条件是否过严导致没有匹配到边或面。"
    elif "Unsupported fastener" in err_msg:
        advice = f"紧固件规格名不受支持，内置支持规格为: {list(FASTENER_TABLE.keys())}。"
    elif isinstance(exc, ValueError) and ("empty" in err_msg.lower() or "intersect" in err_msg.lower()):
        advice = "实体已被完全切空或截面未切到零件，请核对切割体尺寸和截面高度坐标。"

    return {
        "error_type": err_type,
        "error_message": err_msg,
        "location": line_info,
        "diagnosis_cn": advice,
        "traceback_tail": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
    }


def get_workspace_status(studio_root: str | Path) -> dict[str, Any]:
    """Return a fast, token-compact index of all parts & assemblies in `projects/`."""
    root = Path(studio_root).expanduser().resolve()
    proj_root = root / "projects"
    items: list[dict[str, Any]] = []
    if not proj_root.is_dir():
        return {"studio_root": str(root), "projects": []}

    for pdir in sorted(proj_root.iterdir()):
        if not pdir.is_dir() or pdir.name.startswith("."):
            continue
        part_py = pdir / "part.py"
        asm_py = pdir / "assembly.py"
        mf_part = pdir / "out" / "manifest.json"
        mf_asm = pdir / "out" / "assembly_manifest.json"

        entry: dict[str, Any] = {"name": pdir.name}
        if part_py.exists():
            entry["kind"] = "part"
            entry["script"] = str(part_py.relative_to(root))
            entry["params"] = extract_params_schema(part_py)
            if mf_part.exists():
                try:
                    mf = json.loads(mf_part.read_text(encoding="utf-8"))
                    entry["mode"] = mf.get("mode")
                    entry["bbox_mm"] = mf.get("verification", {}).get("bbox", {}).get("size_mm")
                    entry["ok"] = mf.get("verification", {}).get("ok")
                    entry["anchors"] = [
                        f"{a['id']}:{a['desc']}@({a['pos'][0]},{a['pos'][1]},{a['pos'][2]})"
                        for a in mf.get("registry", {}).get("anchors", [])
                    ]
                    entry["outputs"] = {k: v for k, v in mf.get("files", {}).items() if v}
                except Exception:
                    pass
        elif asm_py.exists():
            entry["kind"] = "assembly"
            entry["script"] = str(asm_py.relative_to(root))
            if mf_asm.exists():
                try:
                    mf = json.loads(mf_asm.read_text(encoding="utf-8"))
                    entry["mode"] = mf.get("mode")
                    entry["bbox_mm"] = mf.get("verification", {}).get("bbox", {}).get("size_mm")
                    entry["ok"] = mf.get("verification", {}).get("ok")
                    entry["parts_count"] = mf.get("verification", {}).get("parts_count")
                    entry["est_weight_g"] = mf.get("bom", {}).get("total_estimated_print_weight_g")
                    entry["outputs"] = {k: v for k, v in mf.get("files", {}).items() if v}
                except Exception:
                    pass
        else:
            continue

        refs = [str(f.relative_to(root)) for f in sorted((pdir / "refs").glob("*")) if f.is_file()] if (pdir / "refs").is_dir() else []
        if refs:
            entry["refs"] = refs
        items.append(entry)

    return {"studio_root": str(root), "projects_count": len(items), "projects": items}


def lookup_engineering_spec(query: str = "") -> dict[str, Any]:
    """Query metric fastener dimensions (FDM/CNC/Exact), bearings, extrusions, or template APIs."""
    q = query.strip().upper()
    out: dict[str, Any] = {}

    # 1. Fastener match (e.g. "M3", "M4", "ALL", "")
    matched_fasteners = [k for k in FASTENER_TABLE if (not q or q in k or k in q or "FASTENER" in q or "螺丝" in query or "螺母" in query or "热熔" in query)]
    if matched_fasteners:
        f_res = {}
        for sz in matched_fasteners:
            tol_fdm = ToleranceProfile.for_mode("fdm")
            tol_cnc = ToleranceProfile.for_mode("cnc")
            f_res[sz] = {
                "fdm": {
                    "clearance_d": tol_fdm.hole_d(sz, "clearance"),
                    "close_d": tol_fdm.hole_d(sz, "close"),
                    "self_tap_d": tol_fdm.hole_d(sz, "tap"),
                    "counterbore_thru_cb_depth": tol_fdm.counterbore_dims(sz),
                    "heatset_bore_depth_lead": tol_fdm.heatset_dims(sz),
                    "nut_trap_s_h": tol_fdm.nut_trap_dims(sz),
                },
                "cnc": {
                    "clearance_d": tol_cnc.hole_d(sz, "clearance"),
                    "tap_drill_d": tol_cnc.hole_d(sz, "tap"),
                    "counterbore_thru_cb_depth": tol_cnc.counterbore_dims(sz),
                },
            }
        out["fasteners"] = f_res

    # 2. Bearing match
    matched_brg = [k for k in BEARING_TABLE if (not q or k in q or "BEARING" in q or "轴承" in query)]
    if matched_brg:
        out["bearings_od_id_width_mm"] = {k: BEARING_TABLE[k] for k in matched_brg}

    # 3. Extrusion match
    matched_ext = [k for k in EXTRUSION_SPECS if (not q or k in q or "EXTRUSION" in q or "型材" in query)]
    if matched_ext:
        out["extrusions"] = {
            k: {
                "pitch_mm": EXTRUSION_SPECS[k].pitch_mm,
                "slot_width_mm": EXTRUSION_SPECS[k].slot_width_mm,
                "default_screw": EXTRUSION_SPECS[k].default_screw,
            }
            for k in matched_ext
        }

    # 4. Template signatures
    if not q or any(w in query.lower() for w in ("template", "enclosure", "gear", "din", "外壳", "齿轮", "导轨", "母版")):
        tpl_funcs = [
            make_split_enclosure,
            cut_panel_port,
            make_extrusion_bracket,
            make_extrusion_mount_plate,
            make_din_rail_clip,
            make_spur_gear,
            make_bearing_block,
        ]
        out["templates_api"] = {
            fn.__name__: str(inspect.signature(fn))
            for fn in tpl_funcs
        }
        out["EnclosureParams_fields"] = {
            k: v.default for k, v in EnclosureParams.__dataclass_fields__.items() if not isinstance(v.default, type(v.default_factory))
        }

    return out
