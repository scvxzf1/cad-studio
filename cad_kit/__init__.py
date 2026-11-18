"""cad_kit: Agentic Structural CAD Toolkit (FDM / CNC / Exact)."""

from .features import (
    FeatureAnchor,
    FeatureRegistry,
    make_counterbore_cutter,
    make_countersink_cutter,
    make_heatset_cutter,
    make_hole_cutter,
    make_mounting_boss,
    make_nut_trap_cutter,
    make_rib,
    make_slot_cutter,
    safe_chamfer,
    safe_fillet,
)
from .inspector import inspect_cad_file, shape_to_trimesh
from .mesh_ops import center_mesh_on_bed, modify_mesh
from .renderer import compute_volumetric_diff, render_diff_sheet, render_preview_sheet
from .tolerances import FASTENER_TABLE, FastenerEntry, ManufacturingMode, ToleranceProfile, get_fastener
from .verify import verify_part_geometry
from .viewer_html import export_viewer_html

__all__ = [
    "FASTENER_TABLE",
    "FastenerEntry",
    "FeatureAnchor",
    "FeatureRegistry",
    "ManufacturingMode",
    "ToleranceProfile",
    "center_mesh_on_bed",
    "compute_volumetric_diff",
    "export_viewer_html",
    "get_fastener",
    "inspect_cad_file",
    "make_counterbore_cutter",
    "make_countersink_cutter",
    "make_heatset_cutter",
    "make_hole_cutter",
    "make_mounting_boss",
    "make_nut_trap_cutter",
    "make_rib",
    "make_slot_cutter",
    "modify_mesh",
    "render_diff_sheet",
    "render_preview_sheet",
    "safe_chamfer",
    "safe_fillet",
    "shape_to_trimesh",
    "verify_part_geometry",
]
