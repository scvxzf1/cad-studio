"""Semantic Feature Registry & Reusable Structural CAD Builders (build123d Algebra Mode).

Coordinate & Orientation Convention (locked across CAD Studio):
    -Y = 正面 Front   | +Y = 背面 Back
    -X = 左侧 Left    | +X = 右侧 Right
    +Z = 顶面 Top     | -Z = 底面 Bottom (Default FDM build plate at Z=0)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
import math
from typing import Iterable, Sequence

from build123d import (
    Align,
    Axis,
    Box,
    Cone,
    Cylinder,
    Edge,
    Part,
    Plane,
    Polygon,
    Pos,
    RegularPolygon,
    Rot,
    ShapeList,
    SlotOverall,
    chamfer,
    extrude,
    fillet,
)

from .tolerances import ManufacturingMode, ToleranceProfile, get_fastener


@dataclass
class FeatureAnchor:
    """Named spatial anchor projected onto 6-view drawings and interactive 3D viewer."""

    id: str
    pos: tuple[float, float, float]
    desc: str
    kind: str = "feature"  # hole | slot | boss | rib | cutout | mate | datum
    spec: str = ""
    normal: tuple[float, float, float] | None = None


@dataclass
class FeatureRegistry:
    """Collects semantic feature anchors & build metadata for a part."""

    part_name: str
    mode: ManufacturingMode = "fdm"
    description: str = ""
    anchors: list[FeatureAnchor] = field(default_factory=list)
    reference_files: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def mark(
        self,
        feature_id: str,
        pos: Sequence[float],
        desc: str,
        kind: str = "feature",
        spec: str = "",
        normal: Sequence[float] | None = None,
    ) -> tuple[float, float, float]:
        """Register a named feature and return its (x, y, z) tuple for inline convenience."""
        pt = (round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(pos[2]), 3))
        nrm = (
            (round(float(normal[0]), 3), round(float(normal[1]), 3), round(float(normal[2]), 3))
            if normal is not None
            else None
        )
        # Replace if duplicate ID exists
        self.anchors = [a for a in self.anchors if a.id != feature_id]
        self.anchors.append(
            FeatureAnchor(id=feature_id, pos=pt, desc=desc, kind=kind, spec=spec, normal=nrm)
        )
        return pt

    def add_ref(self, path: str) -> None:
        if path not in self.reference_files:
            self.reference_files.append(path)

    def warn(self, msg: str) -> None:
        self.notes.append(msg)

    def to_dict(self) -> dict:
        return {
            "part_name": self.part_name,
            "mode": self.mode,
            "description": self.description,
            "reference_files": list(self.reference_files),
            "notes": list(self.notes),
            "anchors": [asdict(a) for a in self.anchors],
        }


# ---------------------------------------------------------------------------
# Cutter & Structural Feature Generators (build123d Algebra Mode)
# Convention:
#   Cutters have their datum origin at z = 0 on the ENTRY SURFACE and cut
#   DOWNWARD along -Z (with +Z overcut relief where appropriate so boolean
#   differences never leave zero-thickness coplanar skins).
#   To drill on another face, transform with Pos(x, y, z) * Rot(...).
# ---------------------------------------------------------------------------


def make_hole_cutter(
    tol: ToleranceProfile,
    size: str = "M3",
    depth: float = 40.0,
    kind: str = "clearance",
    overcut: float = 2.0,
) -> Part:
    """Create a cylindrical hole cutter entering at Z=0 and extending to Z=-depth."""
    d = tol.hole_d(size, kind=kind)  # type: ignore[arg-type]
    total_h = depth + overcut
    return Pos(0, 0, overcut) * Cylinder(
        radius=d / 2.0,
        height=total_h,
        align=(Align.CENTER, Align.CENTER, Align.MAX),
    )


def make_counterbore_cutter(
    tol: ToleranceProfile,
    size: str = "M3",
    depth: float = 40.0,
    cbore_depth: float | None = None,
    close_fit: bool = False,
    head_overcut: float = 15.0,
    upside_down_bridge: bool = False,
) -> Part:
    """Create a Socket Head Cap Screw (SHCS) counterbore cutter.

    Entry surface is Z=0:
      - Counterbore pocket goes from Z = +head_overcut down to Z = -cbore_h
      - Shaft hole goes from Z = -cbore_h down to Z = -depth
      - If upside_down_bridge=True and tol.mode=='fdm', leaves a 1-layer (0.2mm)
        sacrificial bridge membrane at the counterbore step so upside-down counterbores
        print cleanly without supports!
    """
    thru_d, cbore_d, default_cb_h = tol.counterbore_dims(size, close_fit=close_fit)
    cb_h = cbore_depth if cbore_depth is not None else default_cb_h

    head_cut = Pos(0, 0, head_overcut) * Cylinder(
        radius=cbore_d / 2.0,
        height=cb_h + head_overcut,
        align=(Align.CENTER, Align.CENTER, Align.MAX),
    )
    shaft_start_z = -cb_h - (tol.layer_height if (upside_down_bridge and tol.mode == "fdm") else 0.0)
    shaft_len = max(1.0, depth + shaft_start_z + 1.0)
    shaft_cut = Pos(0, 0, shaft_start_z + (0.0 if (upside_down_bridge and tol.mode == "fdm") else 0.2)) * Cylinder(
        radius=thru_d / 2.0,
        height=shaft_len,
        align=(Align.CENTER, Align.CENTER, Align.MAX),
    )
    return head_cut + shaft_cut


def make_countersink_cutter(
    tol: ToleranceProfile,
    size: str = "M3",
    depth: float = 40.0,
    close_fit: bool = False,
    head_overcut: float = 10.0,
) -> Part:
    """Create a 90-degree Flat Head Countersunk Screw (FHCS) cutter entering at Z=0."""
    thru_d, csink_d, angle_deg = tol.countersink_dims(size, close_fit=close_fit)
    half_angle = math.radians(angle_deg / 2.0)
    cone_h = ((csink_d - thru_d) / 2.0) / math.tan(half_angle)

    over_cyl = Pos(0, 0, head_overcut) * Cylinder(
        radius=csink_d / 2.0,
        height=head_overcut,
        align=(Align.CENTER, Align.CENTER, Align.MAX),
    )
    cone = Cone(
        bottom_radius=thru_d / 2.0,
        top_radius=csink_d / 2.0,
        height=cone_h,
        align=(Align.CENTER, Align.CENTER, Align.MAX),
    )
    shaft = Cylinder(
        radius=thru_d / 2.0,
        height=depth,
        align=(Align.CENTER, Align.CENTER, Align.MAX),
    )
    return over_cyl + cone + shaft


def make_heatset_cutter(
    tol: ToleranceProfile,
    size: str = "M3",
    screw_relief_depth: float = 3.0,
    overcut: float = 1.0,
) -> Part:
    """Create a stepped pocket cutter for brass heat-set inserts entering at Z=0.

    In 'cnc' mode, gracefully falls back to a standard metric tap drill hole.
    """
    if tol.mode == "cnc":
        f = get_fastener(size)
        return make_hole_cutter(tol, size=size, depth=f.heatset_depth + screw_relief_depth, kind="tap")

    bore_d, pocket_h, lead_d, lead_h = tol.heatset_dims(size)
    thru_relief_d = tol.hole_d(size, "clearance")

    lead_cut = Pos(0, 0, overcut) * Cylinder(
        radius=lead_d / 2.0,
        height=lead_h + overcut,
        align=(Align.CENTER, Align.CENTER, Align.MAX),
    )
    main_bore = Cylinder(
        radius=bore_d / 2.0,
        height=pocket_h,
        align=(Align.CENTER, Align.CENTER, Align.MAX),
    )
    cutter = lead_cut + main_bore
    if screw_relief_depth > 0:
        relief = Pos(0, 0, -pocket_h + 0.2) * Cylinder(
            radius=thru_relief_d / 2.0,
            height=screw_relief_depth + 0.2,
            align=(Align.CENTER, Align.CENTER, Align.MAX),
        )
        cutter = cutter + relief
    return cutter


def make_nut_trap_cutter(
    tol: ToleranceProfile,
    size: str = "M3",
    pocket_depth: float | None = None,
    bolt_depth: float = 35.0,
    side_slot_len: float = 0.0,
    flat_along_x: bool = True,
) -> Part:
    """Create a metric hex nut trap cutter (top drop-in OR captive side-slot).

    - Default (side_slot_len == 0): Hex pocket opens upward at Z=0 down to -pocket_depth,
      with coaxial bolt clearance shaft extending along Z.
    - Side-slot captive nut (side_slot_len > 0): Hex pocket is centered at Z=0 with height
      equal to nut thickness, plus a rectangular insertion channel extending +Y by side_slot_len!
    """
    s, nut_h = tol.nut_trap_dims(size)
    circum_r = (s / 2.0) / math.cos(math.pi / 6.0)
    rot_deg = 0.0 if flat_along_x else 30.0
    thru_d = tol.hole_d(size, "clearance")

    if side_slot_len > 0:
        hex_prism = Rot(0, 0, rot_deg) * extrude(
            RegularPolygon(radius=circum_r, side_count=6),
            amount=nut_h / 2.0,
            both=True,
        )
        # Rectangular slide-in channel along +Y with width = s (flats parallel to YZ when rot_deg=30)
        slot_hex = Rot(0, 0, 30.0) * extrude(
            RegularPolygon(radius=circum_r, side_count=6),
            amount=nut_h / 2.0,
            both=True,
        )
        channel = Pos(0, side_slot_len / 2.0, 0) * Box(
            s,
            side_slot_len,
            nut_h,
            align=(Align.CENTER, Align.CENTER, Align.CENTER),
        )
        shaft = Cylinder(
            radius=thru_d / 2.0,
            height=bolt_depth,
            align=(Align.CENTER, Align.CENTER, Align.CENTER),
        )
        return slot_hex + channel + shaft

    p_depth = pocket_depth if pocket_depth is not None else nut_h
    hex_pocket = Pos(0, 0, 1.0) * Rot(0, 0, rot_deg) * extrude(
        RegularPolygon(radius=circum_r, side_count=6),
        amount=-(p_depth + 1.0),
    )
    shaft = Pos(0, 0, 1.0) * Cylinder(
        radius=thru_d / 2.0,
        height=bolt_depth + 1.0,
        align=(Align.CENTER, Align.CENTER, Align.MAX),
    )
    return hex_pocket + shaft


def make_slot_cutter(
    tol: ToleranceProfile,
    size_or_width: str | float = "M3",
    center_span: float = 8.0,
    depth: float = 40.0,
    overcut: float = 2.0,
) -> Part:
    """Create an elongated stadium adjustment slot along the X axis, cutting from Z=0 to -depth."""
    if isinstance(size_or_width, str):
        w = tol.hole_d(size_or_width, "clearance")
    else:
        w = float(size_or_width) + tol.inner_hole_comp
    overall_len = w + max(0.1, float(center_span))
    sketch = SlotOverall(width=overall_len, height=w)
    return Pos(0, 0, overcut) * extrude(sketch, amount=-(depth + overcut))


def make_mounting_boss(
    outer_d: float,
    height: float,
    base_flare: float = 1.5,
    flare_height: float = 2.0,
) -> Part:
    """Create a cylindrical standoff boss growing from Z=0 to Z=+height with conical root reinforcement."""
    cyl = Cylinder(
        radius=outer_d / 2.0,
        height=height,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )
    if base_flare > 0 and flare_height > 0 and height > flare_height:
        cone = Cone(
            bottom_radius=(outer_d / 2.0) + base_flare,
            top_radius=outer_d / 2.0,
            height=min(flare_height, height),
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        return cyl + cone
    return cyl


def make_rib(
    length: float,
    height: float,
    thickness: float,
    tip_flat: float = 0.8,
) -> Part:
    """Create a right-triangle structural gusset rib in the XZ plane.

    Origin (0, 0, 0) is the right-angle corner:
      - Base runs from X = 0 to X = +length (at Z = 0)
      - Vertical wall runs from Z = 0 to Z = +height (at X = 0)
      - Centered on Y in [-thickness/2, +thickness/2]
      - Includes optional small flat tip_flat so razor-sharp 0mm edges don't fail slicing.
    """
    pts = [(0.0, 0.0), (length, 0.0)]
    if tip_flat > 0:
        pts.extend([(length, tip_flat), (tip_flat, height), (0.0, height)])
    else:
        pts.append((0.0, height))
    poly = Polygon(*pts, align=None)
    rib_2d = Plane.XZ * poly
    return extrude(rib_2d, amount=thickness / 2.0, both=True)


def safe_fillet(
    part: Part,
    edges: Iterable[Edge],
    radius: float,
    reg: FeatureRegistry | None = None,
    label: str = "fillet",
) -> Part:
    """Apply fillet safely; if OCCT kernel rejects exact radius, step down gracefully."""
    edge_list = ShapeList(edges)
    if radius <= 0 or len(edge_list) == 0:
        return part
    for scale in (1.0, 0.85, 0.7, 0.5, 0.35):
        r = round(radius * scale, 3)
        if r < 0.05:
            break
        try:
            res = fillet(edge_list, radius=r)
            if scale < 1.0 and reg is not None:
                reg.warn(f"[{label}] Fillet stepped down from {radius}mm to {r}mm due to adjacent edge constraint.")
            return res
        except Exception:
            continue
    if reg is not None:
        reg.warn(f"[{label}] Fillet ({radius}mm) skipped: incompatible edge topology.")
    return part


def safe_chamfer(
    part: Part,
    edges: Iterable[Edge],
    length: float,
    reg: FeatureRegistry | None = None,
    label: str = "chamfer",
) -> Part:
    """Apply chamfer safely; if OCCT kernel rejects exact length, step down gracefully."""
    edge_list = ShapeList(edges)
    if length <= 0 or len(edge_list) == 0:
        return part
    for scale in (1.0, 0.8, 0.6, 0.4):
        c = round(length * scale, 3)
        if c < 0.05:
            break
        try:
            res = chamfer(edge_list, length=c)
            if scale < 1.0 and reg is not None:
                reg.warn(f"[{label}] Chamfer stepped down from {length}mm to {c}mm due to adjacent edge constraint.")
            return res
        except Exception:
            continue
    if reg is not None:
        reg.warn(f"[{label}] Chamfer ({length}mm) skipped: incompatible edge topology.")
    return part
