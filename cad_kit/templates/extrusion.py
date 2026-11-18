"""European T-Slot / V-Slot Aluminum Extrusion (2020/2040/3030/4040) & 35mm DIN Rail Templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from build123d import Align, Axis, Box, Part, Plane, Polygon, Pos, Rot, extrude

from ..features import (
    FeatureRegistry,
    make_counterbore_cutter,
    make_heatset_cutter,
    make_hole_cutter,
    make_rib,
    make_slot_cutter,
    safe_chamfer,
    safe_fillet,
)
from ..tolerances import ToleranceProfile

ExtrusionSeries = Literal["2020", "2040", "3030", "4040"]


@dataclass(frozen=True)
class ExtrusionSpec:
    series: str
    pitch_mm: float
    slot_width_mm: float
    tongue_depth_mm: float
    default_screw: str


EXTRUSION_SPECS: dict[str, ExtrusionSpec] = {
    "2020": ExtrusionSpec("2020", 20.0, 6.0, 1.8, "M4"),
    "2040": ExtrusionSpec("2040", 20.0, 6.0, 1.8, "M4"),
    "3030": ExtrusionSpec("3030", 30.0, 8.0, 2.3, "M5"),
    "4040": ExtrusionSpec("4040", 40.0, 8.0, 2.6, "M6"),
}


def make_tslot_tongue(
    series: ExtrusionSeries = "2020",
    length: float = 30.0,
    tol: ToleranceProfile | None = None,
) -> Part:
    """Create a trapezoidal anti-rotation alignment tongue along the X axis, protruding -Z from Z=0.

    Union onto a Z=0 bottom face (or rotate onto a mating wall) so the bracket locks into the
    aluminum profile's center T-slot without twisting!
    """
    tol = tol or ToleranceProfile.for_mode("fdm")
    spec = EXTRUSION_SPECS[series]
    gap = tol.clearance("snug")
    w_base = spec.slot_width_mm - 2.0 * gap
    w_tip = max(2.0, w_base - 1.0)  # Lead-in chamfered taper
    h = spec.tongue_depth_mm

    pts = [
        (-w_base * 0.5, 0.2),
        (+w_base * 0.5, 0.2),
        (+w_base * 0.5, -0.3),
        (+w_tip * 0.5, -h),
        (-w_tip * 0.5, -h),
        (-w_base * 0.5, -0.3),
    ]
    prof = Plane.YZ * Polygon(*pts, align=None)
    return extrude(prof, amount=length * 0.5, both=True)


def make_extrusion_bracket(
    series: ExtrusionSeries = "2020",
    arm_len: float = 42.0,
    wall_t: float = 5.0,
    gusset_t: float = 3.6,
    screw_size: str | None = None,
    with_tongues: bool = True,
    tol: ToleranceProfile | None = None,
    reg: FeatureRegistry | None = None,
) -> Part:
    """Build a heavy-duty 90-degree aluminum extrusion corner gusset bracket.

    Mating faces:
      - Bottom horizontal arm rests on profile #1 at Z = 0 (extends along -Y..+Y? Locked: back face at Y = +arm_len/2?
        Even cleaner: Origin (0,0,0) is the outer corner! Base runs X in [-width/2, +width/2], Y in [-arm_len, 0], Z >= 0;
        Vertical wall at Y = 0 runs Z in [0, +arm_len], mating against profile #2 at Y = 0!).
    """
    tol = tol or ToleranceProfile.for_mode("fdm")
    spec = EXTRUSION_SPECS[series]
    screw = screw_size or spec.default_screw
    width = spec.pitch_mm

    # Horizontal arm: Y in [-arm_len, 0], Z in [0, wall_t]
    horiz = Pos(0, -arm_len * 0.5, 0) * Box(width, arm_len, wall_t, align=(Align.CENTER, Align.CENTER, Align.MIN))
    # Vertical arm: Y in [-wall_t, 0], Z in [0, arm_len]
    vert = Pos(0, -wall_t * 0.5, 0) * Box(width, wall_t, arm_len, align=(Align.CENTER, Align.CENTER, Align.MIN))
    part = horiz + vert

    # Left & Right side gusset ribs spanning from (Y=-wall_t, Z=wall_t) forward (-Y) and upward (+Z)
    rib_span = arm_len - wall_t - 1.5
    rib = Rot(0, 0, -90) * make_rib(length=rib_span, height=rib_span, thickness=gusset_t, tip_flat=1.2)
    rx = width * 0.5 - gusset_t * 0.5
    part = part + Pos(-rx, -wall_t + 0.2, wall_t - 0.2) * rib
    part = part + Pos(+rx, -wall_t + 0.2, wall_t - 0.2) * rib

    # Anti-rotation T-slot tongues along center (X=0) on Z=0 bottom face and Y=0 back face
    if with_tongues:
        t_len = arm_len - 8.0
        # Tongue along Y on Z=0 bottom face
        bot_tongue = Pos(0, -arm_len * 0.5 - 1.0, 0) * Rot(0, 0, 90) * make_tslot_tongue(series, length=t_len, tol=tol)
        # Tongue along Z on Y=0 back face (+Y protrusion)
        back_tongue = Pos(0, 0, arm_len * 0.5 + 1.0) * Rot(90, 0, 90) * make_tslot_tongue(series, length=t_len, tol=tol)
        part = part + bot_tongue + back_tongue

    # T-nut mounting slots / holes along center X=0
    hole_offset = wall_t + (arm_len - wall_t) * 0.56
    slot_cut = Rot(0, 0, 90) * make_slot_cutter(tol, screw, center_span=4.0, depth=wall_t + 6.0)
    h1 = (0.0, -hole_offset, wall_t)
    part = part - Pos(*h1) * slot_cut
    if reg:
        reg.mark("H_BOT", h1, f"底臂{series}型材T型螺母孔", kind="hole", spec=f"{screw}调节槽", normal=(0, 0, 1))

    # Vertical arm T-nut hole (drilled from Y = -wall_t toward +Y)
    h2 = (0.0, -wall_t, hole_offset)
    vert_cut = make_hole_cutter(tol, screw, depth=wall_t + 6.0, kind="clearance")
    part = part - Pos(*h2) * Rot(-90, 0, 0) * vert_cut
    if reg:
        reg.mark("H_BACK", h2, f"立臂{series}型材T型螺母孔", kind="hole", spec=f"{screw}通孔", normal=(0, -1, 0))

    return part


def make_extrusion_mount_plate(
    series: ExtrusionSeries = "2020",
    length_x: float = 56.0,
    width_y: float = 36.0,
    thickness_z: float = 5.0,
    hole_span_x: float = 36.0,
    screw_size: str | None = None,
    tol: ToleranceProfile | None = None,
    reg: FeatureRegistry | None = None,
) -> Part:
    """Flat adapter plate with integrated bottom T-slot alignment tongue and counterbored T-nut holes."""
    tol = tol or ToleranceProfile.for_mode("fdm")
    spec = EXTRUSION_SPECS[series]
    screw = screw_size or spec.default_screw

    plate = Box(length_x, width_y, thickness_z, align=(Align.CENTER, Align.CENTER, Align.MIN))
    plate = safe_fillet(plate, plate.edges().filter_by(Axis.Z), 3.5, reg, "转接板四角圆角")

    tongue = make_tslot_tongue(series=series, length=length_x - 8.0, tol=tol)
    part = plate + tongue

    cb = make_counterbore_cutter(tol, screw, depth=thickness_z + 6.0)
    for idx, sx in enumerate((-1, +1), start=1):
        pos = (sx * hole_span_x * 0.5, 0.0, thickness_z)
        part = part - Pos(*pos) * cb
        if reg:
            reg.mark(f"T{idx}", pos, f"{series}型材锁紧沉头孔#{idx}", kind="hole", spec=screw, normal=(0, 0, 1))
    return part


def make_din_rail_clip(
    width_x: float = 18.0,
    top_deck_y: float = 48.0,
    deck_thick_z: float = 4.5,
    mount_hole_span_y: float = 30.0,
    mount_screw: str = "M3",
    tol: ToleranceProfile | None = None,
    reg: FeatureRegistry | None = None,
) -> Part:
    """Build a standard 35mm Top-Hat DIN Rail (EN 60715 TS35 x 7.5mm) snap-on mounting clip!

    Print orientation tip: Print on its side (-X or +X face on bed) so the compliant latch
    perimeters run continuously along the spring arm for maximum snap strength!
    Geometry orientation here:
      - Top flat equipment mounting deck is at Z = [0, +deck_thick_z], centered at X=0, Y=0.
      - DIN rail channel (35.0mm wide along Y) recesses underneath in -Z (`Z < 0`), with:
        * Fixed upper catch hook at +Y side (`Y ≈ +17.5mm`)
        * Cantilever spring snap latch + flathead screwdriver release tab at -Y side (`Y ≈ -17.5mm`).
    """
    tol = tol or ToleranceProfile.for_mode("fdm")
    gap = tol.clearance("snug")
    rail_span = 35.0 + 2.0 * gap     # ~35.24mm in FDM for crisp rattle-free snap
    rail_lip_t = 1.15 + gap          # Standard TS35 steel flange thickness is 1.0mm
    jaw_depth_z = 5.5

    half_r = rail_span * 0.5

    # 2D profile in YZ plane, extruded symmetrically along X by width_x!
    # Upper fixed hook (+Y side) + Top deck + Lower compliant spring arm (-Y side)
    y_top_outer = max(top_deck_y * 0.5, half_r + 5.0)
    y_bot_outer = -max(top_deck_y * 0.5, half_r + 6.5)

    pts_yz = [
        # Top deck upper surface (Z = deck_thick_z)
        (y_bot_outer, deck_thick_z),
        (y_top_outer, deck_thick_z),
        # Fixed hook on +Y side
        (y_top_outer, -jaw_depth_z),
        (half_r - 2.1, -jaw_depth_z),
        (half_r - 2.1, -(jaw_depth_z - 1.8)),
        (half_r + 0.2, -(jaw_depth_z - 1.8)),
        (half_r + 0.2, 0.0),
        # Rail backstop ceiling (Z = 0) across to compliant spring root
        (-half_r + 6.0, 0.0),
        # Compliant relief slot roof -> flexible spring beam along -Y!
        (-half_r + 6.0, 1.8),
        (-half_r - 2.2, 1.8),
        (-half_r - 2.2, -jaw_depth_z - 1.5),   # Pry tab extension
        (-half_r - 5.2, -jaw_depth_z - 1.5),   # Screwdriver release tab tip
        (-half_r - 5.2, -jaw_depth_z + 0.5),
        (-half_r - 3.8, -jaw_depth_z + 0.5),
        (-half_r - 3.8, 3.4),                   # Flex beam outer back
        (y_bot_outer, 3.4),
    ]
    body = extrude(Plane.YZ * Polygon(*pts_yz, align=None), amount=width_x * 0.5, both=True)

    # Add the angled snap tooth catch on the inside of the lower spring arm at Y = -half_r
    tooth_pts = [
        (-half_r - 2.3, -(rail_lip_t + 0.2)),
        (-half_r + 1.4, -(rail_lip_t + 0.5)),  # Retaining shelf catching 35mm rail edge
        (-half_r - 2.3, -jaw_depth_z),         # Smooth 40-deg entry ramp!
    ]
    tooth = extrude(Plane.YZ * Polygon(*tooth_pts, align=None), amount=width_x * 0.5, both=True)
    part = body + tooth

    # Add 2 equipment mounting holes (heat-set or clearance) on the top deck
    hs = make_heatset_cutter(tol, mount_screw, screw_relief_depth=1.5)
    for idx, sy in enumerate((-1, +1), start=1):
        py = sy * mount_hole_span_y * 0.5
        if abs(py) < y_top_outer - 4.0:
            hpos = (0.0, py, deck_thick_z)
            part = part - Pos(*hpos) * hs
            if reg:
                reg.mark(f"D{idx}", hpos, f"导轨卡扣设备安装孔#{idx}", kind="hole", spec=f"{mount_screw}热熔/攻丝", normal=(0, 0, 1))

    if reg:
        reg.mark("RAIL_35", (0.0, 0.0, -1.0), "35mm欧标DIN导轨卡槽中心", kind="mate", spec=f"开口{rail_span:.2f}mm", normal=(0, 0, -1))
    return part
