"""Mechanical Power Transmission & Bearing Mount Templates (渐开线正齿轮 / D型轴孔 / 轴承座)."""

from __future__ import annotations

import math

from build123d import Align, Axis, Box, Cylinder, Part, Plane, Polygon, Pos, Rot, extrude

from ..features import (
    FeatureRegistry,
    make_counterbore_cutter,
    make_hole_cutter,
    make_nut_trap_cutter,
    safe_chamfer,
    safe_fillet,
)
from ..tolerances import ToleranceProfile


BEARING_TABLE: dict[str, tuple[float, float, float]] = {
    # code: (outer_d, inner_d, width_b) in mm
    "608":  (22.0, 8.0, 7.0),
    "625":  (16.0, 5.0, 5.0),
    "624":  (13.0, 4.0, 5.0),
    "623":  (10.0, 3.0, 4.0),
    "MR85": (8.0,  5.0, 2.5),
    "6000": (26.0, 10.0, 8.0),
}


def make_d_shaft_cutter(
    shaft_d: float = 5.0,
    flat_d: float = 4.5,
    depth: float = 20.0,
    tol: ToleranceProfile | None = None,
    fit: str = "snug",
    overcut: float = 2.0,
) -> Part:
    """Create a motor D-shaft bore cutter entering at Z=0 down to -depth (flat faces +X).

    Example: Standard NEMA17 stepper shaft has shaft_d=5.0, flat_d=4.5.
    """
    tol = tol or ToleranceProfile.for_mode("fdm")
    gap = tol.clearance(fit)  # type: ignore[arg-type]
    r = shaft_d * 0.5 + gap
    flat_x = (flat_d - shaft_d * 0.5) + gap

    cyl = Pos(0, 0, overcut) * Cylinder(
        radius=r,
        height=depth + overcut,
        align=(Align.CENTER, Align.CENTER, Align.MAX),
    )
    # Remove the flat chord zone (x > flat_x) from the cylinder cutter so solid retains the D-key flat!
    chord_keep = Pos(flat_x, 0, overcut + 1.0) * Box(
        r * 2.0,
        r * 2.5,
        depth + overcut + 2.0,
        align=(Align.MIN, Align.CENTER, Align.MAX),
    )
    return cyl - chord_keep


def _involute_xy(base_r: float, t: float) -> tuple[float, float]:
    x = base_r * (math.cos(t) + t * math.sin(t))
    y = base_r * (math.sin(t) - t * math.cos(t))
    return x, y


def make_spur_gear(
    module: float = 1.5,
    teeth: int = 24,
    thickness: float = 6.0,
    pressure_angle_deg: float = 20.0,
    bore_d: float = 5.0,
    d_shaft_flat: float = 0.0,
    hub_d: float = 0.0,
    hub_h: float = 0.0,
    set_screw: str | None = None,
    tol: ToleranceProfile | None = None,
    reg: FeatureRegistry | None = None,
) -> Part:
    """Generate a true involute profile spur gear with optional D-bore, hub, and captive M3 nut set-screw!"""
    if teeth < 6:
        raise ValueError("Spur gear teeth count must be >= 6")
    tol = tol or ToleranceProfile.for_mode("fdm")
    # Apply slight backlash radial inset in FDM so meshed printed gears spin freely without binding
    backlash_ang = (0.08 / (module * teeth)) if tol.mode == "fdm" else 0.0

    alpha = math.radians(pressure_angle_deg)
    pitch_r = 0.5 * module * teeth
    addendum_r = pitch_r + module
    dedendum_r = max(0.8, pitch_r - 1.25 * module)
    base_r = pitch_r * math.cos(alpha)

    # Angular half-tooth thickness at pitch circle
    inv_alpha = math.tan(alpha) - alpha
    half_tooth_pitch_ang = (math.pi / (2.0 * teeth)) - backlash_ang
    base_half_ang = half_tooth_pitch_ang + inv_alpha

    # Sample involute flank from max(base_r, dedendum_r) to addendum_r
    t_max = math.sqrt(max(0.0, (addendum_r / base_r) ** 2 - 1.0))
    t_min = math.sqrt(max(0.0, (max(base_r, dedendum_r) / base_r) ** 2 - 1.0))
    samples = 9
    right_flank: list[tuple[float, float]] = []
    if dedendum_r < base_r:
        right_flank.append((dedendum_r * math.cos(-base_half_ang), dedendum_r * math.sin(-base_half_ang)))
    for i in range(samples + 1):
        t = t_min + (t_max - t_min) * (i / samples)
        ix, iy = _involute_xy(base_r, t)
        # Rotate by -base_half_ang, mirror Y? Standard involute curves +Y as t grows;
        # rotating by -base_half_ang and flipping Y tapers inward toward tooth centerline (+X)!
        c, s = math.cos(base_half_ang), math.sin(base_half_ang)
        rx = ix * c + iy * s
        ry = -ix * s + iy * c
        # Wait: at t=0, (ix,iy)=(base_r, 0) -> (rx, ry)=(base_r*cos(b), -base_r*sin(b))!
        # Let's check: at pitch circle (t = tan(alpha)), polar angle of (ix, iy) is inv_alpha,
        # so rx_polar = -base_half_ang + inv_alpha = -half_tooth_pitch_ang! Exact!
        rx2 = ix * math.cos(-base_half_ang) - iy * math.sin(-base_half_ang)
        ry2 = ix * math.sin(-base_half_ang) + iy * math.cos(-base_half_ang)
        if math.atan2(ry2, rx2) <= 0.0:
            right_flank.append((rx2, ry2))

    # Mirror right_flank across X-axis (y -> -y) to get left_flank in CCW order!
    left_flank = [(x, -y) for (x, y) in reversed(right_flank)]
    single_tooth = right_flank + left_flank

    # Replicate all N teeth around 360 degrees into one closed 2D polygon
    all_pts: list[tuple[float, float]] = []
    step_ang = 2.0 * math.pi / teeth
    for k in range(teeth):
        ang = k * step_ang
        ca, sa = math.cos(ang), math.sin(ang)
        for px, py in single_tooth:
            all_pts.append((px * ca - py * sa, px * sa + py * ca))

    gear = extrude(Polygon(*all_pts, align=None), amount=thickness)

    total_h = thickness
    if hub_d > 0 and hub_h > 0:
        hub = Pos(0, 0, thickness - 0.2) * Cylinder(
            radius=hub_d * 0.5,
            height=hub_h + 0.2,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        gear = gear + hub
        total_h = thickness + hub_h

    if bore_d > 0:
        if d_shaft_flat > 0:
            cutter = Pos(0, 0, total_h) * make_d_shaft_cutter(
                shaft_d=bore_d, flat_d=d_shaft_flat, depth=total_h + 4.0, tol=tol
            )
            spec_str = f"D轴 Ø{bore_d}/{d_shaft_flat}mm"
        else:
            r_bore = bore_d * 0.5 + tol.clearance("snug")
            cutter = Pos(0, 0, -1.0) * Cylinder(
                radius=r_bore, height=total_h + 2.0, align=(Align.CENTER, Align.CENTER, Align.MIN)
            )
            spec_str = f"圆孔 Ø{2*r_bore:.2f}mm"
        gear = gear - cutter
        if reg:
            reg.mark("BORE", (0, 0, total_h), "齿轮中心轴孔", kind="hole", spec=spec_str, normal=(0, 0, 1))

    if set_screw and hub_d > bore_d + 6.0 and hub_h >= 6.0:
        sz_z = thickness + hub_h * 0.5
        nut_x = 0.5 * (bore_d * 0.5 + hub_d * 0.5)
        trap = Pos(nut_x, 0, sz_z) * Rot(0, 90, 0) * make_nut_trap_cutter(
            tol, set_screw, bolt_depth=hub_d, side_slot_len=hub_h
        )
        gear = gear - trap
        if reg:
            reg.mark("H_SET", (hub_d * 0.5, 0, sz_z), f"{set_screw}顶丝与防转螺母槽", kind="hole", spec=set_screw, normal=(1, 0, 0))

    if reg:
        reg.mark("PITCH", (pitch_r, 0, thickness * 0.5), f"分度圆 (模数m={module}, 齿数z={teeth})", kind="datum", spec=f"分度圆Ø{2*pitch_r:.1f}")
    return gear


def make_bearing_block(
    bearing_code: str = "608",
    shaft_center_z: float = 18.0,
    base_len_x: float = 54.0,
    block_thick_y: float = 12.0,
    base_thick_z: float = 5.5,
    mount_screw: str = "M4",
    tol: ToleranceProfile | None = None,
    reg: FeatureRegistry | None = None,
) -> Part:
    """Create a bolt-down pillow block housing for standard ball bearings (608, 625, MR85, 6000)."""
    code = bearing_code.upper()
    if code not in BEARING_TABLE:
        raise ValueError(f"Unknown bearing '{bearing_code}'. Supported: {list(BEARING_TABLE.keys())}")
    od, id_d, b_wid = BEARING_TABLE[code]
    tol = tol or ToleranceProfile.for_mode("fdm")

    fit_r = od * 0.5 + tol.clearance("press")
    housing_r = fit_r + max(4.0, od * 0.22)
    actual_thick_y = max(block_thick_y, b_wid + 2.5)

    base = Box(base_len_x, actual_thick_y, base_thick_z, align=(Align.CENTER, Align.CENTER, Align.MIN))
    arch_cyl = Pos(0, 0, shaft_center_z) * Rot(90, 0, 0) * Cylinder(
        radius=housing_r, height=actual_thick_y, align=(Align.CENTER, Align.CENTER, Align.CENTER)
    )
    pedestal = Box(housing_r * 2.0, actual_thick_y, shaft_center_z, align=(Align.CENTER, Align.CENTER, Align.MIN))
    part = base + arch_cyl + pedestal

    # Trim anything below Z=0 if shaft_center_z < housing_r
    trim_below = Pos(0, 0, 0) * Box(base_len_x * 2, actual_thick_y * 2, 50.0, align=(Align.CENTER, Align.CENTER, Align.MAX))
    part = part - trim_below

    # Cut bearing press-fit pocket from front (-Y) to depth b_wid, plus inner retaining lip thru-hole!
    y_front = -actual_thick_y * 0.5
    pocket = Pos(0, y_front - 1.0, shaft_center_z) * Rot(-90, 0, 0) * Cylinder(
        radius=fit_r, height=b_wid + 1.0, align=(Align.CENTER, Align.CENTER, Align.MIN)
    )
    lip_r = od * 0.5 - 1.4
    thru = Pos(0, 0, shaft_center_z) * Rot(90, 0, 0) * Cylinder(
        radius=lip_r, height=actual_thick_y + 4.0, align=(Align.CENTER, Align.CENTER, Align.CENTER)
    )
    part = (part - pocket) - thru

    # Base mounting holes
    hx = 0.5 * (housing_r + base_len_x * 0.5)
    cb = make_counterbore_cutter(tol, mount_screw, depth=base_thick_z + 4.0)
    for idx, sx in enumerate((-1, +1), start=1):
        hpos = (sx * hx, 0.0, base_thick_z)
        part = part - Pos(*hpos) * cb
        if reg:
            reg.mark(f"H{idx}", hpos, f"轴承座底脚安装孔#{idx}", kind="hole", spec=mount_screw, normal=(0, 0, 1))

    if reg:
        reg.mark("BRG", (0, y_front, shaft_center_z), f"{code}轴承压入腔", kind="hole", spec=f"Ø{2*fit_r:.2f}×深{b_wid}mm", normal=(0, -1, 0))
    return part
