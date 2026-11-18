"""Parametric Split Electronics Enclosure Template (上下分体子母止口电子外壳母版).

Features:
  - Interlocking Lip & Groove seam compensated automatically for FDM / CNC / Exact clearance
  - 4 corner wall-integrated screw bosses (brass heat-set / tap in bottom, SHCS counterbore in lid)
  - Optional custom PCB mounting standoffs on floor
  - Optional side cooling ventilation louvers
  - Optional wall-mounting flange ears with elongated M4 slots
  - Helper `cut_panel_port` for 1-line USB-C, DC barrel jack, SMA, or rectangular screen cutouts
  - Layouts:
      * 'print_plate': Bottom box & upside-down Top lid arranged side-by-side on Z=0 (100% support-free!)
      * 'assembled'  : Lid positioned above bottom box (optional explode_z gap for inspection)
      * 'bottom'     : Bottom enclosure half only
      * 'lid'        : Top lid half only (oriented flat on Z=0 for printing or upright)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from build123d import Align, Axis, Box, Cylinder, Part, Pos, Rot, SlotOverall, extrude

from ..features import (
    FeatureRegistry,
    make_counterbore_cutter,
    make_heatset_cutter,
    make_hole_cutter,
    make_mounting_boss,
    make_slot_cutter,
    safe_chamfer,
    safe_fillet,
)
from ..tolerances import ToleranceProfile

EnclosureLayout = Literal["print_plate", "assembled", "bottom", "lid"]
PortKind = Literal["usb_c", "dc_jack", "sma", "rect", "circle"]


@dataclass
class EnclosureParams:
    outer_x: float = 86.0              # 外壳左右总长 (X)
    outer_y: float = 58.0              # 外壳前后总宽 (Y)
    bottom_h: float = 22.0             # 下壳高度 (不含止口凸起, Z)
    lid_h: float = 8.0                 # 上盖外露高度 (Z)
    wall_t: float = 2.8                # 侧壁厚度 (建议 >= 2.4mm 以容纳子母止口)
    floor_t: float = 2.4               # 底板与顶盖板厚度
    corner_r: float = 5.0              # 四角外圆角半径
    lip_h: float = 2.2                 # 上下盖咬合止口高度
    screw_size: str = "M3"             # 四角锁紧螺丝规格
    flange_ears: bool = False          # 是否生成左右两侧挂墙安装耳片
    flange_ear_ext: float = 11.0       # 单侧挂墙耳片外伸宽度
    flange_screw: str = "M4"           # 挂墙耳片腰型孔规格
    vent_slots: int = 5                # 左右侧壁散热条孔数量 (0=关闭)
    vent_len_z: float = 10.0           # 散热槽竖直长度
    vent_wid_y: float = 2.2            # 散热槽宽度
    pcb_standoffs: list[tuple[float, float]] = field(default_factory=list)  # PCB孔相对中心(x,y)列表
    pcb_standoff_h: float = 5.0        # PCB支撑柱距内底面高度
    pcb_screw: str = "M2.5"            # PCB固定孔规格


def _rounded_rect_prism(lx: float, ly: float, hz: float, rad: float, z0: float = 0.0) -> Part:
    b = Pos(0, 0, z0) * Box(lx, ly, hz, align=(Align.CENTER, Align.CENTER, Align.MIN))
    r = min(rad, min(lx, ly) * 0.45)
    if r > 0.15:
        b = safe_fillet(b, b.edges().filter_by(Axis.Z), r)
    return b


def make_split_enclosure(
    params: EnclosureParams | None = None,
    tol: ToleranceProfile | None = None,
    reg: FeatureRegistry | None = None,
    layout: EnclosureLayout = "print_plate",
    explode_z: float = 10.0,
) -> Part:
    """Build parametric bottom enclosure, top lid, or both."""
    p = params or EnclosureParams()
    tol = tol or ToleranceProfile.for_mode("fdm")
    gap = tol.clearance("snug")

    inner_x = p.outer_x - 2.0 * p.wall_t
    inner_y = p.outer_y - 2.0 * p.wall_t
    inner_r = max(0.8, p.corner_r - p.wall_t)

    # Lip split geometry: bottom has inner raised lip; lid has outer skirt + inner recess step
    half_wall = p.wall_t * 0.5
    lip_outer_x = p.outer_x - 2.0 * (half_wall + gap * 0.5)
    lip_outer_y = p.outer_y - 2.0 * (half_wall + gap * 0.5)
    lip_outer_r = max(0.6, p.corner_r - (half_wall + gap * 0.5))

    groove_inner_x = p.outer_x - 2.0 * (half_wall - gap * 0.5)
    groove_inner_y = p.outer_y - 2.0 * (half_wall - gap * 0.5)
    groove_inner_r = max(0.6, p.corner_r - (half_wall - gap * 0.5))

    # Corner boss centers tucked snugly inside corners
    boss_d = 8.4 if p.screw_size.upper() == "M3" else 7.2
    boss_inset = max(p.corner_r * 0.85, p.wall_t + boss_d * 0.42)
    bx = p.outer_x * 0.5 - boss_inset
    by = p.outer_y * 0.5 - boss_inset
    corner_xy = [(-bx, -by), (+bx, -by), (-bx, +by), (+bx, +by)]

    build_bottom = layout in ("print_plate", "assembled", "bottom")
    build_lid = layout in ("print_plate", "assembled", "lid")

    bottom_part: Part | None = None
    if build_bottom:
        outer_shell = _rounded_rect_prism(p.outer_x, p.outer_y, p.bottom_h, p.corner_r, z0=0.0)
        lip_shell = _rounded_rect_prism(lip_outer_x, lip_outer_y, p.lip_h + 1.0, lip_outer_r, z0=p.bottom_h - 1.0)
        cavity = _rounded_rect_prism(inner_x, inner_y, p.bottom_h + p.lip_h + 2.0, inner_r, z0=p.floor_t)
        bottom_part = (outer_shell + lip_shell) - cavity

        # Add 4 corner screw columns inside the cavity up to Z = p.bottom_h (flush with split line)
        col_h = p.bottom_h - p.floor_t + 0.4
        for idx, (cx, cy) in enumerate(corner_xy, start=1):
            col = Pos(cx, cy, p.floor_t - 0.4) * Cylinder(
                radius=boss_d * 0.5,
                height=col_h,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
            )
            # Bridge ribs tying corner column cleanly into X and Y walls so there is zero narrow gap
            tie_x = Pos(cx + (-2.5 if cx < 0 else 2.5), cy, p.floor_t - 0.4) * Box(
                5.0, boss_d * 0.85, col_h, align=(Align.CENTER, Align.CENTER, Align.MIN)
            )
            tie_y = Pos(cx, cy + (-2.5 if cy < 0 else 2.5), p.floor_t - 0.4) * Box(
                boss_d * 0.85, 5.0, col_h, align=(Align.CENTER, Align.CENTER, Align.MIN)
            )
            bottom_part = bottom_part + col + tie_x + tie_y

        # Trim anything protruding outside outer_x / outer_y due to corner ties
        keep_env = _rounded_rect_prism(p.outer_x, p.outer_y, p.bottom_h + p.lip_h + 5.0, p.corner_r, z0=-1.0)
        bottom_part = bottom_part & keep_env

        # Optional wall-mounting flange ears on Left (-X) & Right (+X) at Z=0
        if p.flange_ears:
            ear_t = max(3.6, p.floor_t + 1.2)
            total_ear_x = p.outer_x + 2.0 * p.flange_ear_ext
            ear_wid_y = min(p.outer_y * 0.72, 38.0)
            ear_plate = _rounded_rect_prism(total_ear_x, ear_wid_y, ear_t, 4.0, z0=0.0)
            # Re-cut interior cavity so ear plate doesn't fill inside of box
            bottom_part = (bottom_part + ear_plate) - cavity
            slot_cut = Rot(0, 0, 90) * make_slot_cutter(tol, p.flange_screw, center_span=4.0, depth=ear_t + 4.0)
            ex = p.outer_x * 0.5 + p.flange_ear_ext * 0.52
            for eid, sx in [("E1", -1), ("E2", +1)]:
                epos = (sx * ex, 0.0, ear_t)
                if reg:
                    reg.mark(eid, epos, f"{'左' if sx < 0 else '右'}侧挂墙腰型耳孔", kind="hole", spec=p.flange_screw, normal=(0, 0, 1))
                bottom_part = bottom_part - Pos(*epos) * slot_cut

        # Drill 4 corner brass heat-set / tap holes at Z = p.bottom_h
        hs_cut = make_heatset_cutter(tol, p.screw_size, screw_relief_depth=4.0)
        for idx, (cx, cy) in enumerate(corner_xy, start=1):
            if reg:
                reg.mark(f"H{idx}", (cx, cy, p.bottom_h), f"下壳角柱孔#{idx}", kind="hole", spec=f"{p.screw_size}热熔/攻丝", normal=(0, 0, 1))
            bottom_part = bottom_part - Pos(cx, cy, p.bottom_h) * hs_cut

        # Optional PCB standoffs on interior floor (Z = p.floor_t)
        if p.pcb_standoffs:
            sb_solid = make_mounting_boss(outer_d=6.6, height=p.pcb_standoff_h, base_flare=1.2, flare_height=1.6)
            sb_hole = make_hole_cutter(tol, p.pcb_screw, depth=p.pcb_standoff_h + 1.2, kind="tap")
            for pidx, (px, py) in enumerate(p.pcb_standoffs, start=1):
                z_top = p.floor_t + p.pcb_standoff_h
                bottom_part = bottom_part + Pos(px, py, p.floor_t - 0.2) * sb_solid
                bottom_part = bottom_part - Pos(px, py, z_top) * sb_hole
                if reg:
                    reg.mark(f"PCB{pidx}", (px, py, z_top), f"PCB支撑柱#{pidx}", kind="boss", spec=p.pcb_screw, normal=(0, 0, 1))

        # Optional Left/Right side ventilation louvers
        if p.vent_slots > 0 and p.bottom_h >= p.floor_t + p.vent_len_z + 5.0:
            pitch_y = (p.vent_wid_y * 2.2)
            start_y = -0.5 * (p.vent_slots - 1) * pitch_y
            vz_center = p.floor_t + (p.bottom_h - p.floor_t) * 0.52
            slot_sketch = SlotOverall(width=p.vent_len_z, height=p.vent_wid_y)
            # Cut through both -X and +X walls cleanly (avoiding corner bosses)
            if abs(start_y) + p.vent_wid_y < by - boss_d * 0.65:
                v_cutter_single = Rot(0, 90, 0) * extrude(slot_sketch, amount=p.outer_x * 0.6, both=True)
                for k in range(p.vent_slots):
                    yk = start_y + k * pitch_y
                    bottom_part = bottom_part - Pos(0.0, yk, vz_center) * v_cutter_single
                if reg:
                    reg.mark("V1", (-p.outer_x * 0.5, 0.0, vz_center), "侧壁散热百叶槽组", kind="slot", spec=f"{p.vent_slots}槽", normal=(-1, 0, 0))

        if tol.bed_chamfer > 0:
            bf = bottom_part.faces().sort_by(Axis.Z)[0]
            bottom_part = safe_chamfer(bottom_part, bf.outer_wire().edges(), tol.bed_chamfer, reg, "下壳底边倒角")

    lid_part: Part | None = None
    if build_lid:
        # Build lid in print-ready upside-down orientation FIRST:
        # Top exterior rests flat at Z = 0, lid side skirt rises to Z = +p.lid_h!
        lid_outer = _rounded_rect_prism(p.outer_x, p.outer_y, p.lid_h, p.corner_r, z0=0.0)
        lid_cavity = _rounded_rect_prism(inner_x, inner_y, p.lid_h + 2.0, inner_r, z0=p.floor_t)
        # Interlocking groove step cut at the rim (from Z = p.lid_h - groove_depth to Z = p.lid_h + 1)
        groove_depth = min(p.lip_h + 0.25, p.lid_h - p.floor_t - 0.6)
        groove_cut = _rounded_rect_prism(
            groove_inner_x,
            groove_inner_y,
            groove_depth + 1.0,
            groove_inner_r,
            z0=p.lid_h - groove_depth,
        )
        lid_part = (lid_outer - lid_cavity) - groove_cut

        # Add 4 corner screw bosses inside lid up to Z = p.lid_h - groove_depth (or mating plane Z = p.lid_h!)
        # Wait: corner bosses sit INSIDE inner_x/inner_y or overlap lip? If boss inset puts them inside lip,
        # they rise all the way to Z = p.lid_h to clamp directly onto bottom_part's corner columns!
        lid_col_h = p.lid_h - p.floor_t + 0.2
        for cx, cy in corner_xy:
            col = Pos(cx, cy, p.floor_t - 0.2) * Cylinder(
                radius=(boss_d - 0.4) * 0.5,
                height=lid_col_h,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
            )
            lid_part = lid_part + col
        # Re-clear the groove ring only where lip enters (around perimeter outside lip inner wall)
        lip_ring_clear = groove_cut - _rounded_rect_prism(
            inner_x - 0.2, inner_y - 0.2, groove_depth + 2.0, inner_r, z0=p.lid_h - groove_depth - 0.5
        )
        lid_part = lid_part - lip_ring_clear

        # Drill 4 corner M3 counterbores from exterior top face (currently at Z=0, drilling +Z!)
        cb_cut = make_counterbore_cutter(
            tol,
            p.screw_size,
            depth=p.lid_h + 4.0,
            upside_down_bridge=(layout == "print_plate" or layout == "lid"),
        )
        for idx, (cx, cy) in enumerate(corner_xy, start=1):
            # cb_cut enters at Z=0 going -Z; rotate 180 deg around X so counterbore head is at Z=0 going +Z!
            lid_part = lid_part - Pos(cx, cy, 0.0) * Rot(180, 0, 0) * cb_cut

        if tol.bed_chamfer > 0:
            lf = lid_part.faces().sort_by(Axis.Z)[0]
            lid_part = safe_chamfer(lid_part, lf.outer_wire().edges(), tol.bed_chamfer, reg, "上盖顶边倒角")

    if layout == "bottom":
        assert bottom_part is not None
        return bottom_part
    if layout == "lid":
        assert lid_part is not None
        return lid_part
    if layout == "print_plate":
        assert bottom_part is not None and lid_part is not None
        shift_y = p.outer_y + 12.0
        placed_lid = Pos(0.0, shift_y, 0.0) * lid_part
        if reg:
            reg.expected_solids = 2
            reg.mark("LID", (0.0, shift_y, p.lid_h), "分体上盖(已翻平免支撑排版)", kind="datum", spec=f"高{p.lid_h}mm", normal=(0, 0, 1))
        return bottom_part + placed_lid
    if layout == "assembled":
        assert bottom_part is not None and lid_part is not None
        # Flip lid 180 deg around X so Z=0 exterior becomes top Z = bottom_h + explode_z + lid_h
        z_lid_top = p.bottom_h + explode_z + p.lid_h
        placed_lid = Pos(0.0, 0.0, z_lid_top) * Rot(180, 0, 0) * lid_part
        if reg:
            reg.expected_solids = 2 if explode_z > 0.01 else 0
            reg.mark("LID", (0.0, 0.0, z_lid_top), "装配态上盖", kind="datum", spec=f"爆炸间距{explode_z}mm", normal=(0, 0, 1))
        return bottom_part + placed_lid
    raise ValueError(f"Unknown enclosure layout: {layout}")


def cut_panel_port(
    part: Part,
    wall: Literal["-Y", "+Y", "-X", "+X"] = "-Y",
    port_type: PortKind = "usb_c",
    offset_along_wall: float = 0.0,
    center_z: float = 10.0,
    custom_size: tuple[float, float] = (12.0, 6.0),
    wall_coord: float | None = None,
    tol: ToleranceProfile | None = None,
    reg: FeatureRegistry | None = None,
    anchor_id: str = "P_IO",
    desc: str = "",
) -> Part:
    """Cut a standard I/O connector opening (USB-C with plug relief, DC jack, SMA, rect, circle) on a vertical wall."""
    tol = tol or ToleranceProfile.for_mode("fdm")
    gap2 = 2.0 * tol.clearance("snug")
    bb = part.bounding_box()

    if wall == "-Y":
        wc = wall_coord if wall_coord is not None else bb.min.Y
        pos = (offset_along_wall, wc, center_z)
        rot = Rot(-90, 0, 0)   # Cutter -Z points toward +Y (into wall)
        nrm = (0, -1, 0)
    elif wall == "+Y":
        wc = wall_coord if wall_coord is not None else bb.max.Y
        pos = (offset_along_wall, wc, center_z)
        rot = Rot(90, 0, 0)    # Cutter -Z points toward -Y
        nrm = (0, 1, 0)
    elif wall == "-X":
        wc = wall_coord if wall_coord is not None else bb.min.X
        pos = (wc, offset_along_wall, center_z)
        rot = Rot(0, 90, 0) * Rot(0, 0, 90)
        nrm = (-1, 0, 0)
    elif wall == "+X":
        wc = wall_coord if wall_coord is not None else bb.max.X
        pos = (wc, offset_along_wall, center_z)
        rot = Rot(0, -90, 0) * Rot(0, 0, -90)
        nrm = (1, 0, 0)
    else:
        raise ValueError(f"Unsupported wall '{wall}'")

    if port_type == "usb_c":
        # Inner tongue opening 9.2 x 3.4 + outer overmold recess 12.8 x 6.8 (1.4mm deep into outer wall)
        inner = Pos(0, 0, 2.0) * extrude(SlotOverall(9.2 + gap2, 3.4 + gap2), amount=-12.0)
        recess = Pos(0, 0, 2.0) * extrude(SlotOverall(12.8 + gap2, 6.8 + gap2), amount=-3.4)
        cutter = inner + recess
        spec_txt = "USB-C带插头避让槽"
    elif port_type == "dc_jack":
        d = 8.2 + gap2
        cutter = Pos(0, 0, 2.0) * Cylinder(radius=d * 0.5, height=12.0, align=(Align.CENTER, Align.CENTER, Align.MAX))
        spec_txt = f"DC电源孔 Ø{d:.1f}"
    elif port_type == "sma":
        d = 6.5 + gap2
        cutter = Pos(0, 0, 2.0) * Cylinder(radius=d * 0.5, height=12.0, align=(Align.CENTER, Align.CENTER, Align.MAX))
        spec_txt = f"SMA天线孔 Ø{d:.1f}"
    elif port_type == "circle":
        d = custom_size[0] + gap2
        cutter = Pos(0, 0, 2.0) * Cylinder(radius=d * 0.5, height=12.0, align=(Align.CENTER, Align.CENTER, Align.MAX))
        spec_txt = f"圆孔 Ø{d:.1f}"
    elif port_type == "rect":
        w, h = custom_size[0] + gap2, custom_size[1] + gap2
        cutter = Pos(0, 0, 2.0) * Box(w, h, 12.0, align=(Align.CENTER, Align.CENTER, Align.MAX))
        spec_txt = f"方窗 {w:.1f}×{h:.1f}"
    else:
        raise ValueError(f"Unknown port_type '{port_type}'")

    if reg:
        reg.mark(anchor_id, pos, desc or spec_txt, kind="cutout", spec=spec_txt, normal=nrm)
    return part - Pos(*pos) * rot * cutter
