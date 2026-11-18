"""Demo Structural Part: Reinforced Sensor L-Bracket (传感器L型加强安装支架)

Demonstrates:
  1. Referencing an existing STEP component (`refs/sensor_module.step`) with zero-collision clearance
  2. Dual-mode tolerances (`fdm` heat-set inserts + elephant-foot chamfer vs `cnc` metric tap drills)
  3. Semantic Feature ID anchors (`H1..H4`, `B1..B2`, `S1`, `R1..R2`) projected onto 10-view & 3D HTML
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from build123d import Align, Axis, Box, Part, Pos, Rot

from cad_kit import (
    FeatureRegistry,
    ManufacturingMode,
    ToleranceProfile,
    make_counterbore_cutter,
    make_heatset_cutter,
    make_mounting_boss,
    make_rib,
    make_slot_cutter,
    safe_chamfer,
    safe_fillet,
)


@dataclass
class Params:
    """All tunable bracket dimensions in millimeters (mm)."""

    base_len_x: float = 52.0         # 底板左右总长 (X)
    base_wid_y: float = 42.0         # 底板前后总深 (Y, 预留背板与PCB后缘1mm装配间隙)
    base_thick_z: float = 4.5        # 底板厚度 (Z)
    back_wall_h: float = 32.0        # 背面立板总高 (Z, 加高至32mm容纳双上安装孔)
    back_wall_t: float = 5.0         # 背面立板厚度 (Y)
    boss_span_x: float = 26.0        # 传感器支撑柱左右孔距 (±13.0, 对应参考件孔位)
    boss_pos_y: float = 5.0          # 传感器支撑柱前后坐标 (Y=+5.0, 对应参考件孔位)
    boss_top_z: float = 12.0         # 支撑柱顶面高度 (托举 Z=12.0 的传感器PCB)
    boss_outer_d: float = 9.0        # 热熔螺母支撑柱外径
    front_slot_w: float = 18.0       # 正面走线避空开口宽度 (放宽至18mm方便插拔插头)
    front_slot_d: float = 15.0       # 正面走线开口向内深度
    base_slot_x: float = 19.0        # 底座固定腰型孔 X 坐标 (±19.0)
    base_slot_y: float = -10.0       # 底座固定腰型孔 Y 坐标
    base_slot_span: float = 4.5      # 腰型孔调节行程
    rib_len_y: float = 16.0          # 左右三角加强筋前伸长度
    rib_h_z: float = 18.0            # 左右三角加强筋高度
    rib_thick: float = 3.2           # 加强筋厚度
    back_hole_z: float = 20.0        # 背板中心沉头安装孔高度
    corner_fillet: float = 3.5       # 底板前侧外圆角


def build(
    params: Params | None = None,
    mode: ManufacturingMode = "fdm",
) -> tuple[Part, FeatureRegistry, Params]:
    p = params or Params()
    tol = ToleranceProfile.for_mode(mode)
    reg = FeatureRegistry(
        part_name="demo_sensor_bracket",
        mode=mode,
        description="传感器模块L型加强安装支架 (含热熔螺母柱、走线避空槽、背板沉头孔与三角加强筋)",
    )

    ref_step = Path(__file__).parent / "refs" / "sensor_module.step"
    if ref_step.exists():
        reg.add_ref(str(ref_step))

    y_back = p.base_wid_y / 2.0
    y_front = -p.base_wid_y / 2.0
    x_half = p.base_len_x / 2.0

    # 1. 底板 + 背面立板 (L型主体)
    base = Box(
        p.base_len_x,
        p.base_wid_y,
        p.base_thick_z,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )
    back_wall = Pos(0, y_back - p.back_wall_t / 2.0, 0) * Box(
        p.base_len_x,
        p.back_wall_t,
        p.back_wall_h,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )
    part = base + back_wall

    # 2. 左右两侧直角三角加强筋 (R1, R2) — 从背板向前(-Y)伸出
    rib_y_anchor = y_back - p.back_wall_t + 0.2
    rib_z_anchor = p.base_thick_z - 0.2
    # make_rib 默认在 XZ 平面 (+X, +Z)，绕 Z 轴转 -90 度使其沿 -Y (向前) 延伸！
    rib_shape = Rot(0, 0, -90) * make_rib(
        length=p.rib_len_y,
        height=p.rib_h_z,
        thickness=p.rib_thick,
    )
    rx_offset = x_half - p.rib_thick / 2.0 - 1.5
    reg.mark("R1", (-rx_offset, rib_y_anchor - p.rib_len_y * 0.4, rib_z_anchor + p.rib_h_z * 0.4), "左侧三角加强筋", kind="rib", spec=f"厚{p.rib_thick}mm")
    reg.mark("R2", (+rx_offset, rib_y_anchor - p.rib_len_y * 0.4, rib_z_anchor + p.rib_h_z * 0.4), "右侧三角加强筋", kind="rib", spec=f"厚{p.rib_thick}mm")
    part = part + Pos(-rx_offset, rib_y_anchor, rib_z_anchor) * rib_shape
    part = part + Pos(+rx_offset, rib_y_anchor, rib_z_anchor) * rib_shape

    # 3. 传感器 M3 热熔螺母支撑凸台 (B1, B2)
    boss_h = p.boss_top_z - (p.base_thick_z - 0.2)
    boss_solid = make_mounting_boss(outer_d=p.boss_outer_d, height=boss_h, base_flare=1.5, flare_height=2.2)
    bx = p.boss_span_x / 2.0
    for bid, sign_x, side_cn in [("B1", -1, "左"), ("B2", +1, "右")]:
        part = part + Pos(sign_x * bx, p.boss_pos_y, p.base_thick_z - 0.2) * boss_solid
        reg.mark(
            bid,
            (sign_x * bx, p.boss_pos_y, p.boss_top_z),
            f"{side_cn}传感器M3支撑柱",
            kind="boss",
            spec="M3热熔螺母" if mode == "fdm" else "M3攻丝孔",
            normal=(0, 0, 1),
        )

    # 4. 底板前侧两条竖直转角导圆角
    front_z_edges = (
        part.edges()
        .filter_by(Axis.Z)
        .filter_by(lambda e: e.center().Y < y_front + 1.0)
    )
    part = safe_fillet(part, front_z_edges, p.corner_fillet, reg, "前侧外圆角")

    # 5. 正面居中走线与底部接插件避空U型槽 (S1)
    slot_cut = Pos(0, y_front + p.front_slot_d / 2.0 - 1.0, -1.0) * Box(
        p.front_slot_w,
        p.front_slot_d + 2.0,
        p.base_thick_z + 10.0,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )
    reg.mark("S1", (0, y_front + 3.0, p.base_thick_z / 2.0), "正面走线避空U槽", kind="slot", spec=f"宽{p.front_slot_w}mm", normal=(0, -1, 0))
    part = part - slot_cut

    # 6. 支撑柱顶面打 M3 热熔螺母孔 / CNC攻丝孔
    hs_cutter = make_heatset_cutter(tol, size="M3", screw_relief_depth=3.5)
    for sign_x in (-1, +1):
        part = part - Pos(sign_x * bx, p.boss_pos_y, p.boss_top_z) * hs_cutter

    # 7. 底板前侧左右 M3 腰型调节安装孔 (H1, H2)
    slot_hole = make_slot_cutter(tol, "M3", center_span=p.base_slot_span, depth=p.base_thick_z + 4.0)
    for hid, sign_x, side_cn in [("H1", -1, "左前"), ("H2", +1, "右前")]:
        hpos = reg.mark(
            hid,
            (sign_x * p.base_slot_x, p.base_slot_y, p.base_thick_z),
            f"{side_cn}底座M3腰型孔",
            kind="hole",
            spec=f"M3长槽+{p.base_slot_span}mm",
            normal=(0, 0, 1),
        )
        part = part - Pos(*hpos) * slot_hole

    # 8. 背面立板中心 M4 沉头安装孔 (H3: 从前侧内壁 -Y 朝背面 +Y 打穿，使螺丝头藏在支架内侧!)
    inner_back_y = y_back - p.back_wall_t
    h3_pos = reg.mark(
        "H3",
        (0.0, inner_back_y, p.back_hole_z),
        "背板中心M4沉头固定孔",
        kind="hole",
        spec="M4沉头",
        normal=(0, -1, 0),
    )
    # 从 -Y 表面往 +Y 方向钻入: Rot(-90, 0, 0) 将原 -Z 钻孔方向旋至 +Y!
    cb_m4 = make_counterbore_cutter(tol, "M4", depth=p.back_wall_t + 5.0, head_overcut=8.0)
    part = part - Pos(*h3_pos) * Rot(-90, 0, 0) * cb_m4

    # 9. 底面 (-Z) 防象脚微倒角
    if tol.bed_chamfer > 0:
        bottom_face = part.faces().sort_by(Axis.Z)[0]
        part = safe_chamfer(part, bottom_face.outer_wire().edges(), tol.bed_chamfer, reg, "底面防象脚倒角")

    return part, reg, p


if __name__ == "__main__":
    from cad_kit.cli import run_part_build
    run_part_build(__file__)
