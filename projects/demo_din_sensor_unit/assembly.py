"""Multi-Part CAD Assembly: demo_din_sensor_unit"""

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


def build_assembly(mode: str = "fdm") -> Assembly:
    tol = ToleranceProfile.for_mode(mode)  # type: ignore[arg-type]
    asm = Assembly(
        name="demo_din_sensor_unit",
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
