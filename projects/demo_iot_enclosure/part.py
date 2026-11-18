"""Parametric Split Electronics Enclosure: demo_iot_enclosure (上下盖子母止口分体电子盒)"""

from __future__ import annotations

from dataclasses import dataclass
from build123d import Part

from cad_kit import (
    EnclosureLayout,
    EnclosureParams,
    FeatureRegistry,
    ManufacturingMode,
    ToleranceProfile,
    cut_panel_port,
    make_split_enclosure,
)


@dataclass
class Params:
    outer_x: float = 86.0
    outer_y: float = 58.0
    bottom_h: float = 22.0
    lid_h: float = 8.0
    wall_t: float = 2.8
    corner_r: float = 5.0
    flange_ears: bool = True
    vent_slots: int = 5
    usb_c_z: float = 11.0
    layout: EnclosureLayout = "print_plate"  # 'print_plate' | 'assembled' | 'bottom' | 'lid'


def build(
    params: Params | None = None,
    mode: ManufacturingMode = "fdm",
) -> tuple[Part, FeatureRegistry, Params]:
    p = params or Params()
    tol = ToleranceProfile.for_mode(mode)
    reg = FeatureRegistry(part_name="demo_iot_enclosure", mode=mode, description="demo_iot_enclosure 上下分体电子控制盒")

    enc_cfg = EnclosureParams(
        outer_x=p.outer_x,
        outer_y=p.outer_y,
        bottom_h=p.bottom_h,
        lid_h=p.lid_h,
        wall_t=p.wall_t,
        corner_r=p.corner_r,
        flange_ears=p.flange_ears,
        vent_slots=p.vent_slots,
        pcb_standoffs=[(-28.0, -18.0), (28.0, -18.0), (-28.0, 18.0), (28.0, 18.0)],
        pcb_standoff_h=5.0,
    )
    part = make_split_enclosure(enc_cfg, tol=tol, reg=reg, layout=p.layout)

    # 在下壳正面 (-Y = -outer_y/2) 开带插头避让沉槽的 USB-C 接口
    if p.layout != "lid":
        part = cut_panel_port(
            part,
            wall="-Y",
            port_type="usb_c",
            offset_along_wall=0.0,
            center_z=p.usb_c_z,
            wall_coord=-p.outer_y * 0.5,
            tol=tol,
            reg=reg,
            anchor_id="USB_C",
            desc="正面Type-C接口",
        )
    return part, reg, p


if __name__ == "__main__":
    from cad_kit.cli import run_part_build
    run_part_build(__file__)
