"""35mm Standard TS35 DIN Rail Snap-On Clip: demo_din_clip"""

from __future__ import annotations

from dataclasses import dataclass
from build123d import Part

from cad_kit import FeatureRegistry, ManufacturingMode, ToleranceProfile, make_din_rail_clip


@dataclass
class Params:
    width_x: float = 18.0
    top_deck_y: float = 48.0
    deck_thick_z: float = 4.5
    mount_hole_span_y: float = 30.0
    mount_screw: str = "M3"


def build(
    params: Params | None = None,
    mode: ManufacturingMode = "fdm",
) -> tuple[Part, FeatureRegistry, Params]:
    p = params or Params()
    tol = ToleranceProfile.for_mode(mode)
    reg = FeatureRegistry(part_name="demo_din_clip", mode=mode, description="35mm 欧标 DIN 导轨快拆弹性卡扣底座")
    part = make_din_rail_clip(
        width_x=p.width_x,
        top_deck_y=p.top_deck_y,
        deck_thick_z=p.deck_thick_z,
        mount_hole_span_y=p.mount_hole_span_y,
        mount_screw=p.mount_screw,
        tol=tol,
        reg=reg,
    )
    return part, reg, p
