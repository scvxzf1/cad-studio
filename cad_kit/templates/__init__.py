"""Industrial parametric structural templates for CAD Studio."""

from .enclosure import EnclosureLayout, EnclosureParams, cut_panel_port, make_split_enclosure
from .extrusion import (
    EXTRUSION_SPECS,
    ExtrusionSeries,
    ExtrusionSpec,
    make_din_rail_clip,
    make_extrusion_bracket,
    make_extrusion_mount_plate,
    make_tslot_tongue,
)
from .mechanics import BEARING_TABLE, make_bearing_block, make_d_shaft_cutter, make_spur_gear

__all__ = [
    "BEARING_TABLE",
    "EXTRUSION_SPECS",
    "EnclosureLayout",
    "EnclosureParams",
    "ExtrusionSeries",
    "ExtrusionSpec",
    "cut_panel_port",
    "make_bearing_block",
    "make_d_shaft_cutter",
    "make_din_rail_clip",
    "make_extrusion_bracket",
    "make_extrusion_mount_plate",
    "make_split_enclosure",
    "make_spur_gear",
    "make_tslot_tongue",
]
