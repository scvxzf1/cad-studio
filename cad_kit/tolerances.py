"""Dual-mode manufacturing tolerances & metric fastener specifications.

Supports switching seamlessly between:
- 'fdm'   : FDM 3D printing (Bambu Studio PLA/PETG/ABS/ASA inner-hole XY compensation,
            heat-set brass inserts, plastic self-tap holes, elephant-foot bed chamfer,
            sliding/snap clearances).
- 'cnc'   : Subtractive CNC machining / sheet metal (ISO 273 medium/close clearance holes,
            ISO metric tap drill sizes, standard H7/g6 & sliding fits, edge break chamfer).
- 'exact' : Zero-compensation nominal CAD geometry (theoretical reference dimensions).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

ManufacturingMode = Literal["fdm", "cnc", "exact"]
HoleKind = Literal["clearance", "close", "tap", "heatset"]
FitKind = Literal["press", "snug", "slide", "loose"]


@dataclass(frozen=True)
class FastenerEntry:
    """Metric ISO fastener & hardware nominal table (units: mm)."""

    size: str
    nominal_d: float
    pitch: float
    iso_close_d: float       # ISO 273 fine/close series
    iso_medium_d: float      # ISO 273 medium series
    cnc_tap_drill_d: float   # Standard metric tap drill (nominal_d - pitch)
    fdm_self_tap_d: float    # Direct screw-into-plastic hole diameter
    shcs_head_d: float       # ISO 4762 / DIN 912 Socket Head Cap Screw head diameter
    shcs_head_h: float       # ISO 4762 Socket Head Cap Screw head height
    fhcs_head_d: float       # ISO 10642 Flat Head Countersunk 90-deg head diameter
    hex_nut_s: float         # ISO 4032 Hex nut width across flats (s)
    hex_nut_h: float         # ISO 4032 Hex nut height (m)
    heatset_d: float         # Standard brass heat-set insert hole diameter (Voron/CNC Kitchen)
    heatset_depth: float     # Recommended pocket depth (insert length + ~1.0mm melt relief)
    heatset_lead_d: float    # Top entrance alignment step diameter


FASTENER_TABLE: dict[str, FastenerEntry] = {
    "M1.6": FastenerEntry("M1.6", 1.6, 0.35, 1.7, 1.8, 1.25, 1.45, 3.0, 1.6, 3.2, 3.2, 1.3, 2.5, 3.5, 2.7),
    "M2":   FastenerEntry("M2",   2.0, 0.40, 2.2, 2.4, 1.60, 1.80, 3.8, 2.0, 4.0, 4.0, 1.6, 3.2, 4.0, 3.4),
    "M2.5": FastenerEntry("M2.5", 2.5, 0.45, 2.7, 2.9, 2.05, 2.25, 4.5, 2.5, 5.0, 5.0, 2.0, 3.8, 4.8, 4.0),
    "M3":   FastenerEntry("M3",   3.0, 0.50, 3.2, 3.4, 2.50, 2.75, 5.5, 3.0, 6.0, 5.5, 2.4, 4.4, 5.5, 4.7),
    "M4":   FastenerEntry("M4",   4.0, 0.70, 4.3, 4.5, 3.30, 3.65, 7.0, 4.0, 8.0, 7.0, 3.2, 5.6, 6.8, 6.0),
    "M5":   FastenerEntry("M5",   5.0, 0.80, 5.3, 5.5, 4.20, 4.60, 8.5, 5.0, 10.0, 8.0, 4.7, 6.4, 8.0, 6.8),
    "M6":   FastenerEntry("M6",   6.0, 1.00, 6.4, 6.6, 5.00, 5.50, 10.0, 6.0, 12.0, 10.0, 5.2, 8.0, 9.5, 8.4),
    "M8":   FastenerEntry("M8",   8.0, 1.25, 8.4, 9.0, 6.80, 7.40, 13.0, 8.0, 16.0, 13.0, 6.8, 9.6, 11.5, 10.1),
}


def get_fastener(size: str) -> FastenerEntry:
    key = size.strip().upper()
    if key not in FASTENER_TABLE:
        supported = ", ".join(FASTENER_TABLE.keys())
        raise ValueError(f"Unsupported fastener size '{size}'. Supported: {supported}")
    return FASTENER_TABLE[key]


@dataclass(frozen=True)
class ToleranceProfile:
    """Process-specific geometric compensation & clearance preset."""

    mode: ManufacturingMode
    inner_hole_comp: float       # Added to general clearance/counterbore inner diameters (mm)
    cbore_radial_margin: float   # Extra diameter margin on socket head counterbores (mm)
    nut_flat_clearance: float    # Total across-flats clearance for hex nut pockets (mm)
    nut_height_clearance: float  # Extra depth/slot thickness clearance for hex nuts (mm)
    bed_chamfer: float           # Bottom (-Z) chamfer to prevent FDM elephant foot / CNC burr (mm)
    min_wall: float              # Recommended minimum structural wall thickness (mm)
    layer_height: float          # Nominal layer thickness for sacrificial bridges in FDM (mm)
    fit_gaps: dict[str, float]   # Per-side mating gap for press / snug / slide / loose (mm)

    @classmethod
    def for_mode(cls, mode: ManufacturingMode = "fdm") -> "ToleranceProfile":
        m = mode.strip().lower()
        if m == "fdm":
            return cls(
                mode="fdm",
                inner_hole_comp=0.15,       # e.g. M3 ISO close 3.2 -> 3.35mm; medium 3.4 -> 3.55mm
                cbore_radial_margin=0.60,   # M3 SHCS 5.5 + 0.6 + 0.15 = ~6.25mm easy socket access
                nut_flat_clearance=0.22,    # M3 5.5 -> 5.72mm snug drop-in on Bambu printers
                nut_height_clearance=0.25,  # M3 2.4 -> 2.65mm side-slot height
                bed_chamfer=0.40,           # Eliminates first-layer squish / elephant foot
                min_wall=1.20,              # 3 walls @ 0.4mm nozzle
                layer_height=0.20,
                fit_gaps={
                    "press": 0.05,
                    "snug": 0.12,
                    "slide": 0.22,
                    "loose": 0.35,
                },
            )
        if m == "cnc":
            return cls(
                mode="cnc",
                inner_hole_comp=0.0,
                cbore_radial_margin=0.50,   # Standard ISO counterbore radial clearance
                nut_flat_clearance=0.12,
                nut_height_clearance=0.10,
                bed_chamfer=0.20,           # Standard edge break
                min_wall=1.00,
                layer_height=0.0,
                fit_gaps={
                    "press": -0.01,
                    "snug": 0.02,
                    "slide": 0.05,
                    "loose": 0.15,
                },
            )
        if m == "exact":
            return cls(
                mode="exact",
                inner_hole_comp=0.0,
                cbore_radial_margin=0.40,
                nut_flat_clearance=0.0,
                nut_height_clearance=0.0,
                bed_chamfer=0.0,
                min_wall=0.50,
                layer_height=0.0,
                fit_gaps={
                    "press": 0.0,
                    "snug": 0.0,
                    "slide": 0.0,
                    "loose": 0.0,
                },
            )
        raise ValueError(f"Unknown manufacturing mode '{mode}'. Choose 'fdm', 'cnc', or 'exact'.")

    def clearance(self, fit: FitKind = "snug") -> float:
        """Per-side clearance gap (mm) between mating parts."""
        if fit not in self.fit_gaps:
            raise ValueError(f"Unknown fit '{fit}'. Choose from {list(self.fit_gaps.keys())}")
        return self.fit_gaps[fit]

    def hole_d(self, size: str = "M3", kind: HoleKind = "clearance") -> float:
        """Return compensated hole diameter (mm) for the active manufacturing mode."""
        f = get_fastener(size)
        if kind == "clearance":
            return round(f.iso_medium_d + self.inner_hole_comp, 3)
        if kind == "close":
            return round(f.iso_close_d + self.inner_hole_comp, 3)
        if kind == "tap":
            # FDM uses plastic self-tapping diameter; CNC/Exact uses metric tap drill
            return round(f.fdm_self_tap_d if self.mode == "fdm" else f.cnc_tap_drill_d, 3)
        if kind == "heatset":
            # Heat-set table values are already calibrated for FDM brass inserts
            return round(f.heatset_d if self.mode != "cnc" else f.cnc_tap_drill_d, 3)
        raise ValueError(f"Unknown hole kind '{kind}'")

    def counterbore_dims(self, size: str = "M3", close_fit: bool = False) -> tuple[float, float, float]:
        """Return (through_d, cbore_d, cbore_depth) for Socket Head Cap Screws (SHCS)."""
        f = get_fastener(size)
        thru_d = self.hole_d(size, "close" if close_fit else "clearance")
        cbore_d = round(f.shcs_head_d + self.cbore_radial_margin + self.inner_hole_comp, 3)
        cbore_h = round(f.shcs_head_h + (0.3 if self.mode == "fdm" else 0.2), 3)
        return thru_d, cbore_d, cbore_h

    def countersink_dims(self, size: str = "M3", close_fit: bool = False) -> tuple[float, float, float]:
        """Return (through_d, csink_d, angle_deg) for 90-degree Flat Head Screws (FHCS)."""
        f = get_fastener(size)
        thru_d = self.hole_d(size, "close" if close_fit else "clearance")
        csink_d = round(f.fhcs_head_d + (0.35 if self.mode == "fdm" else 0.2), 3)
        return thru_d, csink_d, 90.0

    def heatset_dims(self, size: str = "M3") -> tuple[float, float, float, float]:
        """Return (bore_d, pocket_depth, lead_in_d, lead_in_depth) for brass heat-set inserts."""
        f = get_fastener(size)
        return f.heatset_d, f.heatset_depth, f.heatset_lead_d, 0.8

    def nut_trap_dims(self, size: str = "M3", loose: bool = False) -> tuple[float, float]:
        """Return (across_flats_s, nut_pocket_h) for metric hex nuts."""
        f = get_fastener(size)
        extra = 0.12 if loose and self.mode != "exact" else 0.0
        s = round(f.hex_nut_s + self.nut_flat_clearance + extra, 3)
        h = round(f.hex_nut_h + self.nut_height_clearance + extra, 3)
        return s, h

    def summary_dict(self) -> dict:
        return asdict(self)
