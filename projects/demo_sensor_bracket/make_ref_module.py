"""Helper to generate a realistic reference hardware module (sensor_module.step & .stl) for testing."""

from pathlib import Path
from build123d import Align, Box, Cylinder, Pos, export_step, export_stl


def generate_reference_module(refs_dir: Path) -> tuple[Path, Path]:
    refs_dir.mkdir(parents=True, exist_ok=True)
    # Sensor module mounted at Z = 12..16mm: 36x26x4mm PCB + top lens dome + 2x Ø3.2mm M3 holes at (±13, +3)
    pcb = Pos(0, 2.0, 12.0) * Box(36.0, 26.0, 4.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
    lens = Pos(0, -2.0, 16.0) * Cylinder(radius=6.5, height=5.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
    conn = Pos(0, -9.0, 9.5) * Box(10.0, 6.0, 2.5, align=(Align.CENTER, Align.CENTER, Align.MIN))
    mod = pcb + lens + conn
    for hx in (-13.0, 13.0):
        mod = mod - Pos(hx, 5.0, 10.0) * Cylinder(
            radius=1.6, height=10.0, align=(Align.CENTER, Align.CENTER, Align.MIN)
        )

    step_p = refs_dir / "sensor_module.step"
    stl_p = refs_dir / "sensor_module.stl"
    export_step(mod, str(step_p))
    export_stl(mod, str(stl_p))
    return step_p, stl_p


if __name__ == "__main__":
    s, m = generate_reference_module(Path(__file__).parent / "refs")
    print("Generated:", s, m)
