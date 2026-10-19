"""Generate test route files for the density x imbalance sweep.

Motivation: uniform flow from every approach is "bad for RL" - a trivial
always-switching / fixed-cycle policy serves it just as well. To give the
controller something to learn, we vary the *imbalance* between approaches
(how much green the busy direction deserves) and the *total density*.

Regimes (per-approach weight of total flow: E0, -E1, -E2, -E3):
  uniform        -> [0.25, 0.25, 0.25, 0.25]  non-optimal for RL (fixed cycle is fine)
  optimal        -> [0.40, 0.20, 0.25, 0.15]  moderate imbalance, RL can adapt
  very_optimal   -> [0.70, 0.10, 0.12, 0.08]  strong imbalance, RL should clearly win
                     by holding the dominant direction green longer

Densities (total vehicles per second across all approaches): 0.1 / 0.2 / 0.4

Route files are written to sumo-files/test-routes/<regime>-<density>.rou.xml
with a fixed seed so every sweep run sees identical demand.
"""
import pathlib
import random

ROOT = pathlib.Path(__file__).parents[1]
SUMO_FILES = ROOT / "sumo-files"
OUTPUT_DIR = SUMO_FILES / "test-routes"

# approach order: E0 (left), -E1 (up), -E2 (right), -E3 (down)
REGIMES = {
    "uniform": [0.25, 0.25, 0.25, 0.25],
    "optimal": [0.40, 0.20, 0.25, 0.15],
    "very_optimal": [0.70, 0.10, 0.12, 0.08],
}

DENSITIES = [0.1, 0.2, 0.4]

HORIZON = 2000  # seconds of traffic to emit (matches the sweep sim length)

# (from_edge, to_edge, departLane, vtype) for the triple intersection.
# Splits are the movement mix used by generateRouteFile.py (0.33/0.33/0.34).
MOVEMENTS = {
    "E0":  [("E0", "E1", "0", "car"), ("E0", "E2", "1", "truck"), ("E0", "E3", "2", "bus")],
    "-E1": [("-E1", "-E0", "2", "ev"), ("-E1", "E2", "0", "motorcycle"), ("-E1", "E3", "1", "motorcycle-EW")],
    "-E2": [("-E2", "-E0", "1", "car-EW"), ("-E2", "E1", "2", "truck-EW"), ("-E2", "E3", "0", "bus-EW")],
    "-E3": [("-E3", "-E0", "0", "ev-EW"), ("-E3", "E1", "1", "truck-dup1"), ("-E3", "E2", "2", "bus-dup1")],
}

SPLITS = [0.33, 0.33, 0.34]

VTYPES = """\
    <vType id="car" vClass="passenger" length="5" accel="3.5" decel="2.2" sigma="1.0" maxSpeed="10"/>
    <vType id="car-EW" vClass="passenger" length="5" accel="3.5" decel="2.2" sigma="1.0" maxSpeed="10"/>
    <vType id="motorcycle" vClass="motorcycle" length="3" accel="3.8" decel="2.5" guiShape="motorcycle" sigma="1.0" maxSpeed="10"/>
    <vType id="motorcycle-EW" vClass="motorcycle" length="3" accel="3.8" decel="2.5" guiShape="motorcycle" sigma="1.0" maxSpeed="10"/>
    <vType id="truck" vClass="truck" length="6" accel="2.5" decel="2.2" guiShape="truck" sigma="1.0" maxSpeed="7"/>
    <vType id="truck-dup1" vClass="truck" length="6" accel="2.5" decel="2.2" guiShape="truck" sigma="1.0" maxSpeed="7"/>
    <vType id="truck-EW" vClass="truck" length="6" accel="2.5" decel="2.2" guiShape="truck" sigma="1.0" maxSpeed="7"/>
    <vType id="ev" vClass="emergency" length="7" accel="5.5" decel="2.2" sigma="1.0" maxSpeed="20" guiShape="emergency" speedFactor="2.0" minGapLat="0.2"/>
    <vType id="ev-EW" vClass="emergency" length="7" accel="5.5" decel="2.2" sigma="1.0" maxSpeed="20" guiShape="emergency" speedFactor="2.0" minGapLat="0.2"/>
    <vType id="bus" vClass="bus" length="8" accel="3.5" decel="2.2" guiShape="bus" sigma="1.0" maxSpeed="9"/>
    <vType id="bus-dup1" vClass="bus" length="8" accel="3.5" decel="2.2" guiShape="bus" sigma="1.0" maxSpeed="9"/>
    <vType id="bus-EW" vClass="bus" length="8" accel="3.5" decel="2.2" guiShape="bus" sigma="1.0" maxSpeed="9"/>
"""


def generate_test_routes(horizon: int = HORIZON, seed: int = 42) -> list[pathlib.Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    random.seed(seed)
    for regime, weights in REGIMES.items():
        for density in DENSITIES:
            path = OUTPUT_DIR / f"{regime}-{density}.rou.xml"
            _write_route(path, weights, density, horizon)
            paths.append(path)
    return paths


def _write_route(path: pathlib.Path, weights: list[float], density: float, horizon: int) -> None:
    flows = []
    for approach, weight in zip(MOVEMENTS.keys(), weights):
        rate = density * weight  # vehicles/sec on this approach
        for (frm, to, lane, vtype), split in zip(MOVEMENTS[approach], SPLITS):
            n = int(rate * split * horizon)
            if n <= 0:
                continue
            flows.append((frm, to, lane, vtype, n))
    with open(path, "w") as f:
        f.write("<routes>\n")
        f.write(VTYPES)
        f.write(f'    <interval begin="0" end="{horizon}">\n')
        for frm, to, lane, vtype, n in flows:
            f.write(
                f'        <flow id="{vtype}-{frm.replace("-", "")}-{to.replace("-", "")}" '
                f'departLane="{lane}" from="{frm}" to="{to}" number="{n}"/>\n'
            )
        f.write("    </interval>\n")
        f.write("</routes>\n")


if __name__ == "__main__":
    generated = generate_test_routes()
    for p in generated:
        print(p)
