"""Sweep: no-RL (fixed cycle) vs RL models across the test route grid.

For every (regime, density) route, simulate with:
  - no-RL  (SUMO's auto-rotating fixed cycle)
  - RL-2026 (current best new model)
  - RL-2024 (known 2024 baseline)
and record mean waiting time per action. Then plot:
  row 1: mean waiting vs density
  row 2: RL/no-RL ratio (1.0 = parity, <1 = RL better)

Usage:
  python scripts/sweep.py [--horizon 2000] [--out <dir>]
"""
import csv
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
os.environ.setdefault(
    "SUMO_HOME",
    "/home/bishal/Programming/.venv/lib/python3.11/site-packages/sumo",
)

import contextlib
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).parents[1]
LOGGING_OUTPUT = ROOT / "logging_outputs"
MODELS_PATH = ROOT / "models"

NEW_MODEL = str(
    (MODELS_PATH / "2026-09-19 20_58_34.563767-TrafficIntersection-TripleLaneGUI-ppo" / "best_model.zip").resolve()
)
OLD_MODEL = str(
    (MODELS_PATH / "2024-08-05 21_20_54.921009-TrafficIntersection-TripleLaneGUI-ppo" / "best_model.zip").resolve()
)

REGIMES = ["uniform", "optimal", "very_optimal"]
DENSITIES = [0.1, 0.2, 0.4]

HORIZON = 2000


def run_one(tag, use_rl, route_path, model_path=None, horizon=HORIZON):
    import traffic as tr

    tr.LOG_TO_FILE = True
    tr.LOG_WAITING_TIME = True
    tr.use_gui = False
    tr.CONVERT_LOG_TO_CSV = False
    tr.GENERATE_CUSTOM_ROUTE = False

    uid = f"{tag}-{int(time.time()*1000)}"
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        tr.start_logging(
            use_rl=use_rl,
            log_to_file=True,
            convert_file_to_csv=False,
            nogui=True,
            unique_identifier=uid,
            log_percentage_vehicle_passing=False,
            total_timesteps=horizon,
            traffic_intersection_type="triple",
            model_path=model_path,
            route_file_path=route_path,
        )

    prefix = "rl" if use_rl else "no-rl"
    csv_path = LOGGING_OUTPUT / f"log-output-{prefix}-{uid}.waiting-time.csv"
    waits = []
    try:
        with open(csv_path) as fh:
            r = csv.reader(fh)
            next(r)
            waits = [float(row[1]) for row in r if row]
    except FileNotFoundError:
        print(f"  !! missing output for {tag} ({prefix})")
    finally:
        if csv_path.exists():
            csv_path.unlink()

    max_green, switches = None, None
    green_path = LOGGING_OUTPUT / f"log-output-{prefix}-{uid}.green-duration.csv"
    try:
        with open(green_path) as fh:
            r = csv.reader(fh)
            next(r)
            max_green = [0.0] * 4
            switches = [0] * 4
            for row in r:
                if row:
                    p = int(row[0])
                    max_green[p] = float(row[1])
                    switches[p] = int(row[2])
    except FileNotFoundError:
        print(f"  !! missing green-duration output for {tag} ({prefix})")
    finally:
        if green_path.exists():
            green_path.unlink()

    return {
        "wait": (sum(waits) / len(waits)) if waits else None,
        "max_green": max_green,
        "switches": switches,
    }


def main():
    horizon = HORIZON
    out_dir = ROOT / "sweep_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    import traffic as tr  # ensure module importable once for env registration
    del tr

    results = {}
    for regime in REGIMES:
        results[regime] = {}
        for density in DENSITIES:
            route = ROOT / "sumo-files" / "test-routes" / f"{regime}-{density}.rou.xml"
            print(f"[{regime} {density}]", flush=True)
            rl_new = run_one(f"new-{regime}-{density}", True, route, NEW_MODEL, horizon)
            rl_old = run_one(f"old-{regime}-{density}", True, route, OLD_MODEL, horizon)
            norl = run_one(f"norl-{regime}-{density}", False, route, None, horizon)
            results[regime][density] = {"no_rl": norl, "rl_2026": rl_new, "rl_2024": rl_old}
            fmt = lambda d: f"{d['wait']:.1f}" if d and d["wait"] is not None else "NA"
            mg = lambda d: max(d["max_green"]) if d and d["max_green"] else "NA"
            print(
                f"    no_rl={fmt(norl)}  rl_2026={fmt(rl_new)}  rl_2024={fmt(rl_old)}"
                f"  | maxg no={mg(norl)} 26={mg(rl_new)} 24={mg(rl_old)}",
                flush=True,
            )

    # dump csv
    with open(out_dir / "results.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["regime", "density", "no_rl", "rl_2026", "rl_2024", "ratio_2026", "ratio_2024"])
        for regime in REGIMES:
            for density in DENSITIES:
                r = results[regime][density]
                norl_w = r["no_rl"]["wait"] if r["no_rl"] else None
                new_w = r["rl_2026"]["wait"] if r["rl_2026"] else None
                old_w = r["rl_2024"]["wait"] if r["rl_2024"] else None
                w.writerow([
                    regime, density, norl_w, new_w, old_w,
                    f"{new_w / norl_w:.3f}" if (norl_w and new_w) else "",
                    f"{old_w / norl_w:.3f}" if (norl_w and old_w) else "",
                ])

    # per-phase max green duration + switch counts (deep-dive data)
    with open(out_dir / "green_duration.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["regime", "density", "model", "phase", "max_green", "switch_count"])
        for regime in REGIMES:
            for density in DENSITIES:
                r = results[regime][density]
                for key in ["no_rl", "rl_2026", "rl_2024"]:
                    d = r[key]
                    for p in range(4):
                        w.writerow([
                            regime, density, key, p,
                            d["max_green"][p] if d and d["max_green"] else "",
                            d["switches"][p] if d and d["switches"] else "",
                        ])

    # plot
    colors = {"no_rl": "black", "rl_2026": "tab:red", "rl_2024": "tab:blue"}
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for i, regime in enumerate(REGIMES):
        ax = axes[0][i]
        ax.set_title(f"{regime}")
        for key, color in colors.items():
            xs, ys = [], []
            for d in DENSITIES:
                r = results[regime][d][key]
                if r and r["wait"] is not None:
                    xs.append(d)
                    ys.append(r["wait"])
            ax.plot(xs, ys, "o-", color=color, label=key)
        ax.set_xlabel("density (veh/s)")
        ax.set_ylabel("mean wait/action")
        ax.legend()

        ax = axes[1][i]
        ax.axhline(1.0, color="gray", ls="--", lw=1)
        for key, color in [("rl_2026", "tab:red"), ("rl_2024", "tab:blue")]:
            xs, ys = [], []
            for d in DENSITIES:
                r = results[regime][d]
                norl_w = r["no_rl"]["wait"] if r["no_rl"] else None
                rl_w = r[key]["wait"] if r[key] else None
                if norl_w and rl_w:
                    xs.append(d)
                    ys.append(rl_w / norl_w)
            ax.plot(xs, ys, "o-", color=color, label=f"{key}/no_rl")
        ax.axhline(0.75, color="green", ls=":", lw=1)
        ax.set_xlabel("density (veh/s)")
        ax.set_ylabel("RL / no-RL ratio")
        ax.set_ylim(0, None)
        ax.legend()

        ax = axes[2][i]
        ax.set_title(f"{regime} — max green duration")
        for key, color in colors.items():
            xs, ys = [], []
            for d in DENSITIES:
                r = results[regime][d][key]
                if r and r["max_green"]:
                    xs.append(d)
                    ys.append(max(r["max_green"]))
            ax.plot(xs, ys, "o-", color=color, label=key)
        ax.axhline(30, color="gray", ls="--", lw=1)
        ax.set_xlabel("density (veh/s)")
        ax.set_ylabel("max green (s)")
        ax.legend()
    fig.suptitle(f"Sweep horizon={horizon}s — mean waiting per action & max green duration")
    fig.tight_layout()
    out_png = out_dir / "sweep_plot.png"
    fig.savefig(out_png, dpi=110)
    print(f"\nwrote {out_dir / 'results.csv'}, {out_dir / 'green_duration.csv'} and {out_png}")


if __name__ == "__main__":
    main()
