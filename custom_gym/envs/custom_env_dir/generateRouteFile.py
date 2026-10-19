from genericpath import exists
import random
import os
import pathlib


# per-approach demand weights: the four approaches are E0 (left), -E1 (up),
# -E2 (right), -E3 (down). Weights sum to 1.0 over the four approaches.
REGIME_PROFILES = {
    "uniform": [0.25, 0.25, 0.25, 0.25],
    "optimal": [0.40, 0.20, 0.25, 0.15],
    "very_optimal": [0.70, 0.10, 0.12, 0.08],
}

# per-second total demand across all four approaches (veh/s)
DENSITIES = [0.1, 0.2, 0.4]

# curriculum start/end anchors for progressive difficulty (Lunar-Lander style).
# Start easy (very low demand, heavily imbalanced) and end hard (high uniform
# demand). Difficulty ramps linearly with training progress in [0, 1].
CURRICULUM_START_WEIGHTS = REGIME_PROFILES["very_optimal"]  # [0.70, 0.10, 0.12, 0.08]
CURRICULUM_END_WEIGHTS = REGIME_PROFILES["uniform"]  # [0.25, 0.25, 0.25, 0.25]
CURRICULUM_START_DENSITY = 0.05  # very low demand first
CURRICULUM_END_DENSITY = 0.4  # max sweep density last

# within-episode ramp: demand swings ±CURRICULUM_RAMP around the episode's base
# density (a light "rush-hour" effect). The between-episode curriculum raises
# the base difficulty itself across training.
CURRICULUM_RAMP = 0.1  # ±10%
CURRICULUM_SEGMENTS = 40  # near-continuous ramp

# full 3x3 grid: (regime, density), regime-major so a seed cycles through
# each regime an equal number of times
PROFILES = [
    (regime, density) for regime in REGIME_PROFILES for density in DENSITIES
]


def pick_profile(seed: int | None = None) -> tuple[str, float]:
    """Pick a (regime, density) deterministically from a seed, or at random."""
    if seed is not None:
        return PROFILES[seed % len(PROFILES)]
    return random.choice(PROFILES)


def curriculum_profile(progress: float) -> tuple[list[float], float]:
    """Interpolate (weights, density) linearly between the easy and hard
    anchors based on training progress in [0, 1]. Both anchors sum to 1.0, so
    any intermediate weights are a valid probability distribution."""
    t = min(max(progress, 0.0), 1.0)
    weights = [
        start + (end - start) * t
        for start, end in zip(CURRICULUM_START_WEIGHTS, CURRICULUM_END_WEIGHTS)
    ]
    density = CURRICULUM_START_DENSITY + (CURRICULUM_END_DENSITY - CURRICULUM_START_DENSITY) * t
    return weights, density


def generate_routefile(intersection_type: str, routefile=None, number_of_time_steps = 100000, seed: int | None = None, regime: str | None = None, density: float | None = None, weights: list[float] | None = None, curriculum: bool = False, base_density: float | None = None, base_weights: list[float] | None = None, num_segments: int = CURRICULUM_SEGMENTS) -> pathlib.Path:

    if seed is not None:
        random.seed(seed)

    if(routefile == None):
        routefilePath = pathlib.Path(__file__).parent.joinpath("sumo-files")

        # for naming route files
        range_start, range_end = 1, 100000
        random_file_number = random.randint(range_end, range_end)
    
        if not exists(routefilePath):
            os.mkdir(routefilePath)
        
        routefile = routefilePath.joinpath(f"random-route-{intersection_type}.rou.xml")

        loop_count, max_loop = 0, 15
        tries, max_tries = 0, 5
        while exists(routefile):
            random_file_number = random.randint(range_start, range_end)
            routefile = routefilePath.joinpath(f"{random_file_number}-route-{intersection_type}.rou.xml")

            if loop_count > max_loop:
        
                if tries > max_tries:
                    raise Exception(f"Exceeded maximum tries for creating a distinct route name, \
                                    Tried {tries * max_loop} times before quitting. \n \
                                    Please verify if there are huge number of route files generated previously.")

                tries += 1

                range_end, range_end = 10000, 20000
                loop_count = 0

    # re-seed before content so the filename-selection loop above
    # cannot shift the RNG state of the generated route contents
    if seed is not None:
        random.seed(seed)

    if intersection_type == "single":
        raise Exception("Not implemented yet")
    elif intersection_type == "double":
        # random.seed(42)  # make tests reproducible
        N = number_of_time_steps  # number of time steps
        # demand per second from different directions
        
        pWN = 1. / 80
        pWE = 1. / 58
        pWS = 1. / 92
        pNE = 1. / 31
        pNS = 1. / 15
        pNW = 1. / 22
        pES = 1. / 13
        pEW = 1. / 35
        pEN = 1. / 16
        pSW = 1. / 22
        pSN = 1. / 47
        pSE = 1. / 81

        with open(routefile, "w") as routes:
            print("""<routes>
            <vType id="typeWN" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
            <vType id="typeWE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="3" maxSpeed="16.67" guiShape="passenger"/>
            <vType id="typeWS" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2" maxSpeed="16.67" guiShape="motorcycle"/>
            <vType id="typeNE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
            <vType id="typeNS" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="3" maxSpeed="16.67" guiShape="passenger"/>
            <vType id="typeNW" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2" maxSpeed="16.67" guiShape="passenger"/>
            <vType id="typeES" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
            <vType id="typeEW" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="3" maxSpeed="16.67" guiShape="truck"/>
            <vType id="typeEN" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
            <vType id="typeSW" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
            <vType id="typeSN" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>   
            <vType id="typeSE" accel="0.8" decel="4.5" sigma="0.5" length="7" minGap="3" maxSpeed="25" guiShape="bus"/>
            
            <route id="left_up" edges="E9 E8" />
            <route id="left_right" edges="E9 E10" />
            <route id="left_down" edges="E9 E11" />
            <route id="up_right" edges="-E8 E10" />
            <route id="up_down" edges="-E8 E11" />
            <route id="up_left" edges="-E8 -E9" />
            <route id="right_down" edges="-E10 E11" />
            <route id="right_left" edges="-E10 -E9" />
            <route id="right_up" edges="-E10 E8" />
            <route id="down_left" edges="-E11 -E9" />
            <route id="down_up" edges="-E11 E8" />
            <route id="down_right" edges="-E11 E10" />""", file=routes)
            lastVeh = 0
            vehNr = 0
            for i in range(N):
                if random.uniform(0, 1) < pWN:
                    print('    <vehicle id="left_up_%i" type="typeWN" route="left_up" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pWE:
                    print('    <vehicle id="left_right_%i" type="typeWE" route="left_right" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pWS:
                    print('    <vehicle id="left_down_%i" type="typeWS" route="left_down" depart="%i" color="1,0,0"/>' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pNE:
                    print('    <vehicle id="up_right_%i" type="typeNE" route="up_right" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pNS:
                    print('    <vehicle id="up_down_%i" type="typeNS" route="up_down" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pNW:
                    print('    <vehicle id="up_left_%i" type="typeNW" route="up_left" depart="%i" color="1,0,0"/>' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pES:
                    print('    <vehicle id="right_down_%i" type="typeES" route="right_down" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pEW:
                    print('    <vehicle id="right_left_%i" type="typeEW" route="right_left" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pEN:
                    print('    <vehicle id="right_up_%i" type="typeEN" route="right_up" depart="%i" color="1,0,0"/>' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pSW:
                    print('    <vehicle id="down_left_%i" type="typeWS" route="down_left" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pSN:
                    print('    <vehicle id="down_up_%i" type="typeWN" route="down_up" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pSE:
                    print('    <vehicle id="down_right_%i" type="typeWE" route="down_right" depart="%i" color="1,0,0"/>' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
            print("</routes>", file=routes)
        return routefile
    elif intersection_type == "triple":

        # choices for acceleration, deacceleration pool and occurances
        number_of_choices = 10

        # number of total time steps in sumo
        interval_begin = 0
        interval_end = number_of_time_steps
        interval_range = interval_end - interval_begin

        # number of vehicles, scaled linearly with the horizon so the
        # per-second density is identical for training (79000s) and
        # evaluation (1000s) routes. When a regime is given, each approach
        # gets `horizon * density * weight[approach]`; the uniform default
        # splits the demand evenly across the four approaches.
        if curriculum:
            # within-episode ramp around the episode's base difficulty
            base_vps = base_density if base_density is not None else 0.2
            # keep a copy of per-approach weights alive for the loop below
            _weights = base_weights if base_weights is not None else None
        else:
            base_vps = density if density is not None else 0.2
            if weights is None:
                weights = REGIME_PROFILES[regime] if regime is not None else None
            _weights = weights

        def vehicle_count(weight: float | None, seg_begin: int, seg_end: int, vps: float) -> int:
            base = (
                (seg_end - seg_begin) * vps * weight
                if weight is not None
                else (seg_end - seg_begin) * vps / 4
            )
            return max(int(base * random.uniform(0.5, 1.5)), 1)

        def split_approach(total: int, shares: list[float]) -> list[int]:
            """Split an approach's vehicle total across its movement flows using
            largest-remainder allocation so the counts sum exactly to `total`
            (no vehicles silently lost to rounding). Individual flows may still
            get 0 when `total` is tiny; those are omitted when emitting XML."""
            floors = [int(total * s) for s in shares]
            remainder = total - sum(floors)
            order = sorted(
                range(len(shares)),
                key=lambda i: (total * shares[i]) - floors[i],
                reverse=True,
            )
            for i in order[:remainder]:
                floors[i] += 1
            return floors

        def interval_xml(seg_begin: int, seg_end: int, vps: float, suffix: str = "") -> str:
            w0 = _weights[0] if _weights else None
            w1 = _weights[1] if _weights else None
            w2 = _weights[2] if _weights else None
            w3 = _weights[3] if _weights else None
            from_left = vehicle_count(w0, seg_begin, seg_end, vps)
            from_up = vehicle_count(w1, seg_begin, seg_end, vps)
            from_right = vehicle_count(w2, seg_begin, seg_end, vps)
            from_down = vehicle_count(w3, seg_begin, seg_end, vps)

            c_l = split_approach(from_left, [0.33, 0.33, 0.34])
            c_u = split_approach(from_up, [0.33, 0.33, 0.34])
            c_r = split_approach(from_right, [0.33, 0.33, 0.34])
            c_d = split_approach(from_down, [0.33, 0.33, 0.34])

            def flow(count: int, fid: str, lane: str, frm: str, to: str) -> str:
                # a flow with number=0 makes SUMO warn "has no instances"; only
                # emit a flow when there is at least one vehicle to spawn
                if count < 1:
                    return ""
                return (
                    f'        <flow id="{fid}{suffix}" departLane="{lane}" '
                    f'from="{frm}" to="{to}" number="{count}"/>\n'
                )

            return (
                f'    <interval begin="{seg_begin}" end="{seg_end}">\n'
                + "        <!-- Originating from left-->\n"
                + flow(c_l[0], "car", "0", "E0", "E1")
                + flow(c_l[1], "truck", "1", "E0", "E2")
                + flow(c_l[2], "bus", "2", "E0", "E3")
                + "\n"
                + "        <!-- Originating from up-->\n"
                + flow(c_u[0], "ev", "2", "-E1", "-E0")
                + flow(c_u[1], "motorcycle", "0", "-E1", "E2")
                + flow(c_u[2], "motorcycle-EW", "1", "-E1", "E3")
                + "\n"
                + "        <!-- Originating from right-->\n"
                + flow(c_r[0], "car-EW", "1", "-E2", "-E0")
                + flow(c_r[1], "truck-EW", "2", "-E2", "E1")
                + flow(c_r[2], "bus-EW", "0", "-E2", "E3")
                + "\n"
                + "        <!-- Originating from down-->\n"
                + flow(c_d[0], "ev-EW", "0", "-E3", "-E0")
                + flow(c_d[1], "truck-dup1", "1", "-E3", "E1")
                + flow(c_d[2], "bus-dup1", "2", "-E3", "E2")
                + "\n"
                + "    </interval>"
            )

        # when curriculum is on, split the horizon into segments whose density
        # swings ±CURRICULUM_RAMP around the episode's base demand
        if curriculum:
            n_seg = max(num_segments, 1)
            segment_len = interval_range / n_seg
            intervals = []
            for seg_idx in range(n_seg):
                frac = seg_idx / (n_seg - 1) if n_seg > 1 else 0.0
                seg_vps = base_vps * (1.0 - CURRICULUM_RAMP + 2.0 * CURRICULUM_RAMP * frac)
                seg_begin = interval_begin + int(seg_idx * segment_len)
                seg_end = interval_begin + int((seg_idx + 1) * segment_len)
                intervals.append(interval_xml(seg_begin, seg_end, seg_vps, suffix=f"_{seg_idx}"))
            intervals_txt = "\n".join(intervals)
        else:
            intervals_txt = interval_xml(interval_begin, interval_end, base_vps)

        least_accel, max_accel = 0.5, 5

        acceleration_pool = [(least_accel + random.random() * max_accel) for i in range(number_of_choices)]
        deacceleration_pool: list[float] = [(max_accel - (accel - least_accel)) for accel in acceleration_pool]
        # since decel must be greater than 0
        # deacceleration_pool = [ for value in deacceleration_pool ]
        
        with open(routefile, "w") as routes:
            print(f"""\
<routes>\n \
    <vType id="car"\n \
            vClass="passenger" length="5" accel="{random.choice(acceleration_pool)}" decel="{random.choice(deacceleration_pool)}"\n \
            sigma="1.0" maxSpeed="10"/>\n \
\n \
    <vType id="car-EW"\n \
            vClass="passenger" length="5" accel="{random.choice(acceleration_pool)}" decel="{random.choice(deacceleration_pool)}"\n \
            sigma="1.0" maxSpeed="10"/>\n \
\n \
    <vType id="motorcycle"\n \
            vClass="motorcycle" length="3" accel="{random.choice(acceleration_pool)}" decel="{random.choice(deacceleration_pool)}" guiShape="motorcycle"\n \
            sigma="1.0" maxSpeed="10"/>\n \
\n \
    <vType id="motorcycle-EW"\n \
            vClass="motorcycle" length="3" accel="{random.choice(acceleration_pool)}" decel="{random.choice(deacceleration_pool)}" guiShape="motorcycle"\n \
            sigma="1.0" maxSpeed="10"/>\n \
\n \
    <vType id="truck"\n \
            vClass="truck" length="6" accel="{random.choice(acceleration_pool)}" decel="{random.choice(deacceleration_pool)}" guiShape="truck"\n \
            sigma="1.0" maxSpeed="7"/>\n \
\n \
    <vType id="truck-dup1"\n \
            vClass="truck" length="6" accel="{random.choice(acceleration_pool)}" decel="{random.choice(deacceleration_pool)}" guiShape="truck"\n \
            sigma="1.0" maxSpeed="7"/>\n \
\n \
    <vType id="truck-EW"\n \
            vClass="truck" length="6" accel="{random.choice(acceleration_pool)}" decel="{random.choice(deacceleration_pool)}" guiShape="truck"\n \
            sigma="1.0" maxSpeed="7"/>\n \
\n \
<vType id="ev"\n \
            vClass="emergency" length="7" accel="{random.choice(acceleration_pool)}" decel="{random.choice(deacceleration_pool)}" sigma="1.0"\n \
            maxSpeed="20" guiShape="emergency" speedFactor="2.0"\n \
            minGapLat="0.2"/>\n \
\n \
    <vType id="ev-EW"\n \
            vClass="emergency" length="7" accel="{random.choice(acceleration_pool)}" decel="{random.choice(deacceleration_pool)}" sigma="1.0"\n \
            maxSpeed="20" guiShape="emergency" speedFactor="2.0"\n \
            minGapLat="0.2"/>\n \
\n \
    <vType id="bus"\n \
            vClass="bus" length="8" accel="{random.choice(acceleration_pool)}" decel="{random.choice(deacceleration_pool)}" guiShape="bus"\n \
            sigma="1.0" maxSpeed="9"/>\n \
\n \
    <vType id="bus-dup1"\n \
            vClass="bus" length="8" accel="{random.choice(acceleration_pool)}" decel="{random.choice(deacceleration_pool)}" guiShape="bus"\n \
            sigma="1.0" maxSpeed="9"/>\n \
\n \
    <vType id="bus-EW"\n \
            vClass="bus" length="8" accel="{random.choice(acceleration_pool)}" decel="{random.choice(deacceleration_pool)}" guiShape="bus"\n \
            sigma="1.0" maxSpeed="9"/>\n \
\n \
{intervals_txt}\n \
\n \
</routes>\n\n""", file=routes)
        return routefile
    else:
        raise Exception("Not implemented yet")

if __name__=="__main__":

    intersection_types = ["single", "double", "triple"]

    for intersection in intersection_types:
        try:
            route_file = generate_routefile(intersection)

            # deleting the route file
            if exists(route_file):
                os.remove(route_file)

            print(f"Generation for intersection {intersection} succeeded.")
        except Exception as e:
            print(f"Failed to test for intersection {intersection}, raised exception '{e}'")
