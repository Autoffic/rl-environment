from genericpath import exists
import random
import os
import pathlib


def generate_routefile(intersection_type: str, routefile=None, number_of_time_steps = 100000) -> pathlib.Path:

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

        # choice pool for probabilities
        occurances = [max(int(random.random() * 0.5 * interval_range), 101) for _ in range(number_of_choices)] 

        # number of vehicles
        from_left = random.randrange(100, random.choice(occurances))
        from_up = random.randrange(100, random.choice(occurances))
        from_right = random.randrange(100, random.choice(occurances))
        from_down = random.randrange(100, random.choice(occurances))

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
    <interval begin="{interval_begin}" end="{interval_end}">\n \
        <!-- Originating from left-->\n \
        <flow id="car"   departLane="0" from="E0" to="E1" number="{int(0.33 * from_left)}"/>\n \
        <flow id="truck" departLane="1" from="E0" to="E2" number="{int(0.33 * from_left)}"/>\n \
        <flow id="bus"   departLane="2" from="E0" to="E3" number="{int(0.34 * from_left)}"/>\n \
\n \
        <!-- Originating from up-->\n \
        <flow id="ev"            departLane="2" from="-E1" to="-E0" number="{int(0.33 * from_up)}"/>\n \
        <flow id="motorcycle"    departLane="0" from="-E1" to="E2" number="{int(0.33 * from_up)}"/>\n \
        <flow id="motorcycle-EW" departLane="1" from="-E1" to="E3" number="{int(0.34 * from_up)}"/>\n \
\n \
        <!-- Originating from right-->\n \
        <flow id="car-EW"     departLane="1" from="-E2" to="-E0" number="{int(0.33 * from_right)}"/>\n \
        <flow id="truck-EW"   departLane="2" from="-E2" to="E1" number="{int(0.33 * from_right)}"/>\n \
        <flow id="bus-EW"     departLane="0" from="-E2" to="E3" number="{int(0.34 * from_right)}"/>\n \
\n \
        <!-- Originatin from down-->\n \
        <flow id="ev-EW"      departLane="0" from="-E3" to="-E0" number="{int(0.33 * from_down)}"/>\n \
        <flow id="truck-dup1" departLane="1" from="-E3" to="E1" number="{int(0.33 * from_down)}"/>\n \
        <flow id="bus-dup1"   departLane="2" from="-E3" to="E2" number="{int(0.34 * from_down)}"/>\n \
\n \
    </interval>\n \
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
