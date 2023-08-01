#! /bin/python

'''
This script is to test the working of trained model.
A lot of unnecessary code is here, remove the code for different intersection when not required or 
make it work for general intersection.

For now all the codes is bundled unordered, separated by if else statements.

For model.predict() the argument is an array of number of vehicles in lanes
    The order is west, north, east and south
    The lanes order is from outermost(0) to innermost(2)

    Example: [" E0_0", " E0_1", " E0_2",
              "-E1_0", "-E1_1", "-E1_2",
              "-E2_0", "-E2_1", "-E2_2",
              "-E3_0", "-E3_1", "-E3_2"]

'''



from __future__ import print_function
from __future__ import absolute_import

import os
import subprocess
import sys
import optparse
import time
import numpy

from generateRouteFile import generate_routefile
from setupLaneCounting import setupLaneCounting

from multiprocessing import Process, set_start_method

import tensorflow as tf

# checking for sumo_home variable and exiting if it is not found
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import sumo
import sumo.tools.sumolib as sumolib
from sumo.tools import traci

import gym
from stable_baselines3 import PPO

import sys
from pathlib import Path
import os

# model is enough for prediction so, environment isn't required. 
# Place this code accordingly if environment is also required for additional training
'''
from custom_gym.envs.custom_env_dir import TrafficIntersectionEnvTripleLaneGUI
env = gym.make('TrafficIntersectionEnv{}LaneGUI-v1'.format(TRAFFIC_INTERSECTION_TYPE.capitalize()), sumocfg_file=sumocfg_file, network_file=net_file, route_file=route_file, use_gui=use_gui)
'''

# to change to project relative paths and properly resolve paths in different platforms
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # traffic-intersection-rl-environment-sumo root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# constants
TRAFFIC_INTERSECTION_TYPE="triple"
TOTAL_TIMESTEPS=500000  # This is the sumo timestep which is somewhat independent of steps taken by model in simulation
                       # rather this decides the number of steps in the simulation
GENERATE_CUSTOM_ROUTE=False
LOG_TO_FILE=True # generating log of lane vehicle stats
CONVERT_LOG_TO_CSV=True # converting the generated log output to csv
RL_ON = True # for turning rl on or off
BOTH_OUTPUTS = False # for running both rl and no rl mode and generate ouput on same data

YELLOW_TIME = 10
MIN_GREEN_TIME = 30

# environment stuffs
time_to_teleport = "-1" # setting time to teleport to -1 makes it so that vehicles wont teleport
use_gui = True # gui

# sumo stuffs
SUMO_DIRECTORY= Path(str(ROOT) + "/sumo-files").resolve()
net_file=Path(str(SUMO_DIRECTORY) + "/small-map-{}-lane.net.xml".format(TRAFFIC_INTERSECTION_TYPE)).resolve()
route_file=Path(str(SUMO_DIRECTORY) + "/test-small-map-{}-lane.rou.xml".format(TRAFFIC_INTERSECTION_TYPE)).resolve()
sumocfg_file=Path(str(SUMO_DIRECTORY) + "/small-map-{}-lane.sumocfg".format(TRAFFIC_INTERSECTION_TYPE)).resolve()
additional_file_no_rl=Path(str(SUMO_DIRECTORY) + "/countlanes-triple-lanes.add.xml").resolve()
additional_file_rl=Path(str(SUMO_DIRECTORY) + "/copyof.countlanes-triple-lanes.add.xml").resolve()
logging_output_folder=Path(str(ROOT) + "/logging_outputs").resolve()


# path to save the model or load the model from
models_path = Path(str(ROOT) + "/models").resolve()

# Here, formatting is done as to create error if wrong model is selected
# as, there won't be same model trained at exact same time and upto same timesteps
model_path = Path(str(models_path) + "/2023-07-26 10:02:16.560831-TrafficIntersection-{}LaneGUI-ppo/last_timestep.zip".format(TRAFFIC_INTERSECTION_TYPE.capitalize())).resolve()
model = PPO.load(str(model_path))

def run(use_rl: bool = False, connection_label = None):

    time_spent_on_inference = 0

    # is libsumo isn't used
    if "LIBSUMO_AS_TRACI" not in os.environ or os.environ["LIBSUMO_AS_TRACI"] != '1':
        # choosing the given connection for traffic light switching
        if connection_label is not None:
            traci.switch(connection_label)

    step = 0
    if not use_rl:  # if rl is turned off the continue without intervention
        print("\n Reinforcement Learning is Off. \n")
        while step < TOTAL_TIMESTEPS:
            traci.simulationStep()
            step = traci.simulation.getTime()
    else:
        print("\n Reinforcement Learning is On. \n")
        while step < TOTAL_TIMESTEPS:

            if step == 0: # setting the initial configuration
                if TRAFFIC_INTERSECTION_TYPE == "single":
                    traci.trafficlight.setPhase("J9", 4)
                elif TRAFFIC_INTERSECTION_TYPE == "double":
                    traci.trafficlight.setPhase("J11", 4)
                elif TRAFFIC_INTERSECTION_TYPE == "triple":
                    traci.trafficlight.setPhase("J1", 4)

                traci.simulationStep()
                step = traci.simulation.getTime()
                continue
            current_state = None
            next_configuration = current_state  # Doing this so that next configuration is defined if none of the conditions is met
            junction_with_lights = None

            if TRAFFIC_INTERSECTION_TYPE == "single":

                junction_with_lights = "J9"

                current_state = traci.trafficlight.getPhase(junction_with_lights)


                # starting from left-upper lane, skipping one lane
                vehicle_count_lane_0 = traci.lane.getLastStepVehicleNumber("-E8_0")
                vehicle_count_lane_1 = traci.lane.getLastStepVehicleNumber("-E9_0")
                vehicle_count_lane_2 = traci.lane.getLastStepVehicleNumber("-E10_0")
                vehicle_count_lane_3 = traci.lane.getLastStepVehicleNumber("E7_0")

                lanes_observation = [vehicle_count_lane_0, vehicle_count_lane_1, vehicle_count_lane_2, vehicle_count_lane_3]

                before_prediction = time.time()
                next_configuration, _state = model.predict(lanes_observation, deterministic=True)
                after_prediction = time.time()
                time_spent_on_inference += after_prediction - before_prediction

                if next_configuration == current_state / 2: # dividing by 2 as green phases is 0, 2, 4, 6
                    traci.trafficlight.setPhase(junction_with_lights, int(current_state))
                else:
                    # Trying to turn on yellow light
                    delta_yellow_time = 0
                    traci.trafficlight.setPhase(junction_with_lights, int(current_state + 1))
                    while delta_yellow_time < YELLOW_TIME:
                        traci.simulationStep()
                        delta_yellow_time += 1

            elif TRAFFIC_INTERSECTION_TYPE == "double":

                junction_with_lights = "J11"

                current_state = traci.trafficlight.getPhase(junction_with_lights)


                vehicle_count_lane_0 = traci.lane.getLastStepVehicleNumber("E9_0")
                vehicle_count_lane_1 = traci.lane.getLastStepVehicleNumber("E9_1")
                vehicle_count_lane_2 = traci.lane.getLastStepVehicleNumber("E8_0")
                vehicle_count_lane_3 = traci.lane.getLastStepVehicleNumber("E8_1")
                vehicle_count_lane_4 = traci.lane.getLastStepVehicleNumber("-E10_0")
                vehicle_count_lane_5 = traci.lane.getLastStepVehicleNumber("-E10_1")
                vehicle_count_lane_6 = traci.lane.getLastStepVehicleNumber("-E11_0")
                vehicle_count_lane_7 = traci.lane.getLastStepVehicleNumber("-E11_1")

                lanes_observation = [vehicle_count_lane_0, vehicle_count_lane_1, vehicle_count_lane_2, vehicle_count_lane_3, vehicle_count_lane_4, vehicle_count_lane_5, vehicle_count_lane_6, vehicle_count_lane_7]
                before_prediction = time.time()
                next_configuration, _state = model.predict(lanes_observation, deterministic=True)
                after_prediction = time.time()
                time_spent_on_inference += after_prediction - before_prediction

                if next_configuration == current_state / 2: # dividing by 2 as green phases is 0, 2, 4, 6
                    traci.trafficlight.setPhase(junction_with_lights, current_state)
                else:
                    # Trying to turn on yellow light
                    delta_yellow_time = 0
                    traci.trafficlight.setPhase(junction_with_lights, current_state + 1)
                    while delta_yellow_time < YELLOW_TIME:
                        traci.simulationStep()
                        delta_yellow_time += 1

            elif TRAFFIC_INTERSECTION_TYPE == "triple":

                junction_with_lights = "J1"

                current_state = traci.trafficlight.getPhase(junction_with_lights)

                lanes_to_observe = ["E0_0", "E0_1", "E0_2",
                                "-E1_0", "-E1_1", "-E1_2",
                                "-E2_0", "-E2_1", "-E2_2",
                                "-E3_0", "-E3_1", "-E3_2"]
                    
                lanes_observation = numpy.zeros(lanes_to_observe.__len__())
                for i, lane in enumerate(lanes_to_observe):
                    lanes_observation[i] = traci.lane.getLastStepVehicleNumber(lane)
                    
                before_prediction = time.time()
                next_configuration, _state = model.predict(lanes_observation, deterministic=True)
                after_prediction = time.time()
                time_spent_on_inference += after_prediction - before_prediction

                if next_configuration == current_state / 2: # dividing by 2 as green phases is 0, 2, 4, 6
                    traci.trafficlight.setPhase(junction_with_lights, current_state)
                else:
                    # Trying to turn on yellow light
                    delta_yellow_time = 0
                    traci.trafficlight.setPhase(junction_with_lights, current_state + 1)
                    while delta_yellow_time < YELLOW_TIME:
                        traci.simulationStep()
                        delta_yellow_time += 1
                

            # Turning green light for the predefined period of time
            delta_green_time = 0
            traci.trafficlight.setPhase(junction_with_lights, int(next_configuration * 2))
            while delta_green_time < MIN_GREEN_TIME:
                traci.simulationStep()
                delta_green_time += 1

            step = traci.simulation.getTime()
        
        if step >= TOTAL_TIMESTEPS:
            print(f"Total time spend in inference by model: {time_spent_on_inference}")


def start_logging(use_rl: bool, log_to_file: bool, nogui: bool = True, unique_identifier: str | int =  time.time()) -> Path | None:
    global net_file, route_file, sumocfg_file, additional_file_rl, additional_file_no_rl, logging_output_folder

    if log_to_file:
        output_file = Path(str(logging_output_folder) + \
        f"/log-output-{'rl' if use_rl else 'no-rl'}-{unique_identifier}.xml").resolve().__str__()

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if nogui:
        sumoBinary = sumolib.checkBinary('sumo')
    else:
        sumoBinary = sumolib.checkBinary('sumo-gui')

    # Generating custom route file
    if GENERATE_CUSTOM_ROUTE and TRAFFIC_INTERSECTION_TYPE == "double":
        route_file = generate_routefile()

    # Generating a additional file for logging
    if log_to_file:
        if not os.path.exists(logging_output_folder):
            os.mkdir(logging_output_folder)
                
        setupLaneCounting(begin = 0, end = TOTAL_TIMESTEPS,
        trafficLightSwitchingTime=MIN_GREEN_TIME,
        yellowLightTime=YELLOW_TIME,
        outputFile= output_file,
        additionalFileGenerationPath=additional_file_rl if use_rl else additional_file_no_rl)
    
    connection_label = f"rl-{unique_identifier}" if use_rl else f"norl-{unique_identifier}"
    
    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    if log_to_file:
        traci.start([sumoBinary,
        "-n", str(net_file),
        "-r", str(route_file),
        "--time-to-teleport", time_to_teleport,
        "-a", str(additional_file_rl if use_rl else additional_file_no_rl)], label=connection_label)
    else:
        traci.start([sumoBinary,
        "-n", str(net_file),
        "-r", str(route_file),
        "--time-to-teleport", time_to_teleport], label=connection_label)
    
       

    run(use_rl=use_rl,connection_label=connection_label)

    if "LIBSUMO_AS_TRACI" not in os.environ or os.environ["LIBSUMO_AS_TRACI"] != '1':
        traci.switch(connection_label)
    traci.close()

    if log_to_file:
        convert_to_csv(output_file)
    
    return output_file if log_to_file else None



def convert_to_csv(filename: str):
    global LOG_TO_FILE, CONVERT_LOG_TO_CSV
    # this way is a bit hacky, but solves import error because of different modules
    if LOG_TO_FILE and CONVERT_LOG_TO_CSV:
        subprocess.call([Path(str(tools)).joinpath("xml").joinpath("xml2csv.py").resolve().__str__(), str(filename)])
        print(f"\nConverted output file {filename} to csv successfully.")


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    optParser.add_option("--logging-off", action="store_true",
                        default=False, help="Turn of the lane vehicle logging")
    optParser.add_option("--convert-to-csv", action="store_true",
                        default=False, help="convert the created log xml file to csv")
    optParser.add_option("--turn-off-rl", action="store_true",
                        default=False, help="turn off the reinforcement learning inference and continue using only sumo")
    optParser.add_option("--generate-both-outputs", action="store_true",
                        default=False, help="generate both rl and no rl outputs with same id")
    options, args = optParser.parse_args()
    return options

def start(nogui=True, logging_off=False, convert_to_csv=True, turn_off_rl=False, generate_both_outputs=True):

    # for checking availability of gpu
    gpus = tf.config.list_physical_devices('GPU')   
    
    if len(gpus) > 0:  # for use with gpu
        start_method = 'spawn'
    else:
        start_method = "fork" if sys.platform=="linux" else "forkserver"

    set_start_method(start_method)

    global LOG_TO_FILE, CONVERT_LOG_TO_CSV, RL_ON, BOTH_OUTPUTS

    if logging_off:
        LOG_TO_FILE = False
    if nogui:
        CONVERT_LOG_TO_CSV = True
    if turn_off_rl:
        RL_ON = False
    if generate_both_outputs:
        BOTH_OUTPUTS = True

    common_time = time.time()

    no_rl_kwargs = {
                "use_rl": False,
                "log_to_file": True,
                "nogui": nogui,
                "unique_identifier": common_time,
            }

    rl_kwargs = no_rl_kwargs.copy()
    rl_kwargs["use_rl"] = True

    if not os.path.exists(logging_output_folder):
        os.mkdir(logging_output_folder)


    if BOTH_OUTPUTS:
        
        rl_process = Process(group=None, target=start_logging, kwargs=rl_kwargs)
        rl_process.start()
        no_rl_process = Process(group=None, target=start_logging, kwargs=no_rl_kwargs)
        no_rl_process.start()

        if not rl_process.is_alive():
            print(f"RL logging has stopped, exit code {rl_process.exitcode}")
        
        if not no_rl_process.is_alive():
            print(f"No RL logging has stopped, exit code {no_rl_process.exitcode}")

        rl_process.join()
        no_rl_process.join() 

    elif not turn_off_rl:
        output_file = start_logging(**rl_kwargs)

        if output_file is not None:
            print(f"Generation of {output_file} succeded!")
        else:
            print(f"Generation of CSV failed!")
    else:
        output_file = start_logging(**no_rl_kwargs)

        if output_file is not None:
            print(f"Generation of {output_file} succeded!")
        else:
            print(f"Generation of CSV failed!")



if __name__ == "__main__":

    options = get_options()

    start(**vars(options))
