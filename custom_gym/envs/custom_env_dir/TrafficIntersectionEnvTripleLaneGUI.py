from __future__ import print_function
from __future__ import absolute_import

import os
import sys
from genericpath import exists

# checking for sumo_home variable and exiting if it is not found
if 'SUMO_HOME' in os.environ:
    tools: str = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import sumo
import sumo.tools.sumolib as sumolib
from sumo.tools import traci
import gym
import numpy

from .generateRouteFile import generate_routefile

import pathlib

TRAFFIC_INTERSECTION_TYPE = "triple"
file_separator = os.path.sep
POSSIBLE_TRAFFIC_LIGHT_CONFIGURATION = int(4)
TOTAL_NUMBER_OF_LANES = int(24)
NUMBER_OF_LANES_TO_OBSERVE = int(TOTAL_NUMBER_OF_LANES / 2)


class TrafficIntersectionEnvTripleLaneGUI(gym.Env):

    CONNECTION_LABEL = 0

    def __init__(self, use_gui: bool = True, total_timesteps: int = 5000, delta_time: int=30, min_green: int=15, max_green: int=120, yellow_time: int=7, sumocfg_file: str = None, network_file: str = None, route_file: str = None,) -> None:

        SUMO_FILES = pathlib.Path(__file__).parents[0].joinpath("sumo-files")
        SMALL_MAP_TRIPLE_LANE = SUMO_FILES.joinpath(f"small-map-{TRAFFIC_INTERSECTION_TYPE}-lane")

        # keeping track for deleting other route files created
        self.passed_route_file = route_file

        if sumocfg_file is None:
            sumocfg_file = pathlib.Path(str(SMALL_MAP_TRIPLE_LANE) + ".sumocfg")
        if network_file is None:
            network_file = pathlib.Path(str(SMALL_MAP_TRIPLE_LANE) + ".net.xml")
        if route_file is None:  # generating new route file if no route file is passed
                                # not choosing the existing one (helpful in multiple environments)
            route_file = generate_routefile(intersection_type=TRAFFIC_INTERSECTION_TYPE, number_of_time_steps=total_timesteps)


        self.sumocfg_file = sumocfg_file
        self.network_file = network_file
        self.route_file = route_file

        self.action_space = gym.spaces.Discrete(
            POSSIBLE_TRAFFIC_LIGHT_CONFIGURATION)
        self.observation_space = gym.spaces.Box(low=0, high=1000, shape=(
            int(NUMBER_OF_LANES_TO_OBSERVE), ), dtype=numpy.float64)
        self.state = numpy.zeros(int(NUMBER_OF_LANES_TO_OBSERVE))

        self.total_timesteps = total_timesteps
        self.delta_time = delta_time
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time

        # The lanes are in order as defined in the network file, changing the order tampers with the observation
        # The order is from west, north, east and south
        # The lanes order is from outermost(0) to innermost(2)
        self.lanes_to_observe = ["E0_0", "E0_1", "E0_2",
                                 "-E1_0", "-E1_1", "-E1_2",
                                 "-E2_0", "-E2_1", "-E2_2",
                                 "-E3_0", "-E3_1", "-E3_2"]

        self.label = TrafficIntersectionEnvTripleLaneGUI.CONNECTION_LABEL
        TrafficIntersectionEnvTripleLaneGUI.CONNECTION_LABEL += 1

        # for reward calculation
        self.last_waiting_time: float = 0
        # for information about vehicles
        self.last_step_vehicle_count: float = 1  # to avoid NaN

        if use_gui:
            sumoBinary = sumolib.checkBinary('sumo-gui')
        else:
            sumoBinary = sumolib.checkBinary('sumo')

        self.sumoBinary = sumoBinary
        self.conn: traci = None

    def step(self, action):

        done = False
        info = {}

        junction_with_lights = "J1"

        current_state = self.conn.trafficlight.getPhase(junction_with_lights)

        turn_yellow = True

        # if it's the same phase, then don't turn on yellow light
        if action == current_state / 2:  # dividing by 2 as green phases is 0, 2, 4, 6
            turn_yellow = False

        # Trying to turn on yellow light
        if turn_yellow:
            self.conn.trafficlight.setPhase(
                junction_with_lights, current_state + 1)

            delta_yellow_time = 0
            while delta_yellow_time < self.yellow_time:

                # Remove if unnecessary
                # Testing if phase duration defined in *.net.xml takes precedence over the traci.setPhase()
                self.conn.trafficlight.setPhase(junction_with_lights, current_state + 1)

                self.conn.simulationStep()

                delta_yellow_time += 1

        # Setting the required phase
        self.conn.trafficlight.setPhase(
            junction_with_lights, action * 2)

        delta_green_time = 0
        while delta_green_time < self.min_green:
            # Remove if unnecessary
            # Testing if phase duration defined in *.net.xml takes precedence over the traci.setPhase()
            self.conn.trafficlight.setPhase(junction_with_lights, action * 2)

            self.conn.simulationStep()
            delta_green_time += 1

        current_simulation_time = self.getSimulationTime()
        if current_simulation_time > self.total_timesteps:
            done = True
            self.reset()

        lanes_observation = numpy.zeros(int(NUMBER_OF_LANES_TO_OBSERVE))
        for i, lane in enumerate(self.lanes_to_observe):
            lanes_observation[i] = self.conn.lane.getLastStepVehicleNumber(lane)

        self.state = lanes_observation

        reward = self.calculate_reward()

        info["waiting_time"] = self.last_waiting_time
        info["last_step_vehicle_count"] = self.last_step_vehicle_count

        return self.state, reward, done, info

    def getSimulationTime(self):
        return self.conn.simulation.getTime()

    def reset(self):


        if self.conn is not None:
            # generating new route file each time
            # deleting previously created route files and saving in that same location.

            route_file_dir = str(self.route_file).rsplit(file_separator, 1)[0]

            SUMO_FILES = pathlib.Path(__file__).parents[0].joinpath("sumo-files")
            SMALL_MAP_DOUBLE_LANE = SUMO_FILES.joinpath(f"small-map-{TRAFFIC_INTERSECTION_TYPE}-lane")

            # deleting old route file 
            if self.route_file != pathlib.Path(str(SMALL_MAP_DOUBLE_LANE) + ".rou.xml") and self.route_file != self.passed_route_file:
                if exists(self.route_file):  # maybe deleted by other
                    try:
                        os.remove(self.route_file)
                    except:
                        print(f"Couldn't remove file {self.route_file}")

            # taking the return file path, as it might be already present and new one is generated
            new_route_file = generate_routefile(intersection_type=TRAFFIC_INTERSECTION_TYPE, number_of_time_steps=self.total_timesteps)
            self.route_file = new_route_file

            traci.switch(self.label)
            traci.close()
            self.conn = None

        sumo_cmd = [self.sumoBinary,
                    '-n', self.network_file,
                    '-r', self.route_file,
                    '--waiting-time-memory', '10000',
                    '--start', '--quit-on-end',
                    "--time-to-teleport", "-1"  # This makes it so that the vehicles won't teleport
                    ]

        traci.start(sumo_cmd, label=self.label)
        self.conn = traci.getConnection(self.label)

        # for reward calculation
        self.last_waiting_time = 0

        # setting the vehicle count on observing lane as observation
        lanes_observation = numpy.zeros(int(NUMBER_OF_LANES_TO_OBSERVE))
        for i, lane in enumerate(self.lanes_to_observe):
            lanes_observation[i] = self.conn.lane.getLastStepVehicleNumber(lane)

        self.state = lanes_observation

        return self.state

    def calculate_waiting_time(self) -> float:
        total_waiting_time = 0
        for lane in self.lanes_to_observe:
            total_waiting_time += self.calculate_waiting_time_of_a_lane(lane)
        return total_waiting_time       

    def calculate_waiting_time_of_a_lane(self, lane: str) -> float:
        last_step_vehicles_ids = self.conn.lane.getLastStepVehicleIDs(lane)

        waiting_time = 0
        for vehicle in last_step_vehicles_ids:
            waiting_time += self.conn.vehicle.getAccumulatedWaitingTime(vehicle)

        return waiting_time
    

    def calculate_reward(self) -> float:

        return self.reward_on_average_waiting_time()


    def reward_on_decrease_in_waiting_time(self) -> float:
        """
            Reward based on difference in cumulative waiting time
        """
        
        total_waiting_time = self.calculate_waiting_time()

        decrease_in_waiting_time = self.last_waiting_time - total_waiting_time
        self.last_waiting_time = total_waiting_time

        return decrease_in_waiting_time

    def reward_on_average_waiting_time(self) -> float:
        """
            reward based on inverse of waiting time * total vehicle count
        """

        # setting the vehicle count on observing lane as observation
        lanes_observation = numpy.zeros(int(NUMBER_OF_LANES_TO_OBSERVE))
        for i, lane in enumerate(self.lanes_to_observe):
            lanes_observation[i] = self.conn.lane.getLastStepVehicleNumber(lane)

        vehicle_count = max(1, numpy.sum(lanes_observation))  # setting a minimum value, to avoid NaNs
        waiting_time = max(1, self.calculate_waiting_time())  # setting a minimum value, to avoid NaNs

        # print(f"Last step vehicle count {self.last_step_vehicle_count}")
        # print(f"Last step waiting time {self.last_waiting_time}")

        reward = vehicle_count/waiting_time

        self.last_step_vehicle_count = vehicle_count
        self.last_waiting_time = waiting_time

        return reward
