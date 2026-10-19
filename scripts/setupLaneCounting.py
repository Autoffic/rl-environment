import time
from pathlib import Path
import sys

import train
import os

startTime:int = 0
endTime:int = 500000

trafficLightSwitchingTime: int = train.MIN_GREEN_TIME
yellowLightTime: int = train.YELLOW_TIME
lanes = ["E0_0", "E0_1", "E0_2",
         "-E1_0", "-E1_1", "-E1_2",
         "-E2_0", "-E2_1", "-E2_2",
         "-E3_0", "-E3_1", "-E3_2"]

'''
    The process of traffic light switching is as follows:
        turn on yellow light for yellowLightTime
        turn on green light for trafficLightSwitchingTime
    and repeat.


    so logging the lanes data in the interval of 40 timesteps should give the
    vehicle count at the time of traffic light switching

    Either by predefined configuration or by the model.

    the parameter areaDetector determines whether lane area detector is used.
    Lane area detector gives desired output and is more accurate for counting total entered and leaving vehicles.
'''


def setupLaneCounting(areaDetector: bool = True, begin: int = startTime, end: int = endTime,
                    id: str = f"lanecount-{time.time()}",
                    trafficLightSwitchingTime:int = trafficLightSwitchingTime,
                    yellowLightTime:int = yellowLightTime,
                    outputFile: str | None = None,
                    additionalFileGenerationPath: str | None = None,
                    lanes = lanes):

    
    # to change to project relative paths and properly resolve paths in different platforms
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1].absolute()  # traffic-intersection-rl-environment-sumo root directory
    SUMO_FILES = ROOT.joinpath("sumo-files/").resolve()

    if(outputFile == None):
        outputFile = SUMO_FILES.joinpath(id).resolve().__str__()

    if not os.path.exists(str(SUMO_FILES)):
        os.mkdir(str(SUMO_FILES))

    if(additionalFileGenerationPath == None):
        additionalFileGenerationPath = SUMO_FILES.joinpath("countlanes.add.xml").resolve()

    if not Path.exists(additionalFileGenerationPath):
        open(additionalFileGenerationPath, "x")
    
    if areaDetector:
        period = yellowLightTime + trafficLightSwitchingTime

        with open(additionalFileGenerationPath, "w") as addfile:
            print("<additional>", file=addfile)

            for lane in lanes:
                print(f'\t<laneAreaDetector id="id_{lane}" lane="{lane}" period="{period}" file="{outputFile}"/>'
                    , file=addfile)


            print("</additional>", file=addfile)

    else:

        with open(additionalFileGenerationPath, "w") as addfile:
            print("<additional>\n", file=addfile)

            previousStep: int = begin + yellowLightTime # here light switching is required for the first time
            nextStep: int = previousStep + trafficLightSwitchingTime # here the green period ends and yellow period starts
            jumpSteps: int = trafficLightSwitchingTime + yellowLightTime # the timestep when traffic light switching is required

            for i in range(begin, end, jumpSteps):
                print(f'<laneData id="{id}" file="{outputFile}" begin="{previousStep}" end="{nextStep}"/>', file=addfile)
                previousStep = nextStep + yellowLightTime  # changing to when the green light starts
                nextStep = previousStep + trafficLightSwitchingTime # changing to when the green light ends
                
                # here, vehicle passing during yellow light time isn't calculated, as it can be often unpredictable in real world
        
            print("</additional>", file=addfile)
    return addfile

if __name__=="__main__":
    setupLaneCounting()
