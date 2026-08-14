# traffic-intersection-rl-environment-sumo

A custom OpenAI-Gym environment around a [SUMO](https://eclipse-sumo.org/) simulation
of a small signalized intersection, with reinforcement-learning agents (PPO/A2C)
trained with [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).
The agent chooses the traffic-light phase from per-lane vehicle counts and is
rewarded for keeping vehicle waiting time low; a traditional fixed-cycle
controller is included as a non-RL baseline.

Registered environments (in `custom_gym`):

| Gym ID | Map |
|---|---|
| `TrafficIntersectionEnvSingleLaneGUI-v1` | single-lane approaches |
| `TrafficIntersectionEnvDoubleLaneGUI-v1` | double-lane approaches |
| `TrafficIntersectionEnvTripleLaneGUI-v1` | triple-lane approaches (default) |

## Project layout

```
custom_gym/                     installable package (pip install -e .)
  envs/custom_env_dir/
    TrafficIntersectionEnv{Single,Double,Triple}LaneGUI.py   SUMO + TraCI environments
    generateRouteFile.py                                    demand / route generation
scripts/
  train.py                      PPO/A2C training
  traffic.py                    run a trained model (or no-RL control) + lane logging
  generateRouteFile.py          standalone route-file generation
  setupLaneCounting.py          traditional fixed-cycle controller baseline
sumo-files/                     networks, routes and .sumocfg for the lane maps
requirements.txt
```

## Meeting Requirements

```sh
pip install -r requirements.txt
```

Pins include `gym==0.21.0`, `stable-baselines3==1.6.0`, `eclipse-sumo==1.14.1`,
and `torch>=1.13.1`.

## Installing the environment

```sh
cd custom_gym
pip install -e .
cd ..
```

## Setup Environment
`source ./setup.sh`

## Start training

```sh
python scripts/train.py
```

PPO/A2C training against the SUMO intersection. Runs are configured by the
constants at the top of `scripts/train.py` — intersection type
(`TRAFFIC_INTERSECTION_TYPE`), the SUMO per-episode horizon, total model
timesteps, and the min-green/yellow timings.

## See the results

```sh
python scripts/traffic.py
```

Runs the intersection with a trained model (or fixed-cycle control) and logs
per-lane vehicle statistics to `logging_outputs/`. Useful flags:
`--logging-off`, `--convert-to-csv`, `--generate-both-outputs`.

## Non-RL baseline

```sh
python scripts/setupLaneCounting.py
```

The traditional lane-counting controller with fixed min-green and yellow
intervals, used as the comparison point for the learned policies.

## For GPU support

You need to uninstall the PyTorch CPU version if installed already and install the
GPU version: https://pytorch.org/get-started/locally/

### Problem with requirements.txt?

You may change the version or the name of the dependency to match your platform.
