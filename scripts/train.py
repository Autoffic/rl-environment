from stable_baselines3 import PPO
from stable_baselines3 import A2C
import stable_baselines3.common.env_checker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import gym
import multiprocessing
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from datetime import date, datetime

import sys
from pathlib import Path
import os
import numpy as np


# Training constants
TRAFFIC_INTERSECTION_TYPE="triple"
TOTAL_TIMESTEPS_FOR_SUMO=79000
TOTAL_TIMESTEPS_FOR_MODEL=1000000 # call env.step() this many times

#times
MIN_GREEN_TIME=30
YELLOW_TIME=10


def get_env(env_name = "TrafficIntersectionEnv{}LaneGUI-v1".format(TRAFFIC_INTERSECTION_TYPE.capitalize()), multi=True, use_gui=False, subprocess=True, n_envs=multiprocessing.cpu_count(), min_green=30, yellow_time=10, _seed=1, total_timesteps=TOTAL_TIMESTEPS_FOR_SUMO):

    from custom_gym.envs.custom_env_dir.TrafficIntersectionEnvTripleLaneGUI import TrafficIntersectionEnvTripleLaneGUI

    env_kwargs = {
        "use_gui": use_gui,
        "min_green": min_green,
        "yellow_time": yellow_time,
        "total_timesteps":total_timesteps
    }

    # startmethod based on the platform, fork for linux and forkserver for windows
    # see: https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods
    start_method = "fork" if sys.platform=="linux" else "forkserver"

    if multi:
        if subprocess:
            def make_env(env_id, seed):
                def _init():
                    env = gym.make(env_id, **env_kwargs)
                    env.seed(seed)
                    return env
                return _init

            n_envs = min(n_envs, multiprocessing.cpu_count())  # don't allocate more subprocesses than logical cores.
            envs= [make_env(env_name, seed) for seed in range(n_envs)]

            envs = SubprocVecEnv(envs, start_method=start_method)
        else:
            envs = make_vec_env(env_name, n_envs=n_envs, seed=_seed, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
        
        return envs
    else:
        env = gym.make(env_name, **env_kwargs)
        return env
    

class TensorboardCallback(BaseCallback):
 
    def __init__(self, save_path, verbose=0, save_steps = 5000):
        super(TensorboardCallback, self).__init__(verbose)
        self.save_path = save_path
        self.save_steps = save_steps
        self.max_reward = -10e+10000000

    def _on_step(self) -> bool:
        
        if self.num_timesteps % self.save_steps == 0 and np.average(self.locals["rewards"]) > self.max_reward:
            self.max_reward = max(self.max_reward, np.average(self.locals["rewards"]))
            self.model.save(f"{self.save_path}-{self.num_timesteps}")
        self.logger.record("training:avg_reward", np.average(self.locals["rewards"]))


def train():
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]  # traffic-intersection-rl-environment-sumo root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

    env_name = "TrafficIntersectionEnv{}LaneGUI-v1".format(TRAFFIC_INTERSECTION_TYPE.capitalize())

    startTime = datetime.now()

    # model stuffs
    modelType = "ppo"
    use_gui = False

    # path to save the model or load the model from
    models_path = Path(str(ROOT) + "/models").resolve()
    log_path = Path(str(ROOT) + "/logs/{}-trafficintersection-{}-lane-GUI/".format(modelType, TRAFFIC_INTERSECTION_TYPE)).resolve()

    # environment for training
    env = get_env(env_name=env_name, multi=True, subprocess=True, use_gui=use_gui, n_envs=multiprocessing.cpu_count() - 2 )

    # environment for evaluation
    eval_env = get_env(env_name=env_name, multi=True, subprocess=True, use_gui=use_gui, n_envs=2, total_timesteps=128)
    
    # stable_baselines3.common.env_checker.check_env(env, warn=True, skip_render_check=True)

    model = PPO("MlpPolicy", env, device="auto", verbose=1, batch_size=64, n_steps=256, n_epochs=10, tensorboard_log=log_path, ent_coef=0.1)

    # model = PPO.load(Path(str(models_path) + "/2022_08_26_20_31_22_136701_TrafficIntersection_{}LaneGUI_{}".format(TRAFFIC_INTERSECTION_TYPE.capitalize(), modelType), env=env).resolve().__str__())
    model.set_env(env)

    save_path = Path(str(models_path) + "/{}-TrafficIntersection-{}LaneGUI-{}".format(startTime, TRAFFIC_INTERSECTION_TYPE.capitalize(), modelType)).resolve()

    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, eval_freq=256, log_path=log_path,
                                    deterministic=True, render=False, verbose=1)

    callback_list = CallbackList([eval_callback, TensorboardCallback(save_path=save_path, save_steps=5000)])
    model.learn(total_timesteps=int(TOTAL_TIMESTEPS_FOR_MODEL), reset_num_timesteps=False, log_interval=1, tb_log_name=f"{'GUI' if use_gui else 'CLI'}-{modelType}-{startTime}", callback=callback_list)



if __name__=="__main__":
    train()