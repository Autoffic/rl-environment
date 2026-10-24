from stable_baselines3 import PPO
from stable_baselines3 import A2C
import stable_baselines3.common.env_checker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
import gymnasium as gym
import multiprocessing
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from datetime import datetime

import sys
from pathlib import Path
import os
import numpy as np
import signal

# Training constants
TRAFFIC_INTERSECTION_TYPE = "triple"
TOTAL_TIMESTEPS_FOR_SUMO = 20_000  # per-episode SUMO horizon (shorter -> envs reset
# more often -> less difficulty mixing per rollout,
# and the critic tracks the curriculum faster)
TOTAL_TIMESTEPS_FOR_MODEL = 20_00_000  # call env.step() this many times
EVAL_TIMESTEPS = 1_000

# Lunar-Lander-style curriculum: between-episode difficulty ramps with
# training time (0.05 + very-imbalanced -> 0.4 + uniform), and each route file
# carries a light within-episode ±10% demand ramp.
CURRICULUM = False
UPDATES_PER_STAGE = 1  # advance the difficulty every PPO update (smoothest ramp)

# reward-reform knobs (see TrafficIntersectionEnvTripleLaneGUI.step):
# score is hard-clipped to ±REWARD_CLIP and the spiky quadratic wait-term is cut.
REWARD_CLIP = 200.0
# A/B knob: obs-15 (with elapsed-green) vs obs-14 (AUG18 configuration).
INCLUDE_ELAPSED_GREEN = False
QUEUE_PRESSURE_PENALTY = 0.0
EARLY_TRUNCATION_PENALTY = 0.0

FORCED_SWITCH_PENALTY = 1.0
SWITCH_PENALTY = 15.0
ANTI_STARVATION_THRESHOLD = 60.0
ANTI_STARVATION_PENALTY = 0.25
GENERATE_NEW_ROUTE_FILES = True

# times
MIN_GREEN_TIME = 30
MAX_GREEN_TIME = 120
YELLOW_TIME = 10

# number of environments for training and evaluations
N_TRAIN_ENVS = 8
N_EVAL_ENVS = 2

# Hyper-Parameters
VERBOSE = 1
BATCH_SIZE = 256
N_STEPS = 2048
N_EPOCHS = 10
ENT_COEFF = 0.1
GAMMA = 0.99
LEARNING_RATE = 1e-3
# Evaluation
EVAL_FREQ = N_STEPS  # evaluate after each training session
EVAL_EPISODES = 20  # run evaluation for 20 episodes


def get_env(
    env_name="TrafficIntersectionEnv{}LaneGUI-v1".format(
        TRAFFIC_INTERSECTION_TYPE.capitalize()
    ),
    multi=True,
    use_gui=False,
    subprocess=True,
    n_envs=multiprocessing.cpu_count(),
    min_green=MIN_GREEN_TIME,
    max_green=MAX_GREEN_TIME,
    yellow_time=YELLOW_TIME,
    _seed=1,
    total_timesteps=TOTAL_TIMESTEPS_FOR_SUMO,
    generate_new_route_files: bool = GENERATE_NEW_ROUTE_FILES,
    base_seed: int | None = None,
    forced_switch_penalty: float = FORCED_SWITCH_PENALTY,
    switch_penalty: float = SWITCH_PENALTY,
    anti_starvation_threshold: float = ANTI_STARVATION_THRESHOLD,
    anti_starvation_penalty: float = ANTI_STARVATION_PENALTY,
    early_truncation_penalty: float = EARLY_TRUNCATION_PENALTY,
    passing_reward_weight: float = 1.0,
    waiting_time_reward_weight: float = 0.8,
    reward_scale: float = 100.0,
    reward_clip: float = REWARD_CLIP,
    queue_pressure_penalty: float = QUEUE_PRESSURE_PENALTY,
    curriculum: bool = False,
    curriculum_progress: float | None = None,
    include_elapsed_green: bool = INCLUDE_ELAPSED_GREEN,
):

    from custom_gym.envs.custom_env_dir.TrafficIntersectionEnvTripleLaneGUI import (
        TrafficIntersectionEnvTripleLaneGUI,
    )

    env_kwargs = {
        "use_gui": use_gui,
        "min_green": min_green,
        "max_green": max_green,
        "yellow_time": yellow_time,
        "total_timesteps": total_timesteps,
        "generate_new_route_files": generate_new_route_files,
        "forced_switch_penalty": forced_switch_penalty,
        "switch_penalty": switch_penalty,
        "anti_starvation_threshold": anti_starvation_threshold,
        "anti_starvation_penalty": anti_starvation_penalty,
        "early_truncation_penalty": early_truncation_penalty,
        "passing_reward_weight": passing_reward_weight,
        "waiting_time_reward_weight": waiting_time_reward_weight,
        "reward_scale": reward_scale,
        "reward_clip": reward_clip,
        "queue_pressure_penalty": queue_pressure_penalty,
        "curriculum": curriculum,
        "curriculum_progress": curriculum_progress,
        "include_elapsed_green": include_elapsed_green,
    }

    # startmethod based on the platform, fork for linux and forkserver for windows
    # see: https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods
    start_method = "fork" if sys.platform == "linux" else "spawn"

    if multi:
        if subprocess:

            def make_env(env_id, seed):
                def _init():
                    env_kwargs_per_env = dict(env_kwargs)
                    if base_seed is not None:
                        env_kwargs_per_env["seed"] = base_seed + seed
                    env = Monitor(gym.make(env_id, **env_kwargs_per_env))
                    # env.seed(seed)
                    return env

                return _init

            n_envs = min(
                n_envs, multiprocessing.cpu_count() - 1
            )  # don't allocate more subprocesses than logical cores.
            envs = [make_env(env_name, seed) for seed in range(n_envs)]

            envs = SubprocVecEnv(envs, start_method=start_method)
        else:
            envs = make_vec_env(
                env_name,
                n_envs=n_envs,
                seed=_seed,
                vec_env_cls=DummyVecEnv,
                env_kwargs=env_kwargs,
            )

        return envs
    else:
        env = Monitor(gym.make(env_name, **env_kwargs))
        return env


class TensorboardCallback(BaseCallback):

    def __init__(self, save_path: Path, log_path: Path, verbose=0, save_steps=5000):
        super(TensorboardCallback, self).__init__(verbose)
        self.save_path: Path = save_path
        self.log_path: Path = log_path
        self.save_steps: int = save_steps
        self.max_reward: float = -10e10000000

    def _on_training_start(self) -> None:
        return super()._on_training_start()

    def _on_step(self) -> bool:

        self.logger.record("training:avg_reward", np.average(self.locals["rewards"]))
        self.logger.record(
            "training:average_waiting_time",
            np.average([info["average_waiting_time"] for info in self.locals["infos"]]),
        )
        self.logger.record(
            "training:waiting_time",
            np.average([info["waiting_time"] for info in self.locals["infos"]]),
        )
        self.logger.record(
            "training:vehicle_count",
            np.average(
                [info["last_step_vehicle_count"] for info in self.locals["infos"]]
            ),
        )
        self.logger.record(
            "training:vehicle_entered",
            np.average([info["vehicle_entered"] for info in self.locals["infos"]]),
        )
        self.logger.record(
            "training:forced_switch",
            np.average([info["forced_switch"] for info in self.locals["infos"]]),
        )
        self.logger.record(
            "training:early_truncation",
            np.average([info["early_truncation"] for info in self.locals["infos"]]),
        )
        self.logger.record(
            "training:max_grn_duration",
            np.average([info["max_grn_duration"] for info in self.locals["infos"]]),
        )
        for phase in range(4):
            self.logger.record(
                f"training:max_grn_phase_durn{phase}",
                np.average(
                    [info["max_grn_phase_durn"][phase] for info in self.locals["infos"]]
                ),
            )

        return True

    def _on_training_end(self) -> None:
        return super()._on_training_end()


class CurriculumCallback(BaseCallback):
    """Advance the shared curriculum progress after every 3rd PPO rollout
    update. Envs (fork subprocesses) read the shared value on each reset, so
    the between-episode difficulty rises with training time, not performance."""

    def __init__(self, progress, total_timesteps: int, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)
        self.progress = progress
        self.total_updates = max(int(total_timesteps / (N_STEPS * N_TRAIN_ENVS)), 1)
        self.update_count = 0
        self.progress.value = 0.0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        self.update_count += 1
        if self.update_count % UPDATES_PER_STAGE == 0:
            self.progress.value = min(self.update_count / self.total_updates, 1.0)
        return True


def train():
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]  # traffic-intersection-rl-environment-sumo root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

    env_name = "TrafficIntersectionEnv{}LaneGUI-v1".format(
        TRAFFIC_INTERSECTION_TYPE.capitalize()
    )

    startTime = datetime.now()

    # model stuffs
    modelType = "ppo"
    use_gui = False

    # path to save the model or load the model from
    models_path = Path(str(ROOT) + "/models").resolve()
    log_path = Path(
        str(ROOT)
        + "/logs/{}-trafficintersection-{}-lane-GUI/".format(
            modelType, TRAFFIC_INTERSECTION_TYPE
        )
    ).resolve()

    env_kwargs = {
        "multi": True,
        "subprocess": True,  # mightnot work with spawn start method
    }

    # shared curriculum progress; envs (fork subprocesses) read this at reset
    curriculum_progress = multiprocessing.Value("d", 0.0)

    # environment for training
    env = get_env(
        env_name=env_name,
        **env_kwargs,
        use_gui=use_gui,
        n_envs=N_TRAIN_ENVS,
        generate_new_route_files=GENERATE_NEW_ROUTE_FILES,
        total_timesteps=int(TOTAL_TIMESTEPS_FOR_SUMO),
        curriculum=CURRICULUM,
        curriculum_progress=curriculum_progress,
    )

    # reward-normalization layer: rescales the running return statistics so the
    # shared value function is not chasing the curriculum's shifting reward
    # magnitude (the cause of EV~0 / uniform policy in the Sep-2026 runs).
    # Observations are intentionally NOT normalized (raw, real-world scale, so
    # the trained policy is directly deployable via traffic.py).
    env = VecNormalize(
        env,
        training=True,
        norm_obs=False,
        norm_reward=True,
        clip_reward=10.0,
        gamma=GAMMA,
    )

    # environment for evaluation (fixed 9-grid, no curriculum).
    # Must ALSO be VecNormalize-wrapped (same shape as the training wrapper) or
    # EvalCallback crashes on sync_envs_normalization at the first evaluation.
    # norm_reward=False keeps the eval rewards raw (un-normalized) so
    # eval/mean_reward stays comparable across runs.
    eval_env = get_env(
        env_name=env_name,
        **env_kwargs,
        use_gui=use_gui,
        n_envs=N_EVAL_ENVS,
        total_timesteps=int(EVAL_TIMESTEPS),
        base_seed=42,
    )
    eval_env = VecNormalize(
        eval_env, training=False, norm_obs=False, norm_reward=False, gamma=GAMMA
    )

    # stable_baselines3.common.env_checker.check_env(env, warn=True, skip_render_check=True)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        device="auto",
        verbose=VERBOSE,
        batch_size=BATCH_SIZE,
        n_steps=N_STEPS,
        n_epochs=N_EPOCHS,
        tensorboard_log=log_path,
        ent_coef=ENT_COEFF,
        gamma=GAMMA,
    )

    # model = PPO.load(Path(str(models_path) + "/2022_08_26_20_31_22_136701_TrafficIntersection_{}LaneGUI_{}".format(TRAFFIC_INTERSECTION_TYPE.capitalize(), modelType), env=env).resolve().__str__())
    model.set_env(env)

    # ':' isn't a valid file name in windows, since it is used to distinguish drive
    save_path = Path(
        str(models_path)
        + "/{}-TrafficIntersection-{}LaneGUI-{}".format(
            startTime, TRAFFIC_INTERSECTION_TYPE.capitalize(), modelType
        ).replace(":", "_")
    ).resolve()

    # Use deterministic actions for evaluation   # Also saving the best model
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=EVAL_FREQ,
        log_path=log_path,
        best_model_save_path=save_path,
        deterministic=True,
        render=use_gui,
        verbose=VERBOSE,
        n_eval_episodes=EVAL_EPISODES,
    )

    def save_on_exit(signum, frame):
        # save the model before exiting
        print(f"Saving to {save_path} before exiting.")
        model.save(save_path.joinpath("exit_before_finished.zip"))
        exit(0)

    signal.signal(signal.SIGINT, save_on_exit)

    callback_list = CallbackList(
        [
            eval_callback,
            TensorboardCallback(
                save_path=save_path, log_path=log_path, save_steps=5000
            ),
            CurriculumCallback(
                curriculum_progress, total_timesteps=int(TOTAL_TIMESTEPS_FOR_MODEL)
            ),
        ]
    )
    # ':' isn't a valid name in windows since it is used to differentiate drive
    model.learn(
        total_timesteps=int(TOTAL_TIMESTEPS_FOR_MODEL),
        reset_num_timesteps=False,
        log_interval=1,
        tb_log_name=f"{'GUI' if use_gui else 'CLI'}-{modelType}-{startTime}".replace(
            ":", "_"
        ),
        callback=callback_list,
    )
    # saving the fully trained model
    model.save(save_path.joinpath("last_timestep.zip"))


if __name__ == "__main__":
    train()
