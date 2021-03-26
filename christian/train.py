import os
import argparse

import gym
import numpy as np
import matplotlib.pyplot as plt

import gym
from stable_baselines3 import PPO, HER, A2C, DQN

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    
    set_random_seed(seed)
    return _init

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num tsteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean r: {:.2f} - Last mean r per ep: {:.2f}".format(
                        self.best_mean_reward, mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                        self.model.save(self.save_path)

        return True
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="PPO, DQN, HER, or A2C")
    args = parser.parse_args()
    
    if args.model.lower() not in ['ppo', 'dqn', 'her', 'a2c']:
        assert False, "[ERROR] Invalid model specified"
    
    # Create log dir
    model = args.model.lower()
    log_dir = f"logs/tmp_mp_{model}/"
    os.makedirs(log_dir, exist_ok=True)

    # Setting up environment
    param = {
        'num_levels': 100, 
        'distribution_mode': 'easy', 
        'render': False
    }
    env_id = "procgen:procgen-fruitbot-v0"
    if model in ['ppo', 'a2c']:
        # For Vectorized Processes
        num_cpu = 12
        env = make_vec_env(env_id, 
                           n_envs=num_cpu, 
                           monitor_dir=log_dir,
                           env_kwargs=param)
    else:
        # For Non-Vectorized Processes
        env = gym.make(env_id, **param)
        env = Monitor(env, log_dir)

    # PPO
    if model == 'ppo':
        model = PPO("CnnPolicy", env, verbose=1)
    
    # HER
    elif model == 'her':
        n_sampled_goal = 4
        model = HER(
            "CnnPolicy",
            env,
            DQN,
            n_sampled_goal=n_sampled_goal,
            goal_selection_strategy="future",

            # IMPORTANT: because the env is not wrapped with a 
            #            TimeLimit wrapper
            # we have to manually specify the max number of 
            # steps per episode

            max_episode_length=100,
            verbose=1,
            buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95,
            batch_size=256,
            online_sampling=True,
            policy_kwargs=dict(net_arch=[256, 256, 256]),
        )
    
    # DQN
    elif model == 'dqn':
        model = DQN("CnnPolicy", env, verbose=1)
        
    # A2C
    else:
        model = A2C("CnnPolicy", env, verbose=1)
    
    callback = SaveOnBestTrainingRewardCallback(
        check_freq=1000, log_dir=log_dir)
    timesteps = 1e6 * 5
    model.learn(total_timesteps=timesteps, 
                callback=callback)
