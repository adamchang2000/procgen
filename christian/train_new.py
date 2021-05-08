import os
import argparse
from typing import Any, Callable, Dict, List, Optional, Union

import gym
import numpy as np
import matplotlib.pyplot as plt

import gym
from stable_baselines3 import PPO, A2C, DQN

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, \
    EvalCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy

from CustomCNNPolicy import CustomCNN

def make_env(env_id, rank, log_dir, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        try:
            env.seed(seed + rank)
        except:
            print("[WARN] Can't set random seed for env...")
        env = Monitor(env, log_dir)
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
        self.save_path = os.path.join(log_dir, 'best_train_model')
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
                    print('[TRAIN CHECKPOINT]')
                    print("Num tsteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean r: {:.2f} - Last mean r per ep: {:.2f}".format(
                        self.best_mean_reward, mean_reward))
                    if mean_reward <= self.best_mean_reward:
                        print()

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                        self.model.save(self.save_path)
                        print()

        return True
    
class SaveOnBestTestRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, eval_env, n_eps, verbose=1):
        super(SaveOnBestTestRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_eval_model')
        self.best_mean_reward = -np.inf
        self.eval_env = eval_env
        self.n_eps = n_eps
        
        # Buffers
        self._is_success_buffer = []

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            
    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        info = locals_["info"]
        
        if not isinstance(info, dict):
            info = info[0]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eps,
                render=False,
                deterministic=True,
                return_episode_rewards=True,
                warn=True,
                callback=self._log_success_callback,
            )
            
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward
            
            if self.verbose > 0:
                if len(self._is_success_buffer) > 0:
                    success_rate = np.mean(self._is_success_buffer)
                    if len(self._is_success_buffer) > 100:
                        self._is_success_buffer = self._is_success_buffer[1:]
                else:
                    success_rate = 0.0
                
                print('[EVAL CHECKPOINT]')
                print("Num tsteps: {}".format(self.num_timesteps))
                print(
                    "Best mean r: {:.2f} - Last mean r per ep: {:.2f}".format(
                    self.best_mean_reward, mean_reward))
                print(
                    "Curr Reward: {:.2f} +/- {:.2f}".format(
                    mean_reward, std_reward))
                print(
                    "Curr Ep Length: {:.2f} +/- {:.2f}".format(
                    mean_ep_length, std_ep_length))
                print(
                    "Success Rate: {:.2f}, Buffer Len={}".format(
                        success_rate, len(self._is_success_buffer))
                )
                if mean_reward <= self.best_mean_reward:
                        print()

            # New best model, you could save the agent here
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                # Example for saving best model
                if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)
                    print()

        return True
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="PPO, DQN, HER, or A2C")
    parser.add_argument("-r", "--reward", default='', 
                        help="Put 'new' for new reward usage")
    parser.add_argument("-c", "--cnn_policy", default='', 
                        help="Specify CNN Policy")
    parser.add_argument("-s", "--save_dir", default='logs', 
                        help="Path to save directory")
    parser.add_argument("-g", "--gpu", default='0', 
                        help="GPU Device")
    args = parser.parse_args()
    
    assert args.model.lower() in ['ppo', 'dqn', 'her', 'a2c'], \
        "[ERROR] Invalid model specified"
    assert args.reward.lower() in ['new', ''], \
        "[ERROR] Invalid reward specified"
    assert args.cnn_policy.lower() in ['', 'adam_v1'], \
        "[ERROR] Invalid CNN Policy specified"

    # Create log dir
    model_name = args.model.lower()
    cnn_policy = args.cnn_policy.lower()
    base_dir = args.save_dir
    reward_type = args.reward
    gpu = args.gpu

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    log_dir = f"{base_dir}/{model_name}_reward={reward_type}_cnn={cnn_policy}/"
    train_dir = f'{log_dir}/train'
    eval_dir = f'{log_dir}/eval'
    check_dir = f'{log_dir}/check'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(check_dir, exist_ok=True)

    # Setting up environment
    param = {
        'num_levels': 100, 
        'distribution_mode': 'easy', 
        'render': False
    }
    
    env_id = f"procgen:procgen-fruitbot{reward_type}-v0"
    set_random_seed(0)
    if model_name in ['ppo', 'a2c']:
        # For Vectorized Processes
        num_cpu = 12
        env = make_vec_env(env_id,
                           n_envs=num_cpu, 
                           monitor_dir=train_dir,
                           env_kwargs=param)
        
        eval_id = "procgen:procgen-fruitbot-v0"
        eval_env = gym.make(eval_id, **param)
        eval_env = Monitor(eval_env, eval_dir)
        
    else:
        # For Non-Vectorized Processes
        env = gym.make(env_id, **param)
        env = Monitor(env, train_dir)
        
        eval_id = "procgen:procgen-fruitbot-v0"
        eval_env = gym.make(eval_id, **param)
        eval_env = Monitor(eval_env, eval_dir)

    # PPO
    if model_name == 'ppo':
        if cnn_policy == 'adam_v1':
            policy_kwargs = dict(
                features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=128)
            )
            model = PPO(
                "CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
        else:
            model = PPO("CnnPolicy", env, verbose=1)
    
    # DQN
    elif model_name == 'dqn':
        if cnn_policy == 'adam_v1':
            policy_kwargs = dict(
                features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=128)
            )
            model = DQN(
                "CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
        else:
            model = DQN("CnnPolicy", env, verbose=1)
        
    # A2C
    else:
        if cnn_policy == 'adam_v1':
            policy_kwargs = dict(
                features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=128)
            )
            model = A2C(
                "CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
        else:
            model = A2C("CnnPolicy", env, verbose=1)
    
    trainCallback = SaveOnBestTrainingRewardCallback(
        check_freq=1000, log_dir=train_dir)
    evalCallback = SaveOnBestTestRewardCallback(
        check_freq=1000, log_dir=eval_dir,
        eval_env=eval_env, n_eps=100
    )
    callback = CallbackList([evalCallback, trainCallback])
    
    timesteps = 1e6 * 5
    if os.path.isdir(train_dir) and os.path.isfile(
      os.path.join(train_dir, 'best_train_model.zip')):
      
      model.load(os.path.join(train_dir, 'best_train_model'), env=env)
      model.learn(total_timesteps=timesteps, callback=callback, 
                  reset_num_timesteps=False)
    else:
      model.learn(total_timesteps=timesteps, 
                  callback=callback)






