import os
import gym
import argparse

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="PPO, DQN, HER, or A2C")
    parser.add_argument("-r", "--reward", default='', help="Put 'new' for new reward usage")
    parser.add_argument("-l", "--load_dir", default='logs', help='Path to load directory')
    parser.add_argument("-s", "--save_dir", default='logs_eval', help="Path to save directory")
    parser.add_argument("-v", "--view_res", default=False, type=lambda x : bool(x), help="Render")
    args = parser.parse_args()
    
    assert args.model.lower() in ['ppo', 'dqn', 'her', 'a2c'], "[ERROR] Invalid model specified"
    assert args.reward.lower() in ['new', ''], "[ERROR] Invalid reward specified"
    
    model = args.model.lower()
    base_dir = args.save_dir
    load_dir = args.load_dir
    reward_type = args.reward
    render = args.view_res
    
    # Create log dir
    log_dir = f"{base_dir}/tmp_{model}_reward={reward_type}/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment
    param = {
        'start_level': 0,
        'num_levels': 100, 
        'distribution_mode': 'easy', 
        'render': render
    }
    env_id = "procgen:procgen-fruitbot-v0"
    env = gym.make(env_id, **param)
    env = Monitor(env, log_dir)
    env.observation_space.seed(0)
    env.action_space.seed(0)
    set_random_seed(0)

    # Load the trained agent
    model_fn = None
    if model == 'ppo':
        model_fn = PPO
    if model == 'dqn':
        model_fn = DQN
    if model == 'a2c':
        model_fn = A2C
    
    Model = model_fn.load(f'{load_dir}/best_model.zip', env=env)
    mean_reward, std_reward = evaluate_policy(
        Model, Model.get_env(), n_eval_episodes=100, deterministic=True)

    print(f"[Results: {model}, Reward Training Type: {reward_type}]",
          f"Eval Mean Reward: {mean_reward:.3f},", 
          f"Eval STD Reward: {std_reward:.3f}")
    
    if render:
        obs = env.reset()
        for _ in range(1000):
            action, _states = Model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            env.render()