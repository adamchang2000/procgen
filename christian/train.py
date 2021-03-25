import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.policies import CnnPolicy

param = {
    'num_levels': 100, 
    'distribution_mode': 'hard', 
    'render': True
}
env = gym.make("procgen:procgen-fruitbot-v0", **param)
env = DummyVecEnv([lambda: env])

model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=50_000, log_interval=10)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
