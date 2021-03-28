import gym

param = {'num_levels': 100, 'distribution_mode': 'hard', 'render': True}
env = gym.make("procgen:procgen-fruitbot-v0", **param)

obs = env.reset()
while True:
    action = env.action_space.sample()
    obs, rewards, done, info = env.step(action)
    env.render()

    if done:
        break
