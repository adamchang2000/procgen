import gym3
from procgen import ProcgenGym3Env
import numpy as np

RENDER = True

def main():
    if RENDER:
        env = ProcgenGym3Env(num=2, env_name="fruitbot", render_mode="rgb_array")
        env = gym3.ViewerWrapper(env, info_key="rgb")
    else:
        env = ProcgenGym3Env(num=2, env_name="fruitbot")

    step = 0

    while True:
        env.act(gym3.types_np.sample(env.ac_space, bshape=(env.num,)))
        rew, obs, first = env.observe()

        print(obs.keys())

        print(f"step {step} reward {rew} first {first}")
        step += 1

if __name__ == "__main__":
    main()