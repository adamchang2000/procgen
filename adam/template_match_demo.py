import gym3
from procgen import ProcgenGym3Env
import numpy as np
import cv2

from load_templates import load_templates

RENDER = True

def template_match(obs, templates):
    """template match against RGB window, observation of shape (nxmx3)"""

    rgb = obs['rgb'][0]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) #convert rgb to bgr
    print(bgr.dtype)

    names = []
    filter_out = np.zeros((len(list(templates.keys())), bgr.shape[0], bgr.shape[1]))

    idx = 0
    for k, v in templates.items():
        names.append(k)
        kernel = v['img']
        dst = np.zeros((bgr.shape[0], bgr.shape[1]))
        for i in range(3):
            a = cv2.filter2D(bgr[:,:,i], -1, kernel[:,:,i])
            dst += a
        filter_out[idx] = dst
        idx += 1

    filter_out = filter_out.transpose((1, 2, 0))
    filter_out_max = np.max(filter_out, axis=-1)
    filter_out_amax = np.argmax(filter_out, axis=-1)

    print(filter_out.shape)
    print(filter_out_max.shape)
    print(filter_out_amax.shape)

    print(filter_out_max[:5,:5])
    print(filter_out_amax[:5,:5])
        
    return {}

def main():
    if RENDER:
        env = ProcgenGym3Env(num=2, env_name="fruitbot", render_mode="rgb_array")
        env = gym3.ViewerWrapper(env, info_key="rgb")
    else:
        env = ProcgenGym3Env(num=2, env_name="fruitbot")

    step = 0

    templates = load_templates()
    while True:
        env.act(gym3.types_np.sample(env.ac_space, bshape=(env.num,)))
        rew, obs, first = env.observe()
        template_match(obs, templates)
        print(f"step {step} reward {rew} first {first}")
        step += 1

if __name__ == "__main__":
    main()