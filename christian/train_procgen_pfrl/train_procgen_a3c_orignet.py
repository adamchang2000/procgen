import os

# Prevent numpy from using multiple threads
os.environ["OMP_NUM_THREADS"] = "1" # WHY??

from procgen import ProcgenEnv
from vec_env import VecExtractDictObs
from vec_env import VecMonitor
from vec_env import VecNormalize
from vec_env import VecSqueezeObs
import argparse
from util import logger

import torch
from torch import nn
import pfrl
from pfrl import experiments, utils
from train_agent_async import train_agent_async
import a3c
from pfrl.optimizers import SharedRMSpropEpsInsideSqrt
from pfrl.policies import SoftmaxCategoricalHead

# from policies import ImpalaCNN, MultiBatchImpalaCNN

def parse_args():
    parser = argparse.ArgumentParser(
      description='Process procgen training arguments.')

    # Experiment parameters.
    parser.add_argument(
        '--distribution-mode', type=str, default='easy',
        choices=['easy', 'hard', 'exploration', 'memory', 'extreme'])
    parser.add_argument('--env-name', type=str, default='fruitbot')
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--num-levels', type=int, default=0)
    parser.add_argument('--start-level', type=int, default=0)
    parser.add_argument('--num-threads', type=int, default=4)
    parser.add_argument('--exp-name', type=str, default='trial01')
    parser.add_argument('--log-dir', type=str, default='./log')
    parser.add_argument('--model-file', type=str, default=None)
    parser.add_argument('--method-label', type=str, default='vanilla')

    # PPO parameters.
    parser.add_argument('--gpu', type=int, default=0)
    # parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--clip-range', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--nsteps', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--nepochs', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=25_000_000)
    parser.add_argument('--save-interval', type=int, default=100)


    # A3C parameters.
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--t-max", type=int, default=5)
    parser.add_argument("--beta", type=float, default=1e-2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30 * 60 * 60,  # 30 minutes with 60 fps
        help="Maximum number of frames for each episode.",
    )
    parser.add_argument("--eval-interval", type=int, default=2500)
    parser.add_argument("--eval-n-steps", type=int, default=125000)
    parser.add_argument("--eval-n-episodes", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="./exp_async_results/a3c")


    configs = parser.parse_args(args=[
      '--num-envs',      '1', # For A3C, it might be wise to keep this as 1
      '--processes',     '4',
      '--num-levels',    '500',
      '--start-level',   '100',
      '--num-threads',   '8',
      '--eval-interval', '50_000',
      '--exp-name',      'a3c-trail02-eval_interval=50_000_bb=orig',
      '--out-dir',       './exp_async_results/a3c_eval-interval=50_000_bb=orig',
      '--method-label',  'vanilla',
      '--max-steps',     '5_000_000'
    ])
    configs.steps = configs.max_steps
    return configs

def create_venv(config, is_valid=False):
    venv = ProcgenEnv(
        num_envs=config.num_envs,
        env_name=config.env_name,
        num_levels=0 if is_valid else config.num_levels,
        start_level=0 if is_valid else config.start_level,
        distribution_mode=config.distribution_mode,
        num_threads=config.num_threads,
    )
    venv = VecSqueezeObs(venv, "rgb")
    # venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    return VecNormalize(venv=venv, ob=False)

def lr_setter(env, agent, value):
    for pg in agent.optimizer.param_groups:
        assert "lr" in pg
        pg["lr"] = value

class ModelWrapper(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(obs_size, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 256),
            nn.ReLU()
        )

        self.dist_gen =SoftmaxCategoricalHead()
        self.logit_fc = nn.Linear(256, n_actions)
        self.value_fc = nn.Linear(256, 1)

    def forward(self, x):
        values = []
        logits = []

        for i in range(len(x)):
            xi = x[i]
            xi = self.model(xi)
            logits.append(self.logit_fc(xi))
            values.append(self.value_fc(xi))

        dist = self.dist_gen(torch.stack(logits))
        values = torch.stack(values)
        return dist, values

def main():
    configs = parse_args()

    train_venv = create_venv(configs, is_valid=False)
    # valid_venv = create_venv(configs, is_valid=True)
    obs_size = train_venv.observation_space.low.shape[0]
    n_actions = train_venv.action_space.n   

    # model = nn.Sequential(
    #     nn.Conv2d(obs_size, 16, 8, stride=4),
    #     nn.ReLU(),
    #     nn.Conv2d(16, 32, 4, stride=2),
    #     nn.ReLU(),
    #     nn.Flatten(),
    #     nn.Linear(2592, 256),
    #     nn.ReLU(),
    #     pfrl.nn.Branched(
    #         nn.Sequential(
    #             nn.Linear(256, n_actions),
    #             SoftmaxCategoricalHead(),
    #         ),
    #         nn.Linear(256, 1),
    #     ),
    # )

    model = ModelWrapper(obs_size, n_actions)

    opt = SharedRMSpropEpsInsideSqrt(
      model.parameters(), lr=7e-4, eps=1e-1, alpha=0.99)
    assert opt.state_dict()["state"], (
        "To share optimizer state across processes, the state must be"
        " initialized before training."
    )

    agent = a3c.A3C(
        model=model,
        phi=lambda x : x,
        optimizer=opt,
        t_max=configs.t_max,
        gamma=0.99,
        beta=configs.beta,
        max_grad_norm=40.0   
    )

    lr_decay_hook = experiments.LinearInterpolationHook(
        configs.steps, configs.lr, 0, lr_setter
    )

    # Configure logger.
    log_dir = os.path.join(
        configs.log_dir,
        configs.env_name,
        'nlev_{}_{}'.format(configs.num_levels, configs.distribution_mode),
        configs.method_label,
        configs.exp_name,
    )
    os.makedirs(configs.out_dir, exist_ok=True)
    logger.configure(dir=log_dir, format_strs=['csv', 'stdout'])

    # Editted original train_agent_async due to info correction
    # info = info[0] <=== Accesses current info
    train_agent_async(
        agent=agent,
        outdir=configs.out_dir,
        processes=configs.processes,
        make_env=lambda pidx, test: create_venv(configs, is_valid=test),
        profile=configs.profile,
        steps=configs.steps,
        eval_n_steps=None, # configs.eval_n_steps,
        eval_n_episodes=configs.eval_n_episodes,
        eval_interval=configs.eval_interval,
        global_step_hooks=[lr_decay_hook],
        save_best_so_far_agent=True
    )

if __name__ == '__main__':
    main()
