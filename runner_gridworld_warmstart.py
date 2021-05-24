import itertools
import os
import subprocess
import sys
import asyncio
import copy
import glob
import shutil
from pathlib import Path

from runner_utils import main, slurm_main, construct_varying_keys, construct_jobs


local = '--local' in sys.argv
greene = '--greene' in sys.argv
dry_run = '--dry-run' in sys.argv

GPUS = [0, 1, 2, 3]
MULTIPLEX = 4


# basename = "grid40_warmstart_v6_lowlr_updates1"
# grid = [
#     {
#         # define the task
#         "_main": ["main_gridworld_warmstart.py"],
#         "eval_every": [1],
#         "env": ["gridworld"],
#         "task": ["default"],
#         "env_size": [40],
#         "max_steps": [100],
#         "max_episodes": [500],
#         "video_every": [99999999],
#         "warmup_steps": [2000],
#         "seed": list(range(4)),
#         "no_exploration": [True, False],

#         # density settings
#         "n_state_bins": [40],
#         "n_action_bins": [4],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [10],
#         "update_target_every": [10],

#         # task policy settings
#         "policy_updates_per_step": [1],
#         "policy_lr": [1e-4],
#     },
#     {
#         # define the task
#         "_main": ["main_gridworld_warmstart_bbe.py"],
#         "eval_every": [1],
#         "env": ["gridworld"],
#         "task": ["default"],
#         "env_size": [40],
#         "max_steps": [100],
#         "max_episodes": [500],
#         "video_every": [99999999],
#         "warmup_steps": [2000],
#         "seed": list(range(4)),

#         # density settings
#         "n_state_bins": [40],
#         "n_action_bins": [4],

#         # bbe settings
#         "bonus_scale": [0.1, 1,],

#         # task policy settings
#         "policy_updates_per_step": [1],
#         "policy_lr": [1e-4],
#     },
# ]

basename = "grid40_warmstart_v7_savereplay_lowlr_updates1"
grid = [
    {
        # define the task
        "_main": ["main_gridworld_warmstart.py"],
        "eval_every": [1],
        "env": ["gridworld"],
        "task": ["default"],
        "env_size": [40],
        "max_steps": [100],
        "max_episodes": [500],
        "save_replay_every": [20],
        "video_every": [99999999],
        "warmup_steps": [2000],
        "seed": list(range(1, 4)),
        "no_exploration": [True, False],

        # density settings
        "n_state_bins": [40],
        "n_action_bins": [4],

        # novelty Q settings
        "uniform_update_candidates": [True],
        "n_updates_per_step": [10],
        "update_target_every": [10],

        # task policy settings
        "policy_updates_per_step": [1],
        "policy_lr": [1e-4],
    },
    {
        # define the task
        "_main": ["main_gridworld_warmstart_bbe.py"],
        "eval_every": [1],
        "env": ["gridworld"],
        "task": ["default"],
        "env_size": [40],
        "max_steps": [100],
        "max_episodes": [500],
        "save_replay_every": [20],
        "video_every": [99999999],
        "warmup_steps": [2000],
        "seed": list(range(1, 4)),

        # density settings
        "n_state_bins": [40],
        "n_action_bins": [4],

        # bbe settings
        "bonus_scale": [0.1, 1,],

        # task policy settings
        "policy_updates_per_step": [1],
        "policy_lr": [1e-4],
    },
]



if __name__ == '__main__':
    jobs = construct_jobs(grid, basename)
    if local:
        asyncio.run(main(jobs, MULTIPLEX=MULTIPLEX, GPUS=GPUS, dry_run=dry_run))
    else:
        slurm_main(jobs, MULTIPLEX=MULTIPLEX, greene=greene, dry_run=dry_run)
