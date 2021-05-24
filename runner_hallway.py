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


basename = "hallway_distractors_v13_length4_normal_update1_farther_bigtarget"
grid = [
    {
        # define the task
        "_main": ["main.py"],
        "eval_every": [1],
        "env": ["hallway_distractor"],
        "task": ["velocity_4_distractor", "velocity_4_inverse_distractor",],
        "max_episodes": [500],
        "seed": list(range(4)),
        "no_exploration": [True, False],

        # density settings
        "density": ["keops_kernel_count"],
        "density_state_scale": [0.02],
        "density_action_scale": [1],
        "density_max_obs": [2**15],
        "density_tolerance": [0.9],
        "density_conserve_weight": [True],

        # novelty Q settings
        "uniform_update_candidates": [True],
        "n_updates_per_step": [2],
        "update_target_every": [2],

        # task policy settings
        "policy": ["sac"],
        "policy_updates_per_step": [1],
    },
    # {
    #     # define the task
    #     "_main": ["main_bbe.py"],
    #     "eval_every": [1],
    #     "env": ["hallway_distractor"],
    #     "task": ["velocity_4_distractor", "velocity_4_inverse_distractor",],
    #     "max_episodes": [500],
    #     "seed": list(range(4)),

    #     # density settings
    #     "density": ["keops_kernel_count"],
    #     "density_state_scale": [0.02],
    #     "density_action_scale": [1],
    #     "density_max_obs": [2**15],
    #     "density_tolerance": [0.9],
    #     "density_conserve_weight": [True],

    #     # bbe settings
    #     "bonus_scale": [0.1, 1,],

    #     # task policy settings
    #     "policy": ["sac"],
    #     "policy_updates_per_step": [4],
    # },
]



if __name__ == '__main__':
    jobs = construct_jobs(grid, basename)
    if local:
        asyncio.run(main(jobs, MULTIPLEX=MULTIPLEX, GPUS=GPUS, dry_run=dry_run))
    else:
        slurm_main(jobs, MULTIPLEX=MULTIPLEX, greene=greene, dry_run=dry_run)
