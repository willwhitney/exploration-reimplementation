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
MULTIPLEX = 2


# basename = "pv100_sacqex_v2"
# grid = [
#     {
#         # define the task
#         "_main": ["main_sac_qex.py"],
#         "eval_every": [1],
#         "env": ["point"],
#         "task": ["velocity"],
#         "max_episodes": [500],
#         "max_steps": [100],
#         "seed": list(range(8)),

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.02],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.95],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac_qex"],
#         "policy_updates_per_step": [1],
#     },
# ]

# basename = "pv100_bbe_v4_updates"
# grid = [
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [1],
#         "env": ["point"],
#         "task": ["velocity"],
#         "max_episodes": [500],
#         "seed": list(range(8)),
#         "no_exploration": [False],

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.02],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.95],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac"],
#         "policy_updates_per_step": [4],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [4],
#         "update_target_every": [4],
#     },
#     {
#         # define the task
#         "_main": ["main_bbe.py"],
#         "eval_every": [1],
#         "env": ["point"],
#         "task": ["velocity"],
#         "max_episodes": [500],
#         "seed": list(range(8)),
#         "no_exploration": [False],

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.02],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.95],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac"],
#         "policy_updates_per_step": [4],
#     },
# ]

# basename = "wex_walk_narrow_sparse_v3_seeds"
# grid = [
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [1],
#         "env": ["walker_explore"],
#         "task": ["walk_narrow_sparse"],
#         "seed": list(range(4)),
#         "no_exploration": [True, False],

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [5e-1],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.5],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac"],
#         "policy_updates_per_step": [1],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [2],
#         "update_target_every": [2],
#     },
# ]

# basename = "walker_walk_v1_seeds"
# grid = [
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [1],
#         "env": ["walker"],
#         "task": ["walk"],
#         "seed": list(range(4)),
#         "no_exploration": [True, False],

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [5e-1],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.5],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac"],
#         "policy_updates_per_step": [1],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [2],
#         "update_target_every": [2],
#     },
# ]

# basename = "finger_all_v6_scale_low"
# grid = [
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [1],
#         "env": ["finger_explore"],
#         "task": ["turn_hard_narrow"],
#         "seed": list(range(8)),
#         "no_exploration": [True, False],

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.1],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.5],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac"],
#         "policy_updates_per_step": [1],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [2],
#         "update_target_every": [2],
#     },
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [1],
#         "env": ["finger"],
#         "task": ["turn_hard"],
#         "seed": list(range(8)),
#         "no_exploration": [True, False],

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.1],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.5],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac"],
#         "policy_updates_per_step": [1],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [2],
#         "update_target_every": [2],
#     },
# ]

# finger_all_v1arrow_seed0_no_explorationTrue--rrow_seed0_no_explorationFalse.slurm
# basename = "finger_all_v2_rerun"
# grid = [
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [5],
#         "env": ["finger_explore"],
#         "task": ["turn_hard_narrow"],
#         "seed": [0],
#         "no_exploration": [True, False],

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.34],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.9],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac"],
#         "policy_updates_per_step": [1],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [2],
#         "update_target_every": [2],
#     },
# ]

# basename = "fex_hard_v3_seeds"
# grid = [
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [5],
#         "env": ["finger_explore"],
#         "task": ["turn_hard_narrow"],
#         "seed": list(range(4)),
#         "no_exploration": [True, False],

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [1e-1],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.5],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac"],
#         "policy_updates_per_step": [1],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [2],
#         "update_target_every": [2],
#     },
# ]

# basename = "finger_v2"
# grid = [
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [5],
#         "env": ["finger"],
#         "task": ["turn_hard"],
#         "seed": list(range(4)),
#         "no_exploration": [True, False],

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [1e-1],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.5],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac"],
#         "policy_updates_per_step": [1],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [2],
#         "update_target_every": [2],
#     },
# ]


# basename = "bice_v6_seeds"
# grid = [
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [1],
#         "env": ["ball_in_cup", "ball_in_cup_explore"],
#         "task": ["catch"],
#         "no_exploration": [True, False],
#         "seed": list(range(4)),

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.24],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.9],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac"],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [2],
#         "update_target_every": [2],
#     },
# ]


# basename = "reacher_v10_correct"
# grid = [
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [1],
#         "env": ["reacher_explore"],
#         "task": ["hard_narrow_init"],
#         "max_episodes": [500],
#         "no_exploration": [True, False],
#         "seed": list(range(8)),

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.06],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.9],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac"],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [2],
#         "update_target_every": [2],
#     },
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [1],
#         "env": ["reacher"],
#         "task": ["hard"],
#         "max_episodes": [500],
#         "no_exploration": [True, False],
#         "seed": list(range(8)),

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.06],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.9],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac"],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [2],
#         "update_target_every": [2],
#     },
# ]

# reacher_explore_v8_seedshard_no_explorationFalse_seed4--hard_no_explorationFalse_seed5
# basename = "reacher_explore_v9_rerun"
# grid = [
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [1],
#         "env": ["reacher_explore"],
#         "task": ["hard"],
#         "max_episodes": [500],
#         "no_exploration": [False],
#         "seed": [4, 5],

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.18],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.9],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac"],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [2],
#         "update_target_every": [2],
#     },
# ]


# basename = "hallway_all_v4"
# grid = [
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [1],
#         "env": ["hallway"],
#         "task": ["velocity_1", "velocity_4_inverse_distractor"],
#         "max_episodes": [100],
#         "no_exploration": [True, False],
#         "seed": list(range(4)),

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.068],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.9],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac"],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [2],
#         "update_target_every": [2],
#     },
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [1],
#         "env": ["hallway"],
#         "task": ["velocity_4", "velocity_4_distractor"],
#         "max_episodes": [300],
#         "no_exploration": [True, False],
#         "seed": list(range(4)),

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.068],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.9],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac"],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [2],
#         "update_target_every": [2],
#     },
# ]



# basename = "hallway_vis_v2"
# grid = [
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [1],
#         "env": ["hallway"],
#         "task": ["velocity_4"],
#         "video_every": [1],
#         "no_exploration": [False, True],
#         # "seed": [0, 1],

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.01],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.5],

#         # task policy settings
#         "policy": ["sac"],
#         "policy_updates_per_step": [4],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [2],
#         "update_target_every": [2],
#     },
# ]

# basename = "pv100_v10_greene"
# grid = [
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [1],
#         "env": ["point"],
#         "task": ["velocity"],
#         "seed": list(range(2)),
#         "no_exploration": [True, False],

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [6e-2],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.95],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac"],
#         "policy_updates_per_step": [4],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [2],
#         "update_target_every": [2],
#     },
# ]

basename = "bbe_all_v3_moreseeds"
grid = [

    # reacher
    {
        # define the task
        "_main": ["main_bbe.py"],
        "eval_every": [1],
        "env": ["reacher_explore"],
        "task": ["hard_narrow_init"],
        "max_episodes": [500],
        "seed": list(range(4, 8)),

        # density settings
        "density": ["keops_kernel_count"],
        "density_state_scale": [0.06],
        "density_action_scale": [1],
        "density_max_obs": [2**15],
        "density_tolerance": [0.9],
        "density_conserve_weight": [True],

        # task policy settings
        "policy": ["sac"],
    },
    {
        # define the task
        "_main": ["main_bbe.py"],
        "eval_every": [1],
        "env": ["reacher"],
        "task": ["hard"],
        "max_episodes": [500],
        "seed": list(range(4, 8)),

        # density settings
        "density": ["keops_kernel_count"],
        "density_state_scale": [0.06],
        "density_action_scale": [1],
        "density_max_obs": [2**15],
        "density_tolerance": [0.9],
        "density_conserve_weight": [True],

        # task policy settings
        "policy": ["sac"],
    },

    # ball-in-cup
    {
        # define the task
        "_main": ["main_bbe.py"],
        "eval_every": [1],
        "env": ["ball_in_cup", "ball_in_cup_explore"],
        "task": ["catch"],
        "seed": list(range(4, 8)),

        # density settings
        "density": ["keops_kernel_count"],
        "density_state_scale": [0.078],
        "density_action_scale": [1],
        "density_max_obs": [2**15],
        "density_tolerance": [0.9],
        "density_conserve_weight": [True],

        # task policy settings
        "policy": ["sac"],
    },

    # finger
    {
        # define the task
        "_main": ["main_bbe.py"],
        "eval_every": [1],
        "env": ["finger_explore"],
        "task": ["turn_hard_narrow"],
        "seed": list(range(4, 8)),

        # density settings
        "density": ["keops_kernel_count"],
        "density_state_scale": [0.11],
        "density_action_scale": [1],
        "density_max_obs": [2**15],
        "density_tolerance": [0.5],
        "density_conserve_weight": [True],

        # task policy settings
        "policy": ["sac"],
        "policy_updates_per_step": [1],
    },
    {
        # define the task
        "_main": ["main_bbe.py"],
        "eval_every": [1],
        "env": ["finger"],
        "task": ["turn_hard"],
        "seed": list(range(4, 8)),

        # density settings
        "density": ["keops_kernel_count"],
        "density_state_scale": [0.11],
        "density_action_scale": [1],
        "density_max_obs": [2**15],
        "density_tolerance": [0.5],
        "density_conserve_weight": [True],

        # task policy settings
        "policy": ["sac"],
        "policy_updates_per_step": [1],
    },

    # walker
    {
        # define the task
        "_main": ["main_bbe.py"],
        "eval_every": [1],
        "env": ["walker_explore"],
        "task": ["walk_narrow_sparse"],
        "seed": list(range(4, 8)),

        # density settings
        "density": ["keops_kernel_count"],
        "density_state_scale": [0.16],
        "density_action_scale": [1],
        "density_max_obs": [2**15],
        "density_tolerance": [0.5],
        "density_conserve_weight": [True],

        # task policy settings
        "policy": ["sac"],
        "policy_updates_per_step": [1],
    },
    {
        # define the task
        "_main": ["main_bbe.py"],
        "eval_every": [1],
        "env": ["walker"],
        "task": ["walk"],
        "seed": list(range(4, 8)),

        # density settings
        "density": ["keops_kernel_count"],
        "density_state_scale": [0.16],
        "density_action_scale": [1],
        "density_max_obs": [2**15],
        "density_tolerance": [0.5],
        "density_conserve_weight": [True],

        # task policy settings
        "policy": ["sac"],
        "policy_updates_per_step": [1],
    },
]

# basename = "ufo_all_v3_moreseeds"
# grid = [

#     # reacher
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [1],
#         "env": ["reacher_explore"],
#         "task": ["hard_narrow_init"],
#         "max_episodes": [500],
#         "seed": list(range(4, 8)),

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.06],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.9],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac"],
#         "policy_updates_per_step": [1],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [2],
#         "update_target_every": [2],
#     },
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [1],
#         "env": ["reacher"],
#         "task": ["hard"],
#         "max_episodes": [500],
#         "seed": list(range(4, 8)),

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.06],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.9],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac"],
#         "policy_updates_per_step": [1],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [2],
#         "update_target_every": [2],
#     },

#     # ball-in-cup
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [1],
#         "env": ["ball_in_cup", "ball_in_cup_explore"],
#         "task": ["catch"],
#         "seed": list(range(4, 8)),

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.078],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.9],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac"],
#         "policy_updates_per_step": [1],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [2],
#         "update_target_every": [2],
#     },

#     # finger
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [1],
#         "env": ["finger_explore"],
#         "task": ["turn_hard_narrow"],
#         "seed": list(range(4, 8)),

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.11],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.5],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac"],
#         "policy_updates_per_step": [1],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [2],
#         "update_target_every": [2],
#     },
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [1],
#         "env": ["finger"],
#         "task": ["turn_hard"],
#         "seed": list(range(4, 8)),

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.11],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.5],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac"],
#         "policy_updates_per_step": [1],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [2],
#         "update_target_every": [2],
#     },

#     # walker
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [1],
#         "env": ["walker_explore"],
#         "task": ["walk_narrow_sparse"],
#         "seed": list(range(4, 8)),

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.16],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.5],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac"],
#         "policy_updates_per_step": [1],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [2],
#         "update_target_every": [2],
#     },
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [1],
#         "env": ["walker"],
#         "task": ["walk"],
#         "seed": list(range(4, 8)),

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.16],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.5],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac"],
#         "policy_updates_per_step": [1],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [2],
#         "update_target_every": [2],
#     },
# ]


# basename = "sacqex_all_v2"
# grid = [

#     # reacher
#     {
#         # define the task
#         "_main": ["main_sac_qex.py"],
#         "eval_every": [1],
#         "env": ["reacher_explore"],
#         "task": ["hard_narrow_init"],
#         "max_episodes": [500],
#         "seed": list(range(4)),

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.06],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.9],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac_qex"],
#     },
#     {
#         # define the task
#         "_main": ["main_sac_qex.py"],
#         "eval_every": [1],
#         "env": ["reacher"],
#         "task": ["hard"],
#         "max_episodes": [500],
#         "seed": list(range(4)),

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.06],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.9],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac_qex"],
#     },

#     # ball-in-cup
#     {
#         # define the task
#         "_main": ["main_sac_qex.py"],
#         "eval_every": [1],
#         "env": ["ball_in_cup", "ball_in_cup_explore"],
#         "task": ["catch"],
#         "seed": list(range(4)),

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.078],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.9],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac_qex"],
#     },

#     # finger
#     {
#         # define the task
#         "_main": ["main_sac_qex.py"],
#         "eval_every": [1],
#         "env": ["finger_explore"],
#         "task": ["turn_hard_narrow"],
#         "seed": list(range(4)),

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.11],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.5],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac_qex"],
#         "policy_updates_per_step": [1],
#     },
#     {
#         # define the task
#         "_main": ["main_sac_qex.py"],
#         "eval_every": [1],
#         "env": ["finger"],
#         "task": ["turn_hard"],
#         "seed": list(range(4)),

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.11],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.5],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac_qex"],
#         "policy_updates_per_step": [1],
#     },

#     # walker
#     {
#         # define the task
#         "_main": ["main_sac_qex.py"],
#         "eval_every": [1],
#         "env": ["walker_explore"],
#         "task": ["walk_narrow_sparse"],
#         "seed": list(range(4)),

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.16],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.5],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac_qex"],
#         "policy_updates_per_step": [1],
#     },
#     {
#         # define the task
#         "_main": ["main_sac_qex.py"],
#         "eval_every": [1],
#         "env": ["walker"],
#         "task": ["walk"],
#         "seed": list(range(4)),

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.16],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.5],
#         "density_conserve_weight": [True],

#         # task policy settings
#         "policy": ["sac_qex"],
#         "policy_updates_per_step": [1],
#     },
# ]

# basename = "hallway_midstart_all_v1"
# grid = [
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [1],
#         "env": ["hallway_midstart"],
#         "task": ["velocity_4_offset_p5", "velocity_4_offset_1",
#                  "velocity_4_offset_1p5", "velocity_4_offset_2",],
#         "max_episodes": [500],
#         "seed": list(range(4)),
#         "no_exploration": [True, False],

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.02],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.9],
#         "density_conserve_weight": [True],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [2],
#         "update_target_every": [2],

#         # task policy settings
#         "policy": ["sac"],
#     },
#     {
#         # define the task
#         "_main": ["main_bbe.py"],
#         "eval_every": [1],
#         "env": ["hallway_midstart"],
#         "task": ["velocity_4_offset_p5", "velocity_4_offset_1",
#                  "velocity_4_offset_1p5", "velocity_4_offset_2",],
#         "max_episodes": [500],
#         "seed": list(range(4)),

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.02],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.9],
#         "density_conserve_weight": [True],

#         # bbe settings
#         "bonus_scale": [0.1, 1,],

#         # task policy settings
#         "policy": ["sac"],
#     },
# ]



if __name__ == '__main__':
    jobs = construct_jobs(grid, basename)
    if local:
        asyncio.run(main(jobs, MULTIPLEX=MULTIPLEX, GPUS=GPUS, dry_run=dry_run))
    else:
        slurm_main(jobs, MULTIPLEX=MULTIPLEX, greene=greene, dry_run=dry_run)
