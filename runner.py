import itertools
import os
import subprocess
import sys
import asyncio
import copy
import numpy as np
import glob
import shutil
from pathlib import Path


local = '--local' in sys.argv

GPUS = [0, 1, 2, 3]
MULTIPLEX = 2
CODE_DIR = '..'

excluded_flags = []

# basename = "bice_v1"
# grid = [
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [1],
#         "env": ["ball_in_cup", "ball_in_cup_explore"],
#         "task": ["catch"],
#         "max_steps": [1000],
#         "no_exploration": [True, False],
#         "seed": [0],

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.1],
#         "density_action_scale": [1],
#         "density_max_obs": [2**16],
#         "density_tolerance": [0.5],

#         # task policy settings
#         "policy": ["sac"],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [2],
#         "update_target_every": [2],
#     },
# ]


# basename = "reacher_explore_v3_smalltarget"
# grid = [
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [1],
#         "env": ["reacher_explore"],
#         "task": ["hard", "hard_fixed_init"],
#         "max_steps": [1000],
#         "no_exploration": [True, False],
#         "seed": [0],

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [0.1],
#         "density_action_scale": [1],
#         "density_max_obs": [2**16],
#         "density_tolerance": [0.5],

#         # task policy settings
#         "policy": ["sac"],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [2],
#         "update_target_every": [2],
#     },
# ]

# basename = "hallway_inverse_distractor_v2_closer"
# grid = [
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [1],
#         "env": ["hallway"],
#         "task": ["velocity_4_inverse_distractor"],
#         "max_steps": [1000],
#         "no_exploration": [False, True],
#         "seed": [0, 1],

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
#         "novelty_discount": [0.99],
#     },
# ]

basename = "manipulator_explore_v8_reachlift"
grid = [
    {
        # define the task
        "_main": ["main.py"],
        "eval_every": [1],
        "env": ["manipulator_explore"],
        "task": ["reach_lift_ball"],
        "max_steps": [1000],
        "max_episodes": [10000],
        "no_exploration": [True, False],
        "seed": [0],

        # density settings
        "density": ["keops_kernel_count"],
        "density_state_scale": [0.3, 1],
        "density_action_scale": [1],
        "density_max_obs": [2**15],
        "density_tolerance": [0.5],

        # task policy settings
        "policy": ["sac"],
        "policy_updates_per_step": [1, 2],

        # novelty Q settings
        "uniform_update_candidates": [True],
        "n_updates_per_step": [2],
        "update_target_every": [2],
    },
]

# basename = "pv100_v4_sacupdates"
# grid = [
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [1],
#         "env": ["point"],
#         "task": ["velocity"],
#         "max_steps": [100],
#         "seed": [0],
#         "no_exploration": [True, False],

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [1e-2],
#         "density_action_scale": [1],
#         "density_max_obs": [2**16],
#         "density_tolerance": [0.95],

#         # task policy settings
#         "policy": ["sac"],
#         "policy_updates_per_step": [1, 4],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [2],
#         "update_target_every": [2],
#     },
# ]


# basename = "acrobot_swingup"
# grid = [
#     {
#         # define the task
#         "_main": ["main_jit_density.py"],
#         "eval_every": [1],
#         "env": ["acrobot"],
#         "task": ["swingup"],
#         "max_steps": [1000],

#         # density settings
#         "density": ["kernel_count"],
#         "density_state_scale": [3e-1],
#         "density_action_scale": [1],
#         "density_max_obs": [4096],
#         "density_tolerance": [0.4],

#         # task policy settings
#         "policy": ["sac"],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#     },
# ]

# basename = "hopper_keops_5update_density_state_scale0.3_seeds"
# grid = [
    # {
    #     # define the task
    #     "_main": ["main.py"],
    #     "eval_every": [5],
    #     "env": ["hopper"],
    #     "task": ["hop"],
    #     "max_steps": [1000],
    #     "no_exploration": [False],
    #     "seed": [0],

    #     # density settings
    #     "density": ["keops_kernel_count"],
    #     "density_state_scale": [0.1],
    #     "density_action_scale": [1],
    #     "density_max_obs": [2**15],
    #     "density_tolerance": [0.2, 0.5],

    #     # task policy settings
    #     "policy": ["sac"],

    #     # novelty Q settings
    #     "uniform_update_candidates": [True],
    #     "n_updates_per_step": [5],
    #     "update_target_every": [5],
    # },
    # {
    #     # define the task
    #     "_main": ["main.py"],
    #     "eval_every": [5],
    #     "env": ["hopper"],
    #     "task": ["hop"],
    #     "max_steps": [1000],
    #     "no_exploration": [False],
    #     "seed": [0, 1, 2, 3],

    #     # density settings
    #     "density": ["keops_kernel_count"],
    #     "density_state_scale": [0.3],
    #     "density_action_scale": [1],
    #     "density_max_obs": [2**15],
    #     "density_tolerance": [0.4],

    #     # task policy settings
    #     "policy": ["sac"],

    #     # novelty Q settings
    #     "uniform_update_candidates": [True],
    #     "n_updates_per_step": [5],
    #     "update_target_every": [5],
    # },
# ]


# basename = "manipulator_keops_v4_updates1"
# grid = [
#     {
#         # define the task
#         "_main": ["main.py"],
#         "eval_every": [1],
#         "env": ["manipulator"],
#         "task": ["bring_ball"],
#         "max_steps": [1000],
#         "max_episodes": [10000],
#         "no_exploration": [False],
#         "seed": [0],

#         # density settings
#         "density": ["keops_kernel_count"],
#         "density_state_scale": [1, 0.5],
#         "density_action_scale": [1],
#         "density_max_obs": [2**15],
#         "density_tolerance": [0.1],

#         # task policy settings
#         "policy": ["sac"],

#         # novelty Q settings
#         "uniform_update_candidates": [True],
#         "n_updates_per_step": [1],
#         "update_target_every": [1],
#     },
# ]


def construct_varying_keys(grids):
    all_keys = set().union(*[g.keys() for g in grids])
    merged = {k: set() for k in all_keys}
    for grid in grids:
        for key in all_keys:
            grid_key_value = grid[key] if key in grid else ["<<NONE>>"]
            merged[key] = merged[key].union(grid_key_value)
    varying_keys = {key for key in merged if len(merged[key]) > 1}
    return varying_keys


def construct_jobs(grids):
    jobs = []
    for grid in grids:
        individual_options = [[{key: value} for value in values]
                              for key, values in grid.items()]
        product_options = list(itertools.product(*individual_options))
        jobs += [{k: v for d in option_set for k, v in d.items()}
                 for option_set in product_options]
    return jobs


def construct_job_string(job, name, source_dir=''):
    """construct the string to execute the job"""
    flagstring = f"python -u {source_dir}{job['_main']}"
    for flag in job:
        if flag not in excluded_flags and not flag.startswith('_'):
            if isinstance(job[flag], bool):
                if job[flag]:
                    flagstring = flagstring + " --" + flag
                else:
                    print("WARNING: Excluding 'False' flag " + flag)
            else:
                flagstring = flagstring + " --" + flag + " " + str(job[flag])
    return flagstring + ' --name ' + name


def construct_name(job, varying_keys):
    """construct the job's name out of the varying keys in this sweep"""
    job_name = basename
    for flag in job:
        if flag in varying_keys and not flag.startswith('_'):
            job_name = job_name + "_" + flag + str(job[flag])
    return job_name


def copy_job_source(target_dir):
    # exclude the results dir since it's large and scanning takes forever
    # note that this syntax is extremely dumb!
    # [!r] will exclude every directory that starts with 'r'
    patterns = ['*.xml', '[!r]*/**/*.xml',
                '*.py', '[!r]*/**/*.py']

    for pattern in patterns:
        for f in Path('.').glob(pattern):
            target_path = f'{target_dir}{f}'
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            # print(f"Copying {f} to {target_path}.")
            shutil.copy(f, target_path)


def run_job_slurm(job):
    # construct job name
    job_name = construct_name(job, varying_keys)

    # create slurm dirs if needed
    slurm_log_dir = 'slurm_logs'
    slurm_script_dir = 'slurm_scripts'
    os.makedirs(slurm_script_dir, exist_ok=True)
    os.makedirs(slurm_log_dir, exist_ok=True)

    # copy code to a temp directory
    true_source_dir = '.'
    job_source_dir = f'{CODE_DIR}/exploration-reimplement-clones/{job_name}/'
    os.makedirs(job_source_dir, exist_ok=True)
    copy_job_source(job_source_dir)

    # make the job command
    job_string = construct_job_string(job, job_name, source_dir=job_source_dir)

    # write a slurm script
    slurm_script_path = f'{slurm_script_dir}/{job_name}.slurm'
    with open(slurm_script_path, 'w') as slurmfile:
        slurmfile.write("#!/bin/bash\n")
        slurmfile.write(f"#SBATCH --job-name={job_name}\n")
        slurmfile.write("#SBATCH --open-mode=append\n")
        slurmfile.write(f"#SBATCH --output=slurm_logs/{job_name}.out\n")
        slurmfile.write(f"#SBATCH --error=slurm_logs/{job_name}.err\n")
        slurmfile.write("#SBATCH --export=ALL\n")
        # slurmfile.write("#SBATCH --signal=USR1@600\n")
        slurmfile.write("#SBATCH --time=2-00\n")
        slurmfile.write("#SBATCH -N 1\n")
        slurmfile.write("#SBATCH --mem=32gb\n")

        slurmfile.write("#SBATCH -c 4\n")
        slurmfile.write("#SBATCH --gres=gpu:1\n")
        slurmfile.write("#SBATCH --constraint=turing|volta\n")
        slurmfile.write("#SBATCH --exclude=lion[1-26]\n")

        slurmfile.write("cd " + true_source_dir + '\n')
        slurmfile.write(f"{job_string} &\n")
        slurmfile.write("wait\n")

    # run the slurm script
    print("Dispatching `{}`".format(job_string))
    os.system(f'sbatch {slurm_script_path} &')


async def run_job(gpu_id, job):
    job_name = construct_name(job, varying_keys)
    job_string = construct_job_string(job, job_name)
    job_string = job_string + " --name " + job_name

    print("Dispatching `{}`".format(job_string))
    env = {
        **os.environ,
        'CUDA_VISIBLE_DEVICES': str(gpu_id),
        'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
    }
    proc = await asyncio.create_subprocess_shell(job_string, env=env)
    stdout, stderr = await proc.communicate()


async def worker_fn(gpu_id, queue):
    while True:
        job = await queue.get()
        await run_job(gpu_id, job)
        queue.task_done()


async def main():
    queue = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)

    n_parallel = MULTIPLEX * len(GPUS)
    workers = []
    for i in range(n_parallel):
        gpu_id = GPUS[i % len(GPUS)]
        worker = asyncio.create_task(worker_fn(gpu_id, queue))
        workers.append(worker)

    await queue.join()
    for worker in workers:
        worker.cancel()
    await asyncio.gather(*workers, return_exceptions=True)


def slurm_main():
    for job in jobs:
        run_job_slurm(job)

if __name__ == '__main__':
    jobs = construct_jobs(grid)
    varying_keys = construct_varying_keys(grid)
    if local:
        asyncio.run(main())
    else:
        slurm_main()
