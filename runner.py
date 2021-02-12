import itertools
import os
import subprocess
import asyncio
import yaml
import copy
import numpy as np

GPUS = 4
MULTIPLEX = 4

# jobs = [
#     # no rewards
#     ## gridworld 40
#     "python main.py --eval_every 5 --env gridworld --task default --n_state_bins 40 --env_size 40 --n_action_bins 4 --max_steps 100 --policy uniform --name arxiv2_grid40_puniform",
#     "python main.py --eval_every 5 --env gridworld --task default --n_state_bins 40 --env_size 40 --n_action_bins 4 --max_steps 100 --policy uniform --no_optimistic_updates --no_optimistic_actions --name arxiv2_grid40_noopt_puniform",
#     "python main_ablation_slow.py --eval_every 5 --env gridworld --task default --n_state_bins 40 --env_size 40 --n_action_bins 4 --max_steps 100 --policy uniform --name arxiv2_grid40_slow_puniform",
#     "python main_ablation_intrinsic.py --eval_every 5 --env gridworld --task default --n_state_bins 40 --env_size 40 --n_action_bins 4 --max_steps 100 --explore_only --name arxiv2_grid40_intrinsic_noreward",
#     "python main.py --eval_every 5 --env gridworld --task default --n_state_bins 40 --env_size 40 --n_action_bins 4 --max_steps 100 --policy uniform --no_exploration --name arxiv2_grid40_puniform_noexplore",

#     ## point velocity
#     "python main.py --eval_every 5 --env point --task velocity --n_state_bins 20 --n_action_bins 2 --max_steps 100 --policy uniform --name arxiv2_pv100_puniform",
#     "python main.py --eval_every 5 --env point --task velocity --n_state_bins 20 --n_action_bins 2 --max_steps 100 --policy uniform --no_optimistic_updates --no_optimistic_actions --name arxiv2_pv100_noopt_puniform",
#     "python main_ablation_slow.py --eval_every 5 --env point --task velocity --n_state_bins 20 --n_action_bins 2 --max_steps 100 --policy uniform --name arxiv2_pv100_slow_puniform",
#     "python main_ablation_intrinsic.py --eval_every 5 --env point --task velocity --n_state_bins 20 --n_action_bins 2 --max_steps 100 --explore_only --name arxiv2_pv100_intrinsic_noreward",
#     "python main.py --eval_every 5 --env point --task velocity --n_state_bins 20 --n_action_bins 2 --max_steps 100 --policy uniform --no_exploration --name arxiv2_pv100_puniform_noexplore",

#     # with rewards
#     ## gridworld 40
#     "python main.py --eval_every 5 --env gridworld --task default --n_state_bins 40 --env_size 40 --n_action_bins 4 --max_steps 100 --name arxiv2_grid40",
#     "python main.py --eval_every 5 --env gridworld --task default --n_state_bins 40 --env_size 40 --n_action_bins 4 --max_steps 100 --no_optimistic_updates --no_optimistic_actions --name arxiv2_grid40_noopt",
#     "python main_ablation_slow.py --eval_every 5 --env gridworld --task default --n_state_bins 40 --env_size 40 --n_action_bins 4 --max_steps 100 --name arxiv2_grid40_slow",
#     "python main_ablation_intrinsic.py --eval_every 5 --env gridworld --task default --n_state_bins 40 --env_size 40 --n_action_bins 4 --max_steps 100 --name arxiv2_grid40_intrinsic",
#     "python main.py --eval_every 5 --env gridworld --task default --n_state_bins 40 --env_size 40 --n_action_bins 4 --max_steps 100 --no_exploration --name arxiv2_grid40_noexplore",

#     ## point velocity
#     "python main.py --eval_every 5 --env point --task velocity --n_state_bins 20 --n_action_bins 2 --max_steps 100 --name arxiv2_pv100",
#     "python main.py --eval_every 5 --env point --task velocity --n_state_bins 20 --n_action_bins 2 --max_steps 100 --no_optimistic_updates --no_optimistic_actions --name arxiv2_pv100_noopt",
#     "python main_ablation_slow.py --eval_every 5 --env point --task velocity --n_state_bins 20 --n_action_bins 2 --max_steps 100 --name arxiv2_pv100_slow",
#     "python main_ablation_intrinsic.py --eval_every 5 --env point --task velocity --n_state_bins 20 --n_action_bins 2 --max_steps 100 --name arxiv2_pv100_intrinsic",
#     "python main.py --eval_every 5 --env point --task velocity --n_state_bins 20 --n_action_bins 2 --max_steps 100 --no_exploration --name arxiv2_pv100_noexplore",
# ]



# jobs = [
#     "python main.py --eval_every 1 --env cartpole --task swingup_sparse --density knn_kernel_count --density_state_scale 1e-1 --density_action_scale 1 --max_steps 1000 --policy_lr 1e-4 --name ecp_kkc_plr1e-4_sscale0.1_ascale1",
#     "python main.py --eval_every 1 --env cartpole --task swingup_sparse --density knn_kernel_count --density_state_scale 1e-1 --density_action_scale 1 --max_steps 1000 --policy_lr 1e-4 --name ecp_kkc_plr1e-4_sscale0.1_ascale1",
#     "python main.py --eval_every 1 --env cartpole --task swingup_sparse --density knn_kernel_count --density_state_scale 1e-1 --density_action_scale 1 --max_steps 1000 --policy_lr 1e-4 --name ecp_kkc_plr1e-4_sscale0.1_ascale1",
#     "python main.py --eval_every 1 --env cartpole --task swingup_sparse --density knn_kernel_count --density_state_scale 1e-1 --density_action_scale 1 --max_steps 1000 --policy_lr 1e-4 --name ecp_kkc_plr1e-4_sscale0.1_ascale1",
# ]

excluded_flags = []

basename = "cart_knn"
grid = [
    {
        # define the task
        "_main": ["main.py"],
        "eval_every": [1],
        "env": ["cartpole"],
        "task": ["swingup_sparse"],
        "max_steps": [1000],

        # agent settings
        "density": ["knn_kernel_count"],
        "density_state_scale": [1e-1],
        "density_action_scale": [1],
        "policy_lr": [1e-4],
        "policy_temperature": [3e-1],
        "policy_test_temperature": [1e-1],
    },
    {
        # define the task
        "_main": ["main.py"],
        "eval_every": [1],
        "env": ["cartpole"],
        "task": ["swingup_sparse"],
        "max_steps": [1000],

        # agent settings
        "density": ["knn_kernel_count"],
        "density_state_scale": [1e-1],
        "density_action_scale": [1],
        "policy_lr": [1e-4],
        "policy_temperature": [1e-1],
        "policy_test_temperature": [3e-2],
    },
]


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


def construct_job_string(job):
    """construct the string to execute the job"""
    flagstring = f"python {job['_main']}"
    for flag in job:
        if flag not in excluded_flags and not flag.startswith('_'):
            if isinstance(job[flag], bool):
                if job[flag]:
                    flagstring = flagstring + " --" + flag
                else:
                    print("WARNING: Excluding 'False' flag " + flag)
            else:
                flagstring = flagstring + " --" + flag + " " + str(job[flag])
    return flagstring


def construct_name(job, varying_keys):
    """construct the job's name out of the varying keys in this sweep"""
    jobname = basename
    for flag in job:
        if flag in varying_keys and not flag.startswith('_'):
            jobname = jobname + "_" + flag + str(job[flag])
    return jobname


jobs = construct_jobs(grid)
varying_keys = construct_varying_keys(grid)
job_strings = []
for job in jobs:
    jobname = construct_name(job, varying_keys)
    job_string = construct_job_string(job)
    job_string = job_string + " --name " + jobname
    job_strings.append(job_string)


async def run_job(gpu_id, job):
    command = job.split(' ')

    print("Dispatching `{}`".format(command))
    env = {'CUDA_VISIBLE_DEVICES': str(gpu_id),
           'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
           **os.environ}
    proc = await asyncio.create_subprocess_shell(' '.join(command), env=env)
    stdout, stderr = await proc.communicate()


async def worker_fn(gpu_id, queue):
    while True:
        job = await queue.get()
        await run_job(gpu_id, job)
        queue.task_done()


async def main():
    queue = asyncio.Queue()
    for job in job_strings:
        queue.put_nowait(job)

    n_parallel = MULTIPLEX * GPUS
    workers = []
    for i in range(n_parallel):
        worker = asyncio.create_task(worker_fn(i % GPUS, queue))
        workers.append(worker)

    await queue.join()
    for worker in workers:
        worker.cancel()
    await asyncio.gather(*workers, return_exceptions=True)


asyncio.run(main())
