import itertools
import os
import subprocess
import asyncio
import yaml
import copy
import numpy as np

GPUS = [0, 1]
MULTIPLEX = 1

excluded_flags = []

basename = "pv100_sac_2x"
grid = [
    {
        # define the task
        "_main": ["main_jit_density.py"],
        "eval_every": [1],
        "env": ["point"],
        "task": ["velocity"],
        "max_steps": [100],
        "no_exploration": [True, False]

        # density settings
        "density": ["kernel_count"],
        "density_state_scale": [1e-2],
        "density_action_scale": [1],
        "density_max_obs": [65536],
        "density_tolerance": [0.95],

        # task policy settings
        "policy": ["sac"],
        # "policy_temperature": [3e-1],
        # "policy_test_temperature": [1e-1],
    },
    # {
    #     # define the task
    #     "_main": ["main_jit_density.py"],
    #     "eval_every": [1],
    #     "env": ["cartpole"],
    #     "task": ["swingup_sparse"],
    #     "max_steps": [1000],

    #     # density settings
    #     "density": ["kernel_count"],
    #     "density_state_scale": [1e-1],
    #     "density_action_scale": [1],
    #     "density_max_obs": [65536],
    #     "density_tolerance": [0.6],

    #     # task policy settings
    #     "policy_lr": [1e-4],
    #     "policy_temperature": [1e-1],
    #     "policy_test_temperature": [3e-2],
    # },
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


asyncio.run(main())
