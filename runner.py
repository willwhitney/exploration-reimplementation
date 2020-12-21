import os
import subprocess
import asyncio
import yaml
import copy
import numpy as np

GPUS = 4
MULTIPLEX = 4

jobs = [
    "python main.py --eval_every 1 --env gridworld --task default --n_state_bins 40 --env_size 40 --n_action_bins 4 --max_steps 100 --policy uniform --name arxiv_grid40_puniform",
    "python main.py --eval_every 1 --env gridworld --task default --n_state_bins 40 --env_size 40 --n_action_bins 4 --max_steps 100 --policy uniform --no_optimistic_updates --no_optimistic_actions --name arxiv_grid40_noopt_puniform",
    "python main_ablation_slow.py --eval_every 1 --env gridworld --task default --n_state_bins 40 --env_size 40 --n_action_bins 4 --max_steps 100 --policy uniform --name arxiv_grid40_slow_puniform",
    "python main_ablation_intrinsic.py --eval_every 1 --env gridworld --task default --n_state_bins 40 --env_size 40 --n_action_bins 4 --max_steps 100 --explore_only --name arxiv_grid40_intrinsic_noreward",
    "python main.py --eval_every 1 --env gridworld --task default --n_state_bins 40 --env_size 40 --n_action_bins 4 --max_steps 100 --policy uniform --no_exploration --name arxiv_grid40_puniform_noexplore",
]

seeds = [0]

seeded_jobs = []
for seed in seeds:
    for job in jobs:
        job = f"{job}_seed{seed} --seed {seed}"
        seeded_jobs.append(job)


async def run_job(job, gpu_id):
    command = job.split(' ')

    print("Dispatching `{}`".format(command))
    env = {'CUDA_VISIBLE_DEVICES': str(gpu_id),
           'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
           **os.environ}
    proc = await asyncio.create_subprocess_shell(' '.join(command), env=env)
    stdout, stderr = await proc.communicate()

async def run_jobs(jobs):
    tasks = (run_job(job, i % GPUS) for i, job in enumerate(jobs))
    await asyncio.gather(*tasks)

async def main():
    n_parallel = MULTIPLEX * GPUS
    for i in range(0, len(seeded_jobs), n_parallel):
        await run_jobs(seeded_jobs[i: i + n_parallel])

asyncio.run(main())
