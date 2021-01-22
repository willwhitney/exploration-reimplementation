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

jobs = [
    # "python main.py --eval_every 1 --env point --task velocity --n_state_bins 20 --n_action_bins 2 --density kernel --max_steps 100 --name explore_pv100_kernel_clippednum",
    # "python main.py --eval_every 1 --env point --task velocity --n_state_bins 20 --n_action_bins 2 --density kernel --density_cov_scale 0.1 --max_steps 100 --name explore_pv100_kernelscale0.1_clippednum",
    # "python main.py --eval_every 1 --env point --task velocity --n_state_bins 20 --n_action_bins 2 --density kernel --density_cov_scale 0.01 --max_steps 100 --name explore_pv100_kernelscale0.01_clippednum",

    # "python main.py --eval_every 1 --env point --task velocity --n_state_bins 20 --n_action_bins 2 --density kernel_count --max_steps 100 --name explore_pv100_kcount_clippednum",
    "python main.py --eval_every 1 --env point --task velocity --n_state_bins 20 --n_action_bins 2 --density kernel_count --density_cov_scale 0.1 --max_steps 100 --name epv100kc_scale0.1",
    "python main.py --eval_every 1 --env point --task velocity --n_state_bins 20 --n_action_bins 2 --density kernel_count --density_cov_scale 0.01 --max_steps 100 --name epv100kc_scale0.01",
    "python main.py --eval_every 1 --env point --task velocity --n_state_bins 20 --n_action_bins 2 --density kernel_count --density_cov_scale 0.001 --max_steps 100 --name epv100kc_scale0.001",
]

seeds = [0]

seeded_jobs = []
for seed in seeds:
    for job in jobs:
        job = f"{job}_seed{seed} --seed {seed}"
        seeded_jobs.append(job)


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
    for job in seeded_jobs:
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
