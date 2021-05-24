import itertools
import os
import subprocess
import sys
import asyncio
import copy
import glob
import shutil
from pathlib import Path


excluded_flags = []
CODE_DIR = '..'


def construct_varying_keys(grids):
    all_keys = set().union(*[g.keys() for g in grids])
    merged = {k: set() for k in all_keys}
    for grid in grids:
        for key in all_keys:
            grid_key_value = grid[key] if key in grid else ["<<NONE>>"]
            merged[key] = merged[key].union(grid_key_value)
    varying_keys = {key for key in merged if len(merged[key]) > 1}
    return varying_keys


def construct_name(job, varying_keys, basename=''):
    """construct the job's name out of the varying keys in this sweep"""
    job_name = basename
    for flag in job:
        if flag in varying_keys and not flag.startswith('_'):
            job_name = job_name + "_" + flag + str(job[flag])
    return job_name


def construct_jobs(grids, basename):
    jobs = []
    for grid in grids:
        individual_options = [[{key: value} for value in values]
                              for key, values in grid.items()]
        product_options = list(itertools.product(*individual_options))
        jobs += [{k: v for d in option_set for k, v in d.items()}
                 for option_set in product_options]

    varying_keys = construct_varying_keys(grids)

    for job in jobs:
        job['_name'] = construct_name(job, varying_keys, basename)
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


def run_job_slurm(step_jobs, greene=False, dry_run=False):
    if len(step_jobs) == 0:
        return
    # create slurm dirs if needed
    slurm_log_dir = 'slurm_logs'
    slurm_script_dir = 'slurm_scripts'
    os.makedirs(slurm_script_dir, exist_ok=True)
    os.makedirs(slurm_log_dir, exist_ok=True)

    job_strings = []
    job_names = []
    for job in step_jobs:
        # construct job name
        job_name = job['_name']

        # copy code to a temp directory
        true_source_dir = '.'
        job_source_dir = f'{CODE_DIR}/exploration-reimplement-clones/{job_name}/'
        os.makedirs(job_source_dir, exist_ok=True)
        copy_job_source(job_source_dir)

        # make the job command
        job_string = construct_job_string(job, job_name, source_dir=job_source_dir)
        job_names.append(job_name)
        job_strings.append(job_string)

    common_prefix = os.path.commonprefix(job_names)
    shortened_job_names = []
    for job_name in job_names:
        shortened_name = job_name[len(common_prefix): len(common_prefix) + 40]
        shortened_job_names.append(shortened_name)
    job_name = common_prefix + '--'.join(shortened_job_names)



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
        # slurmfile.write("#SBATCH --time=4:00:00\n")
        slurmfile.write("#SBATCH -N 1\n")
        slurmfile.write("#SBATCH --mem=32gb\n")

        slurmfile.write("#SBATCH --gres=gpu:1\n")

        if not greene:
            slurmfile.write("#SBATCH -c 8\n")
            slurmfile.write("#SBATCH --constraint=turing|volta\n")
            slurmfile.write("#SBATCH --exclude=lion[1-26]\n")
            slurmfile.write("#SBATCH --exclude=vine[3-14]\n")
            slurmfile.write("cd " + true_source_dir + '\n')
            for job_string in job_strings:
                slurmfile.write(f'{job_string} &\n')
            slurmfile.write('wait\n')

        if greene:
            slurmfile.write("#SBATCH -c 8\n")
            slurmfile.write("cd " + true_source_dir + '\n')
            slurmfile.write('singularity exec --nv --overlay /scratch/work/public/singularity/mujoco200-dep-cuda11.1-cudnn8-ubunutu18.04.sqf:ro,$SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "')
            slurmfile.write('source /home/ww1114/.bashrc\n')
            slurmfile.write('echo XLA_PYTHON_CLIENT_PREALLOCATE $XLA_PYTHON_CLIENT_PREALLOCATE\n')
            slurmfile.write('echo `hostname`\n')
            slurmfile.write('nvidia-smi > /tmp/`hostname`.txt\n')
            slurmfile.write('cat /tmp/`hostname`.txt\n')
            # slurmfile.write('source /ext3/env.sh\n')
            slurmfile.write('fish\n')
            slurmfile.write('conda activate jax\n')
            for job_string in job_strings:
                slurmfile.write(f'{job_string} &\n')
            slurmfile.write('wait\n')
            slurmfile.write('"\n')

    # run the slurm script
    for job_string in job_strings:
        print("Dispatching `{}`".format(job_string))

    if not dry_run:
        os.system(f'sbatch {slurm_script_path} &')


async def run_job(gpu_id, job, dry_run=False):
    job_name = job['_name']
    job_string = construct_job_string(job, job_name)
    # job_string = job_string + " --name " + job_name

    print("Dispatching `{}`".format(job_string))
    env = {
        **os.environ,
        'CUDA_VISIBLE_DEVICES': str(gpu_id),
        'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
    }
    if not dry_run:
        proc = await asyncio.create_subprocess_shell(job_string, env=env)
        stdout, stderr = await proc.communicate()


async def worker_fn(gpu_id, queue, dry_run=False):
    while True:
        job = await queue.get()
        await run_job(gpu_id, job, dry_run=dry_run)
        queue.task_done()


async def main(jobs, MULTIPLEX=2, GPUS=[0,1,2,3], dry_run=False):
    queue = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)

    n_parallel = MULTIPLEX * len(GPUS)
    workers = []
    for i in range(n_parallel):
        gpu_id = GPUS[i % len(GPUS)]
        worker = asyncio.create_task(worker_fn(gpu_id, queue, dry_run=dry_run))
        workers.append(worker)

    await queue.join()
    for worker in workers:
        worker.cancel()
    await asyncio.gather(*workers, return_exceptions=True)


def slurm_main(jobs, MULTIPLEX=2, greene=False, dry_run=False):
    for i in range(0, len(jobs), MULTIPLEX):
        step_jobs = jobs[i: i + MULTIPLEX]
        run_job_slurm(step_jobs, greene=greene, dry_run=dry_run)