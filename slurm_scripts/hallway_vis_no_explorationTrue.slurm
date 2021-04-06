#!/bin/bash
#SBATCH --job-name=hallway_vis_no_explorationTrue
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/hallway_vis_no_explorationTrue.out
#SBATCH --error=slurm_logs/hallway_vis_no_explorationTrue.err
#SBATCH --export=ALL
#SBATCH --time=2-00
#SBATCH -N 1
#SBATCH --mem=32gb
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --constraint=turing|volta
#SBATCH --exclude=lion[1-26]
#SBATCH --exclude=vine[3-14]
cd .
python -u ../exploration-reimplement-clones/hallway_vis_no_explorationTrue/main.py --eval_every 1 --env hallway --task velocity_4 --max_steps 1000 --no_exploration --density keops_kernel_count --density_state_scale 0.01 --density_action_scale 1 --density_max_obs 32768 --density_tolerance 0.5 --policy sac --policy_updates_per_step 4 --uniform_update_candidates --n_updates_per_step 2 --update_target_every 2 --novelty_discount 0.99 --name hallway_vis_no_explorationTrue &
wait
