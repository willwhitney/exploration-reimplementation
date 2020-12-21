# Sample-efficient exploration

## Experiments in arXiv paper

### No rewards

Gridworld:
- Our method (factored + fast + optimistic): `python main.py --eval_every 1 --env gridworld --task default --n_state_bins 20 --env_size 20 --n_action_bins 4 --max_steps 100 --policy uniform --name grid20_puniform`
- No optimism: `python main.py --eval_every 1 --env gridworld --task default --n_state_bins 20 --env_size 20 --n_action_bins 4 --max_steps 100 --policy uniform --no_optimistic_updates --no_optimistic_actions --name grid20_noopt_puniform`
- Slow adaptation: `python main_ablation_slow.py --eval_every 1 --env gridworld --task default --n_state_bins 20 --env_size 20 --n_action_bins 4 --max_steps 100 --policy uniform --name grid20_slow_puniform`
- Intrinsic rewards (not factored + slow adaptation + no optimism): `python main_ablation_intrinsic.py --eval_every 1 --env gridworld --task default --n_state_bins 20 --env_size 20 --n_action_bins 4 --max_steps 100 --explore_only --name grid20_intrinsic_noreward`
- No exploration: `python main.py --eval_every 1 --env gridworld --task default --n_state_bins 20 --env_size 20 --n_action_bins 4 --max_steps 100 --policy uniform --no_exploration --name grid20_puniform_noexplore`

Point velocity:
- Our method (factored + fast + optimistic): `python main.py --eval_every 1 --env point --task velocity --n_state_bins 20 --n_action_bins 2 --max_steps 100 --policy uniform --name pv100_puniform`
- No optimism: `python main.py --eval_every 1 --env point --task velocity --n_state_bins 20 --n_action_bins 2 --max_steps 100 --policy uniform --no_optimistic_updates --no_optimistic_actions --name pv100_noopt_puniform`
- Slow adaptation: `python main_ablation_slow.py --eval_every 1 --env point --task velocity --n_state_bins 20 --n_action_bins 2 --max_steps 100 --policy uniform --name pv100_slow_puniform`
- Intrinsic rewards (not factored + slow adaptation + no optimism): `python main_ablation_intrinsic.py --eval_every 1 --env point --task velocity --n_state_bins 20 --n_action_bins 2 --max_steps 100 --explore_only --name pv100_intrinsic_noreward`
- No exploration: `python main.py --eval_every 1 --env point --task velocity --n_state_bins 20 --n_action_bins 2 --max_steps 100 --policy uniform --no_exploration --name pv100_puniform_noexplore`


### With rewards

Gridworld:
- Our method (factored + fast + optimistic): `python main.py --eval_every 1 --env gridworld --task default --n_state_bins 20 --env_size 20 --n_action_bins 4 --max_steps 100 --name grid20`
- No optimism: `python main.py --eval_every 1 --env gridworld --task default --n_state_bins 20 --env_size 20 --n_action_bins 4 --max_steps 100 --no_optimistic_updates --no_optimistic_actions --name grid20_noopt`
- Slow adaptation: `python main_ablation_slow.py --eval_every 1 --env gridworld --task default --n_state_bins 20 --env_size 20 --n_action_bins 4 --max_steps 100 --name grid20_slow`
- Intrinsic rewards (not factored + slow adaptation + no optimism): `python main_ablation_intrinsic.py --eval_every 1 --env gridworld --task default --n_state_bins 20 --env_size 20 --n_action_bins 4 --max_steps 100 --name grid20_intrinsic`
- No exploration: : `python main.py --eval_every 1 --env gridworld --task default --n_state_bins 20 --env_size 20 --n_action_bins 4 --max_steps 100 --no_exploration --name grid20_noexplore`

Point velocity:
- Our method (factored + fast + optimistic): `python main.py --eval_every 1 --env point --task velocity --n_state_bins 20 --n_action_bins 2 --max_steps 100 --name pv100`
- No optimism: `python main.py --eval_every 1 --env point --task velocity --n_state_bins 20 --n_action_bins 2 --max_steps 100 --no_optimistic_updates --no_optimistic_actions --name pv100_noopt`
- Slow adaptation: `python main_ablation_slow.py --eval_every 1 --env point --task velocity --n_state_bins 20 --n_action_bins 2 --max_steps 100 --name pv100_slow`
- Intrinsic rewards (not factored + slow adaptation + no optimism): `python main_ablation_intrinsic.py --eval_every 1 --env point --task velocity --n_state_bins 20 --n_action_bins 2 --max_steps 100 --name pv100_intrinsic`
- No exploration: `python main.py --eval_every 1 --env point --task velocity --n_state_bins 20 --n_action_bins 2 --max_steps 100 --no_exploration --name pv100_noexplore`