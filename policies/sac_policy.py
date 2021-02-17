import numpy as np
import torch

import utils
from experiment_logging import default_logger as logger
from policies.pytorch_sac.agent import sac


def init_fn(state_spec, action_spec, seed,
            lr=1e-4, **kwargs):
    flat_state_spec = utils.flatten_observation_spec(state_spec)
    state_shape = flat_state_spec.shape
    torch.manual_seed(seed)

    sac_args = {}
    sac_args['obs_dim'] = state_shape[0]
    sac_args['action_dim'] = action_spec.shape[0]
    sac_args['action_range'] = [
        float(action_spec.minimum.min()),
        float(action_spec.maximum.max()),
    ]
    sac_args['device'] = 'cuda'

    # parameters taken from sac.yaml
    sac_args['critic_args'] = {
        "obs_dim": sac_args['obs_dim'],
        "action_dim": sac_args['action_dim'],
        "hidden_dim": 1024,
        "hidden_depth": 2,
    }

    sac_args['actor_args'] = {
        "obs_dim": sac_args['obs_dim'],
        "action_dim": sac_args['action_dim'],
        "hidden_dim": 1024,
        "hidden_depth": 2,
        "log_std_bounds": [-5, 2],
    }

    # frequencies // 2  b/c we do fewer policy updates
    sac_args['discount'] = 0.99
    sac_args['init_temperature'] = 0.1
    sac_args['alpha_lr'] = 1e-4
    sac_args['alpha_betas'] = [0.9, 0.999]
    sac_args['actor_lr'] = 1e-4
    sac_args['actor_betas'] = [0.9, 0.999]
    sac_args['actor_update_frequency'] = 1
    sac_args['critic_lr'] = 1e-4
    sac_args['critic_betas'] = [0.9, 0.999]
    sac_args['critic_tau'] = 0.005
    sac_args['critic_target_update_frequency'] = 2  # // 2
    sac_args['batch_size'] = 1024
    sac_args['learnable_temperature'] = True
    sac_args['num_seed_steps'] = 500  # // 2
    return sac.SACAgent(**sac_args)


def action_fn(sac_agent: sac.SACAgent, state, n=1, explore=True):
    state = np.array(state)
    actions, entropy = sac_agent.act_samples(state, n, sample=explore)
    if explore:
        logger.update('train/policy_entropy', entropy)
    else:
        logger.update('test/policy_entropy', entropy)
    return sac_agent, actions


def update_fn(sac_agent: sac.SACAgent, transitions):
    transitions = (np.array(jax_array) for jax_array in transitions)
    sac_agent.update(transitions)
    return sac_agent
