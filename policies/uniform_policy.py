import jax
from jax import numpy as jnp, random

from flax import struct

import jax_specs
import utils
from experiment_logging import default_logger as logger


@struct.dataclass
class PolicyState():
    rng: jnp.ndarray
    action_spec: jax_specs.BoundedArray


def init_fn(state_spec, action_spec, seed, *args, **kwargs):
    j_action_spec = jax_specs.convert_dm_spec(action_spec)
    return PolicyState(rng=random.PRNGKey(seed), action_spec=j_action_spec)

def action_fn(policy_state: PolicyState, s, n=1, explore=True):
    bsize = s.shape[0]
    policy_rng, action_rng = random.split(policy_state.rng)
    actions = utils.sample_uniform_actions(
        policy_state.action_spec, action_rng, bsize * n)
    action_shape = actions.shape[1:]
    actions = actions.reshape((bsize, n, *action_shape))

    if explore:
        logger.update('train/policy_entropy', 0)
    else:
        logger.update('test/policy_entropy', 0)

    policy_state = policy_state.replace(rng=policy_rng)
    return policy_state, actions

def update_fn(policy_state, transitions):
    return policy_state
