import jax
from jax import numpy as jnp, random

from flax import struct

import q_learning


ACTIONS = jnp.arange(4)


@struct.dataclass
class PolicyState():
    q_state: q_learning.QLearnerState
    targetq_state: q_learning.QLearnerState
    rng: jnp.ndarray


def init_fn(seed, state_shape, action_shape):
    q_state = q_learning.init_fn(0,
                                (128, *state_shape),
                                (128, *action_shape))
    targetq_state = q_state
    rng = random.PRNGKey(seed)
    return PolicyState(q_state=q_state, targetq_state=targetq_state, rng=rng)


# should take in a (bsize x *state_shape) of states and return a
# (bsize x n x *action_shape) of candidate actions
@jax.partial(jax.jit, static_argnums=(2, 3))
def action_fn(policy_state, s, n=1, explore=True):
    bsize = s.shape[0]
    q_state, targetq_state, policy_rng = policy_state
    rngs = random.split(policy_rng, bsize * n + 1)
    policy_rng = rngs[0]
    action_rngs = rngs[1:].reshape((bsize, n, -1))
    temp = 1e-1 if explore else 1e-4

    candidate_actions = jnp.expand_dims(ACTIONS, 0)
    candidate_actions = candidate_actions.repeat(bsize, axis=0)
    # import ipdb; ipdb.set_trace()
    actions, values = q_learning.sample_action_boltzmann_n_batch(
        q_state, action_rngs, s, candidate_actions, temp)
    return (q_state, targetq_state, policy_rng), actions


@jax.jit
def update_fn(policy_state, transitions):
    bsize = len(transitions[0])
    q_state, targetq_state, rng = policy_state
    candidate_actions = jnp.expand_dims(ACTIONS, 0)
    candidate_actions = candidate_actions.repeat(bsize, axis=0)

    q_state = q_learning.bellman_train_step(
        q_state, targetq_state, transitions, candidate_actions)
    return (q_state, targetq_state, rng)
