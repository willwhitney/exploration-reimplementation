import jax
from jax import numpy as jnp, random

from flax import struct

import q_learning
import tabular_q_functions as q_functions


TEMP = 1


@struct.dataclass
class PolicyState():
    q_state: q_learning.QLearnerState
    targetq_state: q_learning.QLearnerState
    actions: jnp.ndarray
    rng: jnp.ndarray


def init_fn(seed, state_shape, action_shape, actions, **kwargs):
    q_state = q_functions.init_fn(seed,
                                  (128, *state_shape),
                                  (128, *action_shape),
                                  discount=0.99,
                                  **kwargs)
    targetq_state = q_state
    rng = random.PRNGKey(seed)
    return PolicyState(q_state=q_state, targetq_state=targetq_state,
                       actions=actions, rng=rng)


# should take in a (bsize x *state_shape) of states and return a
# (bsize x n x *action_shape) of candidate actions
@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(2, 3))
def action_fn(policy_state, s, n=1, explore=True):
    bsize = s.shape[0]
    rngs = random.split(policy_state.rng, bsize * n + 1)
    policy_rng = rngs[0]
    action_rngs = rngs[1:].reshape((bsize, n, -1))

    candidate_actions = jnp.expand_dims(policy_state.actions, 0)
    candidate_actions = candidate_actions.repeat(bsize, axis=0)
    if explore:
        actions, values, entropies = q_learning.sample_action_boltzmann_n_batch(
            policy_state.q_state, action_rngs, s, candidate_actions, TEMP)
    else:
        actions, values = q_learning.sample_action_egreedy_n_batch(
            policy_state.q_state, action_rngs, s, candidate_actions, 0.01)
    policy_state = policy_state.replace(rng=policy_rng)
    return policy_state, actions


@jax.jit
def update_fn(policy_state, transitions):
    bsize = len(transitions[0])
    candidate_actions = jnp.expand_dims(policy_state.actions, 0)
    candidate_actions = candidate_actions.repeat(bsize, axis=0)

    q_state, _ = q_functions.bellman_train_step(
        policy_state.q_state, policy_state.targetq_state,
        transitions, candidate_actions)
    policy_state = policy_state.replace(q_state=q_state)
    return policy_state
