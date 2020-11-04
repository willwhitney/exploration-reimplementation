import types
import dataclasses

import jax
from jax import numpy as jnp, random

from flax import struct

import q_learning
import deep_q_functions as q_functions
import utils
import jax_specs


TEMP = 100
N_CANDIDATES = 32


@struct.dataclass
class PolicyState():
    q_state: q_learning.QLearnerState
    targetq_state: q_learning.QLearnerState
    rng: jnp.ndarray
    action_spec: jax_specs.BoundedArray
    n_candidates: int


def init_fn(state_spec, action_spec, seed, n_candidates=32, **kwargs):
    j_action_spec = jax_specs.convert_dm_spec(action_spec)
    q_state = q_functions.init_fn(seed, state_spec, action_spec,
                                  discount=0.99, **kwargs)
    targetq_state = q_state
    rng = random.PRNGKey(seed)
    return PolicyState(q_state=q_state, targetq_state=targetq_state, rng=rng,
                       action_spec=j_action_spec, n_candidates=n_candidates)

@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(2, 3))
def action_fn(policy_state: PolicyState, s, n=1, explore=True):
    bsize = s.shape[0]
    policy_rng, candidate_rng = random.split(policy_state.rng)

    with jax.profiler.TraceContext("sample uniform actions"):
        candidate_actions = utils.sample_uniform_actions(
            policy_state.action_spec, candidate_rng,
            N_CANDIDATES * bsize)

    rngs = random.split(policy_rng, bsize * n + 1)
    policy_rng = rngs[0]
    action_rngs = rngs[1:].reshape((bsize, n, -1))
    candidate_shape = (bsize, N_CANDIDATES,
                        *candidate_actions.shape[1:])
    candidate_actions = candidate_actions.reshape(candidate_shape)
    if explore:
        with jax.profiler.TraceContext("sample boltzmann"):
            actions, values = q_learning.sample_action_boltzmann_n_batch(
                policy_state.q_state, action_rngs, s, candidate_actions, TEMP)
    else:
        with jax.profiler.TraceContext("sample egreedy"):
            actions, values = q_learning.sample_action_egreedy_n_batch(
                policy_state.q_state, action_rngs, s, candidate_actions, 0.01)
    policy_state = policy_state.replace(rng=policy_rng)
    return policy_state, actions

@jax.profiler.trace_function
@jax.jit
def update_fn(policy_state: PolicyState, transitions):
    """A dummy function to allow settings to be static."""
    bsize = len(transitions[0])
    policy_rng, candidate_rng = random.split(policy_state.rng)
    candidate_actions = utils.sample_uniform_actions(
        policy_state.action_spec, candidate_rng,
        N_CANDIDATES * bsize)

    candidate_shape = (bsize, N_CANDIDATES,
                       *candidate_actions.shape[1:])
    candidate_actions = candidate_actions.reshape(candidate_shape)

    q_state, _ = q_functions.bellman_train_step(
        policy_state.q_state, policy_state.targetq_state,
        transitions, candidate_actions)
    policy_state = policy_state.replace(q_state=q_state, rng=policy_rng)
    return policy_state
