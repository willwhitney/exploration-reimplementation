import types
import dataclasses
from scipy import stats

import jax
from jax import numpy as jnp, random, nn

from flax import struct

import q_learning
import deep_q_functions as q_functions
import utils
import jax_specs
from experiment_logging import default_logger as logger


TEMP = 1
TEST_TEMP = 0.3
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
        with jax.profiler.TraceContext("sample explore"):
            actions, _, entropies = q_learning.sample_action_boltzmann_n_batch(
                policy_state.q_state, action_rngs, s, candidate_actions, TEMP)
            logger.update('train/policy_entropy', entropies.mean())
    else:
        with jax.profiler.TraceContext("sample test"):
            actions, _, entropies = q_learning.sample_action_boltzmann_n_batch(
                policy_state.q_state, action_rngs, s, candidate_actions,
                TEST_TEMP)
            logger.update('test/policy_entropy', entropies.mean())

            # actions, values = q_learning.sample_action_egreedy_n_batch(
            #     policy_state.q_state, action_rngs, s, candidate_actions, 0.01)
    policy_state = policy_state.replace(rng=policy_rng)
    return policy_state, actions


@jax.profiler.trace_function
@jax.jit
def update_fn(policy_state: PolicyState, transitions):
    # assert False
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
    # q_state, _ = q_functions.ddqn_train_step(
    #     policy_state.q_state, policy_state.targetq_state,
    #     transitions, candidate_actions)
    # q_state, _ = q_functions.soft_bellman_train_step(
    #     policy_state.q_state, policy_state.targetq_state,
    #     transitions, candidate_actions, TEST_TEMP)
    policy_state = policy_state.replace(q_state=q_state, rng=policy_rng)
    return policy_state
