import math
import numpy as np
from dataclasses import dataclass
from typing import Any

import jax
from jax import numpy as jnp, random, jit, lax

import flax
from flax import nn, optim, struct

import gridworld
import replay_buffer
import q_learning
import tabular_density as density


R_MAX = 100


@struct.dataclass
class AgentState():
    # q_opt: Any
    # targetq_opt: Any
    novq_state: q_learning.QLearnerState
    # replay: Any = struct.field(pytree_node=False)
    density_state: density.DensityState
    policy_state: Any = struct.field(pytree_node=False)
    policy_action_fn: Any = struct.field(pytree_node=False)
    policy_update_fn: Any = struct.field(pytree_node=False)


# @jax.jit
# def stupid_fn(agent_state: AgentState, rng, state):
#     return random.split(rng)


# @jax.jit
# def stupid_transitions_fn(agent_state: AgentState, rng, transitions):
#     candidate_actions = jnp.zeros((len(transitions[0]), 64, 1))
#     return q_learning.bellman_train_step(agent_state.novq_state,
#                                          agent_state.novq_state,
#                                          transitions,
#                                          candidate_actions)


# @jax.jit
# def predict_optimistic_values(agent_state: AgentState, state, actions):
#     expanded_state = jnp.expand_dims(state, axis=0)
#     repeated_state = expanded_state.repeat(len(actions), axis=0)
#     # actions = actions.reshape((actions.shape
#     predicted_values = q_learning.predict_value(
#         agent_state.novq_state, repeated_state, actions).reshape(-1)

#     # n_candidates = candidate_actions.shape[0]
#     # expanded_state = jnp.expand_dims(env_state, axis=0)
#     # repeated_state = expanded_state.repeat(n_candidates, axis=0)

#     # count = agent_state.density.count(state, action)
#     counts = density.get_count_batch(
#         agent_state.density_state, repeated_state, actions)
#     weights = jnp.ones(counts.shape)
#     optimistic_value = weights * predicted_values + (1 - weights) * R_MAX
#     return optimistic_value


# @jax.partial(jax.jit, static_argnums=(0,))
@jax.jit
def stupid_fn(agent_state, state, actions):
    return actions + state.sum()



if __name__ == '__main__':
    rng = random.PRNGKey(0)
    env = gridworld.new(10)
    state_shape = (2, env.size)
    action_shape = (1,)
    batch_size = 128
    max_steps = 100

    # ---------- creating the task policy --------------------
    q_state = q_learning.init_fn(0,
                                 (128, *state_shape),
                                 (128, *action_shape))
    targetq_state = q_state
    novq_state = q_learning.init_fn(1,
                                    (128, *state_shape),
                                    (128, *action_shape))
    rng, policy_rng = random.split(rng, 2)

    # should take in a (bsize x *state_shape) of states and return a
    # (bsize x n x *action_shape) of candidate actions
    @jax.partial(jax.jit, static_argnums=(2, 3))
    def policy_action_fn(policy_state, s, n=1, explore=True):
        bsize = s.shape[0]
        q_state, targetq_state, policy_rng = policy_state
        rngs = random.split(policy_rng, bsize * n + 1)
        policy_rng = rngs[0]
        action_rngs = rngs[1:].reshape((bsize, n, -1))
        temp = 1e-1 if explore else 1e-4

        candidate_actions = jnp.expand_dims(env.actions, 0)
        candidate_actions = candidate_actions.repeat(bsize, axis=0)
        # import ipdb; ipdb.set_trace()
        actions, values = q_learning.sample_action_boltzmann_n_batch(
            q_state, action_rngs, s, candidate_actions, temp)
        # return (q_state, targetq_state, policy_rng), actions
        return policy_state, actions

    @jax.jit
    def policy_update_fn(policy_state, transitions):
        bsize = len(transitions[0])
        q_state, targetq_state, rng = policy_state
        candidate_actions = jnp.expand_dims(env.actions, 0)
        candidate_actions = candidate_actions.repeat(bsize, axis=0)

        q_state = q_learning.bellman_train_step(
            q_state, targetq_state, transitions, candidate_actions)
        return (q_state, targetq_state, rng)
    # --------------------------------------------------------

    density_state = density.new([env.size, env.size], [len(env.actions)])
    replay = replay_buffer.Replay(state_shape, action_shape)

    agent_state = AgentState(novq_state=novq_state,
                             # replay=replay,
                             density_state=density_state,
                             policy_state=(q_state, targetq_state, policy_rng),
                             policy_action_fn=policy_action_fn,
                             policy_update_fn=policy_update_fn)


    print(stupid_fn(agent_state, rng, gridworld.render(env)))
    print(stupid_fn(agent_state, random.split(rng)[0], gridworld.render(env)))

    q_state = q_learning.init_fn(1,
                                 (128, *state_shape),
                                 (128, *action_shape))
    targetq_state = q_state
    rng, policy_rng = random.split(rng, 2)

    policy_state = (q_state, targetq_state, policy_rng)
    agent_state = agent_state.replace(policy_state=policy_state)
    state = gridworld.render(env)
    print(stupid_fn(agent_state, random.split(rng)[0], state))

    for _ in range(batch_size):
        replay.append(state, 0, state, 1)
    transitions = tuple((jnp.array(el) for el in replay.sample(batch_size)))
    # print(stupid_transitions_fn(agent_state, rng, transitions))
    # transitions = tuple((jnp.array(el) for el in replay.sample(batch_size)))
    # print(stupid_transitions_fn(agent_state, rng, transitions))

    # candidate_actions = np.zeros((64,))
    # print(predict_optimistic_values(agent_state, state, candidate_actions))

    # env, state, reward = gridworld.step(env, 2)
    # candidate_actions = np.ones((64,))
    # print(predict_optimistic_values(agent_state, state, candidate_actions))

    candidate_actions = np.zeros((64,))
    print(stupid_fn(agent_state, state, candidate_actions))
    env, state, reward = gridworld.step(env, 2)
    candidate_actions = np.ones((64,))
    print(stupid_fn(agent_state, state, candidate_actions))

    policy_state = agent_state.policy_update_fn(policy_state, transitions)
    agent_state = agent_state.replace(policy_state=policy_state)

    candidate_actions = np.zeros((64,))
    print(stupid_fn(agent_state, state, candidate_actions))
    env, state, reward = gridworld.step(env, 2)
    candidate_actions = np.ones((64,))
    print(stupid_fn(agent_state, state, candidate_actions))

    _, actions = agent_state.policy_action_fn(
        agent_state.policy_state, transitions[0], 64, True)
    agent_state = agent_state.replace(policy_state=agent_state.policy_state)

    candidate_actions = np.zeros((64,))
    print(stupid_fn(agent_state, state, candidate_actions))
    env, state, reward = gridworld.step(env, 2)
    candidate_actions = np.ones((64,))
    print(stupid_fn(agent_state, state, candidate_actions))

    policy_state, actions = agent_state.policy_action_fn(
        agent_state.policy_state, transitions[0], 64, True)
    agent_state = agent_state.replace(policy_state=policy_state)

    candidate_actions = np.zeros((64,))
    print(stupid_fn(agent_state, state, candidate_actions))
    env, state, reward = gridworld.step(env, 2)
    candidate_actions = np.ones((64,))
    print(stupid_fn(agent_state, state, candidate_actions))
