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
import density_estimation


R_MAX = 100

# class ExplorationPolicy:
#     def __init__(self, novelty_q_fn, policy_action_fn,
#                  n_action_samples=64, temperature=1, seed=0):
#         """Construct an ExplorationPolicy.
#         Arguments:
#         - novelty_q_fn: a function that takes an internal state, observed
#             state, and a list of actions and returns a value estimate for each
#             action
#         - policy_action_fn: a function that takes an internal state, an rng, an
#             observed state, and a number of samples, and returns that many
#             sampled actions
#         """
#         # self.novelty_q_fn = novelty_q_fn
#         # self.policy_action_fn = policy_action_fn
#         self.n_action_samples = n_action_samples
#         self.temperature = temperature
#         self.rng = random.PRNGKey(seed)

#         def _sample_reweighted(novelty_q_state, rng, s, actions):
#             q_values = novelty_q_fn(novelty_q_state, s, actions)
#             logits = q_values / temperature
#             action_index = random.categorical(rng, logits)
#             return actions[action_index]

#         def _sample_action(novelty_q_state, policy_state, rng, s):
#             actions = policy_action_fn(policy_state, rng, s, n=self.n_action_samples)
#             rng = random.split(rng, 1)[0]
#             return _sample_reweighted(novelty_q_state, rng, s, actions)

#         self._sample_action = _sample_action
#         self._sample_actions = jax.vmap(_sample_action,
#                                         in_axes=(None, None, 0, 0))

#     def next_rng(self):
#         return random.split(self.rng, 1)[0]

#     def sample_action(self, novelty_q_state, policy_state, s):
#         new_rngs = random.split(self.rng, 2)
#         self.rng = new_rngs[0]
#         a = self._sample_action(novelty_q_state, policy_state, new_rngs[1], s)
#         return a

#     def sample_actions(self, novelty_q_state, policy_state, states):
#         new_rngs = random.split(self.rng, states.shape[0] + 1)
#         self.rng = new_rngs[0]
#         actions = self._sample_actions(
#             novelty_q_state, policy_state, new_rngs[1:], states)
#         return actions

@struct.dataclass
class AgentState():
    # q_opt: Any
    # targetq_opt: Any
    novq_state: q_learning.QLearnerState
    replay: Any = struct.field(pytree_node=False)
    density: Any = struct.field(pytree_node=False)
    policy_state: Any = struct.field(pytree_node=False)
    policy_action_fn: Any = struct.field(pytree_node=False)
    policy_update_fn: Any = struct.field(pytree_node=False)


def compute_novelty_reward(agent_state, states, actions):
    counts = agent_state.density.count(states, actions)
    return (counts + 1e-8) ** (-0.5)


def optimistic_train_step(agent_state, transitions):
    states, actions, next_states, rewards = transitions

    policy_state, candidate_next_actions = agent_state.policy_action_fn(
        agent_state.policy_state, next_states, n=64, explore=True)
    agent_state = agent_state.replace(policy_state=policy_state)

    # candidate actions should be (bsize x 64 x *action_shape)
    import ipdb; ipdb.set_trace()

    optimistic_next_values = predict_optimistic_value_batch(
        agent_state, next_states, candidate_next_actions)
    # optimistic_next_values should be (bsize x 64)
    expected_next_values = optimistic_next_values.mean(axis=1)
    # TODO: try out using maxQ or a lower temp instead of EQ

    novelty_reward = compute_novelty_reward(agent_state, states, actions)
    target_values = novelty_reward + 0.99 * expected_next_values

    novq_state = q_learning.train_step(
        agent_state.novq_state, states, actions, target_values)
    return agent_state.replace(novq_state=novq_state)


def update_novelty_q(agent_state, rng):
    # novq_state = agent_state.novq_state
    # TODO: change to a lax.scan once replay is in JAX
    for _ in range(10):
        transitions = agent_state.replay.sample(128)
        agent_state = optimistic_train_step(agent_state, transitions)
    return agent_state


def update_exploration(agent_state, rng, transition):
    s, a, sp, r = transition
    # add transitions to replay
    agent_state.replay.append(s, a, sp, r)

    # update density on new observations
    agent_state.density.update(jnp.expand_dims(s, axis=0),
                               jnp.expand_dims(a, axis=0))

    # update exploration Q to consistency with new density
    rng, novq_rng = random.split(rng)
    agent_state = update_novelty_q(agent_state, novq_rng)
    return agent_state


def compute_weight(count):
    root_count = count ** 0.5
    root_prior_count = 1.0 ** 0.5
    return root_count / (root_count + root_prior_count)


raise Exception("""This function should operate on only a single action.
                I can use JAX to promote it to taking a vector of actions.""")
def predict_optimistic_value(agent_state: AgentState,
                             env_state, candidate_actions):
    # candidate_actions should be (n_candidates x *action_shape)
    action_values = q_learning.predict_action_values(
        agent_state.novq_state, env_state, candidate_actions)

    n_candidates = candidate_actions.shape[0]
    expanded_state = jnp.expand_dims(env_state, axis=0)
    repeated_state = expanded_state.repeat(n_candidates, axis=0)

    counts = agent_state.density.count(repeated_state, candidate_actions)
    weights = compute_weight(count)

    optimistic_value = weights *

    raise NotImplementedError
predict_optimistic_value_batch = jax.vmap(  # noqa: E305
    predict_optimistic_value, in_axes=(None, 0, 0))


def select_action(agent_state, rng, env_state, candidate_actions, temp=0.1):
    optimistic_values = predict_optimistic_value(
        agent_state, env_state, candidate_actions)
    return q_learning.sample_boltzmann(
        rng, optimistic_values, candidate_actions, temp)


def full_step(agent_state: AgentState, rng, env, train=True):
    # get env state
    s = gridworld.render(env)
    n = 64 if train else 1

    # get candidate actions
    s_batch = jnp.expand_dims(s, axis=0)
    policy_state, candidate_actions = agent_state.policy_action_fn(
        agent_state.policy_state, s_batch, n=n, explore=train)
    agent_state = agent_state.replace(policy_state=policy_state)

    # sample a behavior action from the candidates
    rng, action_rng = random.split(rng)
    a = select_action(agent_state, action_rng, s, candidate_actions)

    # take action and observe outcome
    env, sp, r = gridworld.step(env, int(a))

    if train:
        # update the exploration policy with the observed transition
        rng, update_rng = random.split(rng)
        agent_state = update_exploration(agent_state, update_rng, (s, a, sp, r))

    return agent_state, env, r


def run_episode(agent_state: AgentState, rngs, env, train=True):
    env = gridworld.reset(env)
    score = 0
    for i in range(max_steps):
        agent_state, env, r = full_step(agent_state, rngs[i], env, train=train)
        score += r
    return agent_state, env, score


# def init_fn(policy_state, policy_sample_fn, policy_train_fn):
#     raise NotImplementedError


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
    def policy_action_fn(policy_state, s, n=1, explore=True):
        bsize = s.shape[0]
        q_state, targetq_state, policy_rng = policy_state
        rngs = random.split(policy_rng, bsize * n + 1)
        policy_rng = rngs[0]
        action_rngs = rngs[1:].reshape((bsize, n, -1))
        temp = 1e-1 if explore else 1e-4
        actions, values = q_learning.sample_action_boltzmann_n_batch(
            q_state, action_rngs, s, env.actions, temp)
        return [q_state, targetq_state, policy_rng], actions

    def policy_update_fn(policy_state, transitions):
        q_state, targetq_state, rng = policy_state
        q_state = q_learning.bellman_train_step(
            q_state, targetq_state, transitions, env.actions)
        return [q_state, targetq_state, rng]
    # --------------------------------------------------------

    density = density_estimation.TabularDensity()
    replay = replay_buffer.Replay(state_shape, action_shape)

    agent_state = AgentState(novq_state=novq_state,
                             replay=replay,
                             density=density,
                             policy_state=[q_state, targetq_state, policy_rng],
                             policy_action_fn=policy_action_fn,
                             policy_update_fn=policy_update_fn)

    for episode in range(1000):
        # run an episode
        rngs = random.split(rng, max_steps + 1)
        rng = rngs[0]
        agent_state, env, score = run_episode(agent_state, rngs[1:], env)

        # update the task policy
        # TODO: pull this loop inside the policy_update_fn
        policy_state = agent_state.policy_state
        for _ in range(100):
            transitions = replay.sample(batch_size)
            policy_state = policy_update_fn(policy_state,
                                            transitions)
        # hacky reset of targetq to q
        policy_state = [policy_state[0], policy_state[0], policy_state[2]]
        agent_state = agent_state.replace(policy_state=policy_state)

        # output / visualize
        if episode % 10 == 0:
            rngs = random.split(rng, max_steps + 1)
            rng = rngs[0]
            _, _, test_score = run_episode(
                agent_state, rngs[1:], env, train=False)
            print((f"Episode {episode:4d}"
                   f", Train score {score:3d}"
                   f", Test score {test_score:3d}"))
        if episode % 50 == 0:
            print("\nQ network values")
            q_learning.display_value_map(agent_state.novq_state, env)

        if episode % 1 == 0:
            targetq_state = q_state
