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


# class ExplorationPolicy:
#     def __init__(self, novelty_q_fn, policy_fn,
#                  n_action_samples=64, temperature=1, seed=0):
#         """Construct an ExplorationPolicy.
#         Arguments:
#         - novelty_q_fn: a function that takes an internal state, observed
#             state, and a list of actions and returns a value estimate for each
#             action
#         - policy_fn: a function that takes an internal state, an rng, an
#             observed state, and a number of samples, and returns that many
#             sampled actions
#         """
#         # self.novelty_q_fn = novelty_q_fn
#         # self.policy_fn = policy_fn
#         self.n_action_samples = n_action_samples
#         self.temperature = temperature
#         self.rng = random.PRNGKey(seed)

#         def _sample_reweighted(novelty_q_state, rng, s, actions):
#             q_values = novelty_q_fn(novelty_q_state, s, actions)
#             logits = q_values / temperature
#             action_index = random.categorical(rng, logits)
#             return actions[action_index]

#         def _sample_action(novelty_q_state, policy_state, rng, s):
#             actions = policy_fn(policy_state, rng, s, n=self.n_action_samples)
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
    novq_opt: Any
    replay: Any
    density: Any
    policy_state: Any = struct.field(pytree_node=False)


def update_exploration(agent_state, transitions):
    # add transitions to replay
    pass

    # update density on new observations
    pass

    # update exploration Q to consistency with new density
    pass


def select_action(agent_state, env_state, candidate_actions):
    # can probably reuse the function from q_learning.py
    raise NotImplementedError


def full_step(agent_state, env):
    s = gridworld.render(env)
    candidate_actions =
    a = choose_action(rng, q_opt, s, env.actions)
    # import ipdb; ipdb.set_trace()
    with profiler.TraceContext("env step"):
        env, sp, r = gridworld.step(env, int(a))
    with profiler.TraceContext("replay append"):
        buffer.append(s, a, sp, r)

    if len(buffer) > batch_size:
        with profiler.TraceContext("replay sample"):
            transitions = buffer.sample(batch_size)
        with profiler.TraceContext("bellman step"):
            q_opt = bellman_train_step(q_opt, targetq_opt, transitions)
    return q_opt, env, r


# @jax.profiler.trace_function
def run_episode(rngs, q_opt, env):
    env = gridworld.reset(env)
    score = 0
    for i in range(max_steps):
        q_opt, env, r = full_step(rngs[i], q_opt, env)
        score += r
    return q_opt, env, score


for episode in range(200):
    # with jax.disable_jit():
    time.sleep(0.1)
    rngs = random.split(rng, max_steps + 1)
    rng = rngs[0]
    q_opt, env, score = run_episode(rngs[1:], q_opt, env)
    print(f"Episode {episode}, Score {score}")
    if episode % 1 == 0:
        targetq_opt = q_opt


if __name__ == '__main__':
    rng = random.PRNGKey(0)
    env = gridworld.new(10)
    state_shape = (2, env.size)
    action_shape = (1,)
    batch_size = 128
    max_steps = 100

    q_opt = q_learning.init_fn(0, (128, *state_shape), (128, *action_shape))
    targetq_opt = q_opt
    novq_opt = q_learning.init_fn(0, (128, *state_shape), (128, *action_shape))

    density_est = density.TabularDensity()
    buffer = replay.Replay(state_shape, action_shape)


