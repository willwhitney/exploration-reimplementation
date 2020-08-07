import time
# import math
# import numpy as np
from typing import Any

import jax
from jax import numpy as jnp, random, lax

import flax
from flax import nn, optim, struct

import gridworld
import replay_buffer
import q_learning
import tabular_density as density


R_MAX = 100


@struct.dataclass
class ExplorationState():
    """The pure-JAX components that can be jitted/vmapped.
    """
    novq_state: q_learning.QLearnerState
    density_state: density.DensityState


@struct.dataclass
class AgentState():
    """A container for the entire state; not jittable.
    """
    exploration_state: ExplorationState
    policy_state: Any = struct.field(pytree_node=False)
    policy_action_fn: Any = struct.field(pytree_node=False)
    policy_update_fn: Any = struct.field(pytree_node=False)


@jax.jit
def compute_novelty_reward(exploration_state, states, actions):
    counts = density.get_count_batch(
        exploration_state.density_state, states, actions)
    return (counts + 1e-8) ** (-0.5)


@jax.jit
def optimistic_train_step_candidates(exploration_state,
                                     transitions,
                                     candidate_next_actions):
    states, actions, next_states, rewards = transitions
    optimistic_next_values = predict_optimistic_values_batch(
        exploration_state, next_states, candidate_next_actions)

    # optimistic_next_values should be (bsize x 64)
    expected_next_values = optimistic_next_values.mean(axis=1)
    # TODO: try out using maxQ or a lower temp instead of EQ
    # the current version looks more like policy iteration, where the next
    # policy step happens on the next episode

    novelty_reward = compute_novelty_reward(
        exploration_state, states, actions)
    target_values = novelty_reward + 0.99 * expected_next_values

    novq_state = q_functions.train_step(
        exploration_state.novq_state,
        states, actions, target_values)
    return exploration_state.replace(novq_state=novq_state)


def optimistic_train_step(agent_state, transitions):
    states, actions, next_states, rewards = transitions

    policy_state, candidate_next_actions = agent_state.policy_action_fn(
        agent_state.policy_state, next_states, 64, True)
    agent_state = agent_state.replace(policy_state=policy_state)

    # candidate actions should be (bsize x 64 x *action_shape)
    exploration_state = optimistic_train_step_candidates(
        agent_state.exploration_state, transitions, candidate_next_actions)
    return agent_state.replace(exploration_state=exploration_state)


def update_novelty_q(agent_state, replay, rng):
    for _ in range(10):
        transitions = tuple((jnp.array(el) for el in replay.sample(128)))
        agent_state = optimistic_train_step(agent_state, transitions)
    return agent_state


def update_exploration(agent_state, replay, rng, transition):
    s, a, sp, r = transition

    # update density on new observations
    exploration_state = agent_state.exploration_state
    density_state = density.update_batch(exploration_state.density_state,
                                         jnp.expand_dims(s, axis=0),
                                         jnp.expand_dims(a, axis=0))
    exploration_state = exploration_state.replace(density_state=density_state)
    agent_state = agent_state.replace(exploration_state=exploration_state)

    # update exploration Q to consistency with new density
    rng, novq_rng = random.split(rng)
    agent_state = update_novelty_q(agent_state, replay, novq_rng)
    return agent_state


def compute_weight(count):
    root_count = count ** 0.5
    root_prior_count = 1.0 ** 0.5
    return root_count / (root_count + root_prior_count)


@jax.jit
def predict_optimistic_values(exploration_state: ExplorationState,
                              state, actions):
    expanded_state = jnp.expand_dims(state, axis=0)
    repeated_state = expanded_state.repeat(len(actions), axis=0)
    predicted_values = q_learning.predict_value(
        exploration_state.novq_state, repeated_state, actions).reshape(-1)

    counts = density.get_count_batch(
        exploration_state.density_state, repeated_state, actions)
    weights = compute_weight(counts)
    optimistic_value = weights * predicted_values + (1 - weights) * R_MAX
    return optimistic_value
# predict_optimistic_value_n = jax.vmap(  # noqa: E305
#     predict_optimistic_value, in_axes=(None, None, 0))
predict_optimistic_values_batch = jax.vmap(  # noqa: E305
    predict_optimistic_values, in_axes=(None, 0, 0))


@jax.jit
def select_action(exploration_state, rng, state, candidate_actions, temp=0.1):
    optimistic_values = predict_optimistic_values(
        exploration_state, state, candidate_actions).reshape(-1)

    return q_learning.sample_boltzmann(
        rng, optimistic_values, candidate_actions, temp)


def full_step(agent_state: AgentState, replay, rng, env, train=True):
    # get env state
    s = gridworld.render(env)
    n = 64 if train else 1

    # get candidate actions
    s_batch = jnp.expand_dims(s, axis=0)
    policy_state, candidate_actions = agent_state.policy_action_fn(
        agent_state.policy_state, s_batch, n, train)

    # policy_action_fn deals with batches and we only have one element
    candidate_actions = candidate_actions[0]
    agent_state = agent_state.replace(policy_state=policy_state)

    # sample a behavior action from the candidates
    rng, action_rng = random.split(rng)
    a = select_action(agent_state.exploration_state,
                      action_rng, s, candidate_actions)

    # take action and observe outcome
    env, sp, r = gridworld.step(env, int(a))

    # add transition to replay
    replay.append(s, a, sp, r)

    if train:
        # update the exploration policy with the observed transition
        rng, update_rng = random.split(rng)
        agent_state = update_exploration(
            agent_state, replay, update_rng, (s, a, sp, r))

    # density.display_density_map(
    #     agent_state.exploration_state.density_state, env)
    # time.sleep(0.5)

    return agent_state, env, r


def run_episode(agent_state: AgentState, replay, rngs, env,
                train=True, max_steps=100):
    env = gridworld.reset(env)
    score = 0
    for i in range(max_steps):
        agent_state, env, r = full_step(
            agent_state, replay, rngs[i], env, train=train)
        score += r
    return agent_state, env, score


def main(args):
    rng = random.PRNGKey(args.seed)
    env = gridworld.new(args.env_size)
    state_shape = (2, env.size)
    action_shape = (1,)
    batch_size = 128
    max_steps = 100

    novq_state = q_functions.init_fn(args.seed,
                                     (128, *state_shape),
                                     (128, *action_shape),
                                     env_size=env.size)

    density_state = density.new([env.size, env.size], [len(env.actions)])
    replay = replay_buffer.Replay(state_shape, action_shape)

    policy_state = policy.init_fn(args.seed,
                                  state_shape, action_shape,
                                  env.actions, env_size=env.size)

    exploration_state = ExplorationState(novq_state=novq_state,
                                         density_state=density_state)
    agent_state = AgentState(exploration_state=exploration_state,
                             policy_state=policy_state,
                             policy_action_fn=policy.action_fn,
                             policy_update_fn=policy.update_fn)

    for episode in range(100000):
        # run an episode
        rngs = random.split(rng, max_steps + 1)
        rng = rngs[0]

        agent_state, env, score = run_episode(
            agent_state, replay, rngs[1:], env,
            train=True, max_steps=max_steps)

        # update the task policy
        # TODO: pull this loop inside the policy_update_fn
        policy_state = agent_state.policy_state
        for _ in range(1):
            transitions = tuple((jnp.array(el)
                                 for el in replay.sample(batch_size)))
            policy_state = agent_state.policy_update_fn(policy_state,
                                                        transitions)
        # hacky reset of targetq to q
        policy_state = policy_state.replace(targetq_state=policy_state.q_state)
        agent_state = agent_state.replace(policy_state=policy_state)

        # output / visualize
        if episode % 10 == 0:
            rngs = random.split(rng, max_steps + 1)
            rng = rngs[0]
            _, _, test_score = run_episode(agent_state, replay, rngs[1:], env,
                                           train=False, max_steps=max_steps)
            print((f"Episode {episode:4d}"
                   f", Train score {score:3d}"
                   f", Test score {test_score:3d}"))
        if episode % 1 == 0:
            q_learning.display_value_map(
                agent_state.exploration_state.novq_state, env)
            density.display_density_map(
                agent_state.exploration_state.density_state, env)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0)
    parser.add_argument('--tabular', action='store_true', default=False)
    parser.add_argument('--env_size', type=int, default=5)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    if args.tabular:
        import tabular_q_functions as q_functions
        import policies.tabular_q_policy as policy
    else:
        import deep_q_functions as q_functions
        import policies.deep_q_policy as policy

    jit = not args.debug
    if jit:
        main(args)
    else:
        with jax.disable_jit():
            main(args)
