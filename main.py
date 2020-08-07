import time
# import math
import numpy as np
import matplotlib.pyplot as plt
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
    return (counts + 1) ** (-0.5)


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
def predict_optimistic_value(exploration_state: ExplorationState,
                             state, action):
    expanded_state = jnp.expand_dims(state, axis=0)
    expanded_action = jnp.expand_dims(action, axis=0)
    predicted_value = q_learning.predict_value(exploration_state.novq_state,
                                               expanded_state, expanded_action)
    predicted_value = predicted_value.reshape(tuple())
    count = density.get_count(exploration_state.density_state,
                              state, action)
    weight = compute_weight(count)
    optimistic_value = weight * predicted_value + (1 - weight) * R_MAX
    return optimistic_value
predict_optimistic_value_batch = jax.vmap(  # noqa: E305
    predict_optimistic_value, in_axes=(None, 0, 0))
predict_optimistic_values = jax.vmap(
    predict_optimistic_value, in_axes=(None, None, 0))
predict_optimistic_values_batch = jax.vmap(  # noqa: E305
    predict_optimistic_values, in_axes=(None, 0, 0))


@jax.jit
def select_action(exploration_state, rng, state, candidate_actions, temp=1e-5):
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

    if train:
        # add transition to replay
        replay.append(s, a, sp, r)

        # update the exploration policy with the observed transition
        rng, update_rng = random.split(rng)
        agent_state = update_exploration(
            agent_state, replay, update_rng, (s, a, sp, r))

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


# ----- Visualizations for gridworld ---------------------------------
def display_state(agent_state: AgentState, replay, env, max_steps=100):
    exploration_state = agent_state.exploration_state
    policy_state = agent_state.policy_state

    min_count_map = gridworld.render_function(
        jax.partial(density.get_count_batch, exploration_state.density_state),
        env, reduction=jnp.min)
    sum_count_map = gridworld.render_function(
        jax.partial(density.get_count_batch, exploration_state.density_state),
        env, reduction=jnp.sum)
    novq_map = gridworld.render_function(
        jax.partial(q_learning.predict_value, exploration_state.novq_state),
        env, reduction=jnp.max)
    optimistic_novq_map = gridworld.render_function(
        jax.partial(predict_optimistic_value_batch, exploration_state),
        env, reduction=jnp.max)
    taskq_map = gridworld.render_function(
        jax.partial(q_learning.predict_value, policy_state.q_state),
        env, reduction=jnp.max)
    novelty_reward_map = gridworld.render_function(
        jax.partial(compute_novelty_reward, exploration_state),
        env, reduction=jnp.max)
    traj_map = replay_buffer.render_trajectory(replay, max_steps, env)

    subfigs = [
        (min_count_map, "Visit count (min)"),
        (sum_count_map, "Visit count (sum)"),
        (novq_map, "Novelty value (max)"),
        (optimistic_novq_map, "Optimistic novelty value (max)"),
        (taskq_map, "Task value (max)"),
        (novelty_reward_map, "Novelty reward (max)"),
        (traj_map, "Last trajectory"),
    ]

    fig, axs = plt.subplots(1, len(subfigs))
    for ax, subfig in zip(axs, subfigs):
        render, title = subfig
        img = ax.imshow(render)
        fig.colorbar(img, ax=ax)
        ax.set_title(title)
    fig.set_size_inches(4 * len(subfigs), 3)
    fig.show()
    plt.close(fig)
    time.sleep(3)
# -------------------------------------------------------------------


def main(args):
    rng = random.PRNGKey(args.seed)
    env = gridworld.new(args.env_size)
    state_shape = (2, env.size)
    action_shape = (1,)
    batch_size = 128
    max_steps = 5

    novq_state = q_functions.init_fn(args.seed,
                                     (128, *state_shape),
                                     (128, *action_shape),
                                     env_size=env.size,
                                     discount=0.97)

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
        if episode % 1 == 0:
            rngs = random.split(rng, max_steps + 1)
            rng = rngs[0]
            _, _, test_score = run_episode(agent_state, replay, rngs[1:], env,
                                           train=False, max_steps=max_steps)
            print((f"Episode {episode:4d}"
                   f", Train score {score:3d}"
                   f", Test score {test_score:3d}"))
            if args.vis:
                display_state(agent_state, replay, env, max_steps=max_steps)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0)
    parser.add_argument('--tabular', action='store_true', default=False)
    parser.add_argument('--env_size', type=int, default=5)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--no_vis', action='store_true', default=False)
    args = parser.parse_args()
    args.vis = not args.no_vis

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
