import time
import os
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
import utils


R_MAX = 100


@struct.dataclass
class ExplorationState():
    """The pure-JAX components that can be jitted/vmapped.
    """
    novq_state: q_learning.QLearnerState
    target_novq_state: q_learning.QLearnerState
    density_state: density.DensityState
    temperature: float
    prior_count: float


@struct.dataclass
class AgentState():
    """A container for the entire state; not jittable.
    """
    exploration_state: ExplorationState
    policy_state: Any = struct.field(pytree_node=False)
    policy_action_fn: Any = struct.field(pytree_node=False)
    policy_update_fn: Any = struct.field(pytree_node=False)
    n_candidates: int
    n_update_candidates: int


@jax.jit
def compute_novelty_reward(exploration_state, states, actions):
    """Returns a novelty reward in [0, 1] for each (s, a) pair."""
    counts = density.get_count_batch(
        exploration_state.density_state, states, actions)
    ones = jnp.ones(jnp.array(counts).shape)
    rewards = (counts + 1e-8) ** (-0.5)
    options = jnp.stack([ones, rewards], axis=1)

    # Clip rewards to be at most 1 (when count is 0)
    return jnp.min(options, axis=1)


@jax.profiler.trace_function
@jax.jit
def optimistic_train_step_candidates(exploration_state,
                                     transitions,
                                     candidate_next_actions):
    states, actions, next_states, rewards = transitions
    optimistic_next_values = predict_optimistic_values_batch(
        exploration_state.target_novq_state,
        exploration_state.density_state,
        exploration_state.prior_count,
        next_states, candidate_next_actions)

    # optimistic_next_values should be (bsize x 64)
    temp = exploration_state.temperature
    next_value_probs = nn.softmax(optimistic_next_values / temp, axis=1)
    next_value_elements = (next_value_probs * optimistic_next_values)
    expected_next_values = next_value_elements.sum(axis=1)
    # TODO: try out using maxQ or a lower temp instead of EQ
    #   (and many candidates in the update)
    # the current version looks more like policy iteration, where the next
    #   policy step happens on the next episode

    discount = exploration_state.novq_state.discount
    novelty_reward = compute_novelty_reward(
        exploration_state, states, actions)
    target_values = novelty_reward + discount * expected_next_values

    novq_state = q_functions.train_step(
        exploration_state.novq_state,
        states, actions, target_values)
    return exploration_state.replace(novq_state=novq_state)


@jax.profiler.trace_function
def optimistic_train_step(agent_state, transitions):
    states, actions, next_states, rewards = transitions

    policy_state, candidate_next_actions = agent_state.policy_action_fn(
        agent_state.policy_state, next_states,
        agent_state.n_update_candidates, True)
    agent_state = agent_state.replace(policy_state=policy_state)

    # candidate actions should be (bsize x 64 x *action_shape)
    exploration_state = optimistic_train_step_candidates(
        agent_state.exploration_state, transitions, candidate_next_actions)
    return agent_state.replace(exploration_state=exploration_state)


def update_target_q(agent_state: AgentState):
    exploration_state = agent_state.exploration_state.replace(
        target_novq_state=agent_state.exploration_state.novq_state)
    return agent_state.replace(exploration_state=exploration_state)


def update_novelty_q(agent_state, replay, rng):
    # if len(replay) > 100:
    #     display_state(agent_state, replay, gridworld.new(2))
    for _ in range(1):
        transitions = tuple((jnp.array(el) for el in replay.sample(128)))
        agent_state = optimistic_train_step(agent_state, transitions)
    # agent_state = update_target_q(agent_state)
    return agent_state


@jax.profiler.trace_function
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


def compute_weight(prior_count, count):
    root_count = count ** 0.5
    root_prior_count = prior_count ** 0.5
    return root_count / (root_count + root_prior_count)


@jax.profiler.trace_function
@jax.jit
def predict_optimistic_value(novq_state, density_state, prior_count,
                            #  exploration_state: ExplorationState,
                             state, action):
    expanded_state = jnp.expand_dims(state, axis=0)
    expanded_action = jnp.expand_dims(action, axis=0)
    predicted_value = q_learning.predict_value(novq_state,
                                               expanded_state, expanded_action)
    predicted_value = predicted_value.reshape(tuple())
    count = density.get_count(density_state,
                              state, action)
    weight = compute_weight(prior_count, count)
    optimistic_value = weight * predicted_value + (1 - weight) * R_MAX
    return optimistic_value
predict_optimistic_value_batch = jax.vmap(  # noqa: E305
    predict_optimistic_value, in_axes=(None, None, None, 0, 0))
predict_optimistic_values = jax.vmap(
    predict_optimistic_value, in_axes=(None, None, None, None, 0))
predict_optimistic_values_batch = jax.vmap(  # noqa: E305
    predict_optimistic_values, in_axes=(None, None, None, 0, 0))


@jax.profiler.trace_function
@jax.jit
def select_action(exploration_state, rng, state, candidate_actions):
    optimistic_values = predict_optimistic_values(
        exploration_state.novq_state,
        exploration_state.density_state,
        exploration_state.prior_count,
        state, candidate_actions).reshape(-1)

    # return q_learning.sample_egreedy(
    #     rng, optimistic_values, candidate_actions, 0.0)

    return q_learning.sample_boltzmann(
        rng, optimistic_values, candidate_actions,
        exploration_state.temperature)


@jax.profiler.trace_function
def full_step(agent_state: AgentState, replay, rng, env,
              train=True):
    # get env state
    with jax.profiler.TraceContext("env render"):
        s = gridworld.render(env)

    n = agent_state.n_candidates if train else 1

    # get candidate actions
    with jax.profiler.TraceContext("get action candidates"):
        s_batch = jnp.expand_dims(s, axis=0)
        policy_state, candidate_actions = agent_state.policy_action_fn(
            agent_state.policy_state, s_batch, n, train)

    # policy_action_fn deals with batches and we only have one element
    candidate_actions = candidate_actions[0]
    agent_state = agent_state.replace(policy_state=policy_state)

    # sample a behavior action from the candidates
    with jax.profiler.TraceContext("select action"):
        rng, action_rng = random.split(rng)
        a = select_action(agent_state.exploration_state,
                          action_rng, s, candidate_actions)

    # take action and observe outcome
    with jax.profiler.TraceContext("env step"):
        env, sp, r = gridworld.step(env, int(a))

    if train:
        # add transition to replay
        with jax.profiler.TraceContext("replay append"):
            replay.append(s, a, sp, r)

        # update the exploration policy with the observed transition
        rng, update_rng = random.split(rng)
        agent_state = update_exploration(
            agent_state, replay, update_rng, (s, a, sp, r))

    return agent_state, env, r


def run_episode(agent_state: AgentState, replay, rngs, env,
                train=True, max_steps=100, use_exploration=True):
    env = gridworld.reset(env)
    score = 0
    for i in range(max_steps):
        agent_state, env, r = full_step(
            agent_state, replay, rngs[i], env, train=train)
        score += r
    return agent_state, env, score


# ----- Visualizations for gridworld ---------------------------------
def display_state(agent_state: AgentState, replay, env,
                  max_steps=100, rendering='local', savepath=None):
    exploration_state = agent_state.exploration_state
    policy_state = agent_state.policy_state

    # min_count_map = gridworld.render_function(
    #     jax.partial(density.get_count_batch, exploration_state.density_state),
    #     env, reduction=jnp.min)
    sum_count_map = gridworld.render_function(
        jax.partial(density.get_count_batch, exploration_state.density_state),
        env, reduction=jnp.sum)
    novq_map = gridworld.render_function(
        jax.partial(q_learning.predict_value, exploration_state.novq_state),
        env, reduction=jnp.max)
    optimistic_novq_map = gridworld.render_function(
        jax.partial(predict_optimistic_value_batch,
                    exploration_state.novq_state,
                    exploration_state.density_state,
                    exploration_state.prior_count),
        env, reduction=jnp.max)
    taskq_map = gridworld.render_function(
        jax.partial(q_learning.predict_value, policy_state.q_state),
        env, reduction=jnp.max)
    novelty_reward_map = gridworld.render_function(
        jax.partial(compute_novelty_reward, exploration_state),
        env, reduction=jnp.max)
    traj_map = replay_buffer.render_trajectory(replay, max_steps, env)

    subfigs = [
        # (min_count_map, "Visit count (min)"),
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

    utils.display_figure(fig, rendering, savepath=savepath)
# -------------------------------------------------------------------


def main(args):
    rng = random.PRNGKey(args.seed)
    env = gridworld.new(args.env_size)
    state_shape = (2, env.size)
    action_shape = (1,)
    batch_size = 128

    # drawing only one candidate action sample from the policy
    # will result in following the policy directly
    n_candidates = 64 if args.use_exploration else 1

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
                                         target_novq_state=novq_state,
                                         density_state=density_state,
                                         temperature=args.temperature,
                                         prior_count=args.prior_count)
    agent_state = AgentState(exploration_state=exploration_state,
                             policy_state=policy_state,
                             policy_action_fn=policy.action_fn,
                             policy_update_fn=policy.update_fn,
                             n_candidates=n_candidates,
                             n_update_candidates=args.n_update_candidates)

    for episode in range(1, 100000):
        # run an episode
        rngs = random.split(rng, args.max_steps + 1)
        rng = rngs[0]

        agent_state, env, score = run_episode(
            agent_state, replay, rngs[1:], env,
            train=True, max_steps=args.max_steps)

        # update the task policy
        # TODO: pull this loop inside the policy_update_fn
        policy_state = agent_state.policy_state
        for _ in range(10):
            transitions = tuple((jnp.array(el)
                                 for el in replay.sample(batch_size)))
            policy_state = agent_state.policy_update_fn(policy_state,
                                                        transitions)
        # hacky reset of targetq to q
        policy_state = policy_state.replace(targetq_state=policy_state.q_state)
        agent_state = agent_state.replace(policy_state=policy_state)

        # update the target novelty Q function
        agent_state = update_target_q(agent_state)

        # output / visualize
        if episode % args.eval_every == 0:
            rngs = random.split(rng, args.max_steps + 1)
            rng = rngs[0]
            _, _, test_score = run_episode(
                agent_state, replay, rngs[1:], env,
                train=False, max_steps=args.max_steps)
            print((f"Episode {episode:4d}"
                   f", Train score {score:3d}"
                   f", Test score {test_score:3d}"))
            if args.vis != 'none':
                savepath = f"results/exploration/{args.name}/{episode}.png"
                display_state(agent_state, replay, env,
                              max_steps=args.max_steps, rendering=args.vis,
                              savepath=savepath)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='default')
    parser.add_argument('--seed', default=0)
    parser.add_argument('--tabular', action='store_true', default=False)
    parser.add_argument('--env_size', type=int, default=5)
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--vis', default='local')
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--prior_count', type=float, default=1)
    parser.add_argument('--n_update_candidates', type=int, default=64)

    parser.add_argument('--no_exploration', dest='use_exploration',
                        action='store_false', default=True)
    args = parser.parse_args()

    if args.tabular:
        import tabular_q_functions as q_functions
        import policies.tabular_q_policy as policy
    else:
        # import deep_q_functions as q_functions
        import fullonehot_deep_q_functions as q_functions
        import policies.deep_q_policy as policy

    jit = not args.debug
    if jit:
        main(args)
    else:
        with jax.disable_jit():
            main(args)
