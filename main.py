import time
import os
import math
import queue
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

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
    optimistic_updates: bool
    target_network: bool


@struct.dataclass
class AgentState():
    """A container for the entire state; not jittable.
    """
    exploration_state: ExplorationState
    policy_state: Any = struct.field(pytree_node=False)
    replay: Any = struct.field(pytree_node=False)
    n_candidates: int
    n_update_candidates: int
    prioritized_update: bool


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
@jax.partial(jax.jit, static_argnums=(3, 4))
def train_step_candidates(exploration_state: ExplorationState,
                          transitions,
                          candidate_next_actions,
                          use_target_network,
                          use_optimistic_updates):
    """The jittable component of the optimistic training step."""
    states, actions, next_states, rewards = transitions
    discount = exploration_state.novq_state.discount
    temp = exploration_state.temperature

    if use_target_network:
        target_q_state = exploration_state.target_novq_state
    else:
        target_q_state = exploration_state.novq_state

    if use_optimistic_updates:
        next_values = predict_optimistic_values_batch(
            target_q_state,
            exploration_state.density_state,
            exploration_state.prior_count,
            next_states, candidate_next_actions)
    else:
        next_values = q_learning.predict_action_values_batch(
            target_q_state,
            next_states,
            candidate_next_actions)

    next_value_probs = nn.softmax(next_values / temp, axis=1)
    next_value_elements = (next_value_probs * next_values)
    expected_next_values = next_value_elements.sum(axis=1)
    expected_next_values = expected_next_values.reshape(rewards.shape)

    # compute targets and update
    novelty_reward = compute_novelty_reward(
        exploration_state, states, actions).reshape(rewards.shape)
    q_targets = novelty_reward + discount * expected_next_values

    novq_state, losses = q_functions.train_step(
        exploration_state.novq_state,
        states, actions, q_targets)

    return exploration_state.replace(novq_state=novq_state), losses


@jax.profiler.trace_function
def train_step(agent_state, transitions):
    """A full (optimistic) training step."""
    states, actions, next_states, rewards = transitions

    # candidate actions should be (bsize x 64 x *action_shape)
    policy_state, candidate_next_actions = policy.action_fn(
        agent_state.policy_state, next_states,
        int(agent_state.n_update_candidates), True)
    agent_state = agent_state.replace(policy_state=policy_state)

    # somehow if I don't cast these to bool JAX will recompile the jitted
    # function train_step_candidates on every call...
    exploration_state, losses = train_step_candidates(
        agent_state.exploration_state,
        transitions,
        candidate_next_actions,
        bool(agent_state.exploration_state.target_network),
        bool(agent_state.exploration_state.optimistic_updates))
    agent_state = agent_state.replace(exploration_state=exploration_state)
    return agent_state, losses


@jax.profiler.trace_function
def prioritized_update(agent_state: AgentState, last_transition_id):
    pqueue = queue.PriorityQueue()
    pqueue.put((0, last_transition_id))
    replay = agent_state.replay

    error_threshold = 1e-1
    max_depth = 16
    max_bsize = 128

    def largest_pow(n):
        if n == 0:
            return 0
        else:
            return int(math.pow(2, int(math.log(n, 2))))

    def next_batch():
        queue_size = pqueue.qsize()

        # only use power-of-two batch sizes to limit the number of JIT
        # recompiles inside train_step
        batch_size = min(largest_pow(queue_size), max_bsize)

        transition_ids = []
        while len(transition_ids) < batch_size:
            try:
                _, transition_id = pqueue.get_nowait()
                transition_ids.append(transition_id)
            except queue.Empty:
                break
        if len(transition_ids) > 0:
            transitions = replay.get_transitions(np.array(transition_ids))
            return transitions
        else:
            return None

    def queue_predecessors(s, loss):
        preceding_ids = replay.predecessors(s)
        for preceding_id in preceding_ids:
            pqueue.put((-loss, preceding_id))

    n_updates = 0
    # update the tree up to a max depth
    for step in range(max_depth):
        # get the highest-priority transitions from the queue
        transitions = next_batch()
        if transitions is None:
            break

        # update on the current batch of transitions
        agent_state, losses = train_step(agent_state, transitions)
        n_updates += len(losses)

        # check the loss for each transition
        for i, loss in enumerate(losses):
            if loss > error_threshold:
                start_state = transitions[0][i]

                # add predecessors of transitions with large loss to the queue
                queue_predecessors(start_state, loss)
    # print(f"Max depth: {step}, total updates: {n_updates}")
    return agent_state


def update_target_q(agent_state: AgentState):
    exploration_state = agent_state.exploration_state.replace(
        target_novq_state=agent_state.exploration_state.novq_state)
    return agent_state.replace(exploration_state=exploration_state)


def uniform_update(agent_state, rng):
    for _ in range(10):
        transitions = tuple((jnp.array(el)
                             for el in agent_state.replay.sample(128)))
        agent_state, losses = train_step(agent_state, transitions)
    return agent_state


@jax.profiler.trace_function
def update_exploration(agent_state, rng, transition_id):
    s, a, sp, r = agent_state.replay.get_transitions(transition_id)

    # update density on new observations
    exploration_state = agent_state.exploration_state
    density_state = density.update_batch(exploration_state.density_state,
                                         jnp.expand_dims(s, axis=0),
                                         jnp.expand_dims(a, axis=0))
    exploration_state = exploration_state.replace(density_state=density_state)
    agent_state = agent_state.replace(exploration_state=exploration_state)

    # update exploration Q to consistency with new density
    rng, novq_rng = random.split(rng)

    if agent_state.prioritized_update:
        agent_state = prioritized_update(agent_state, transition_id)
    else:
        agent_state = uniform_update(agent_state, novq_rng)
    return agent_state


def compute_weight(prior_count, count):
    root_count = count ** 0.5
    root_prior_count = prior_count ** 0.5
    return root_count / (root_count + root_prior_count)


@jax.profiler.trace_function
@jax.jit
def predict_optimistic_value(novq_state, density_state, prior_count,
                             state, action):
    expanded_state = jnp.expand_dims(state, axis=0)
    expanded_action = jnp.expand_dims(action, axis=0)
    predicted_value = q_learning.predict_value(novq_state,
                                               expanded_state,
                                               expanded_action)
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

    return q_learning.sample_boltzmann(
        rng, optimistic_values, candidate_actions,
        exploration_state.temperature)


@jax.profiler.trace_function
def full_step(agent_state: AgentState, rng, env,
              train=True):
    # get env state
    with jax.profiler.TraceContext("env render"):
        s = gridworld.render(env)

    n = agent_state.n_candidates if train else 1

    # get candidate actions
    with jax.profiler.TraceContext("get action candidates"):
        s_batch = jnp.expand_dims(s, axis=0)
        policy_state, candidate_actions = policy.action_fn(
            agent_state.policy_state, s_batch, n, train)

    # policy.action_fn deals with batches and we only have one element
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

    novelty_reward = compute_novelty_reward(agent_state.exploration_state,
                                            jnp.expand_dims(s, axis=0),
                                            jnp.expand_dims(a, axis=0))
    novelty_reward = float(novelty_reward)
    if train:
        # add transition to replay
        with jax.profiler.TraceContext("replay append"):
            transition_id = agent_state.replay.append(s, a, sp, r)

        # update the exploration policy with the observed transition
        rng, update_rng = random.split(rng)

        if len(agent_state.replay) > 128:
            agent_state = update_exploration(
                agent_state, update_rng, transition_id)

    return agent_state, env, r, novelty_reward


def run_episode(agent_state: AgentState, rngs, env,
                train=True, max_steps=100, use_exploration=True):
    env = gridworld.reset(env)
    score = 0
    novelty_score = 0
    for i in range(max_steps):
        agent_state, env, r, novelty_r = full_step(
            agent_state, rngs[i], env, train=train)
        score += r
        novelty_score += novelty_r
    return agent_state, env, score, novelty_score


# ----- Visualizations for gridworld ---------------------------------
def display_state(agent_state: AgentState, env: gridworld.GridWorld,
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
    traj_map = replay_buffer.render_trajectory(
        agent_state.replay, max_steps, env)

    # print(f"Max novelty value: {novq_map.max() :.2f}")

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

    # for gridworld we can discretize with one bit per dimension
    replay = replay_buffer.LowPrecisionTracingReplay(
        state_shape, action_shape, min_s=0, max_s=1, n_bins=2)

    policy_state = policy.init_fn(args.seed,
                                  state_shape, action_shape,
                                  env.actions, env_size=env.size)

    exploration_state = ExplorationState(
        novq_state=novq_state,
        target_novq_state=novq_state,
        density_state=density_state,
        temperature=args.temperature,
        prior_count=args.prior_count,
        optimistic_updates=args.optimistic_updates,
        target_network=args.target_network)
    agent_state = AgentState(exploration_state=exploration_state,
                             policy_state=policy_state,
                             replay=replay,
                             n_candidates=n_candidates,
                             n_update_candidates=args.n_update_candidates,
                             prioritized_update=args.prioritized_update)

    for episode in range(1, 100000):
        # run an episode
        rngs = random.split(rng, args.max_steps + 1)
        rng = rngs[0]

        agent_state, env, score, novelty_score = run_episode(
            agent_state, rngs[1:], env,
            train=True, max_steps=args.max_steps)

        # update the task policy
        # TODO: pull this loop inside the policy.update_fn
        policy_state = agent_state.policy_state
        for _ in range(50):
            transitions = tuple((jnp.array(el)
                                 for el in agent_state.replay.sample(batch_size)))
            policy_state = policy.update_fn(policy_state, transitions)
        agent_state = agent_state.replace(policy_state=policy_state)

        # hacky reset of targetq to q
        if episode % 1 == 0:
            policy_state = agent_state.policy_state.replace(
                targetq_state=policy_state.q_state)
            agent_state = agent_state.replace(policy_state=policy_state)

        # update the target novelty Q function
        agent_state = update_target_q(agent_state)

        # output / visualize
        if episode % args.eval_every == 0:
            rngs = random.split(rng, args.max_steps + 1)
            rng = rngs[0]
            _, _, test_score, test_novelty_score = run_episode(
                agent_state, rngs[1:], env,
                train=False, max_steps=args.max_steps)
            print((f"Episode {episode:4d}"
                   f", Train score {score:3d}"
                   f", Train novelty score {novelty_score:3.0f}"
                   f", Test score {test_score:3d}"
                   f", Test novelty score {test_novelty_score:3.0f}"))
            if args.vis != 'none':
                savepath = f"results/exploration/{args.name}/{episode}.png"
                display_state(agent_state, env,
                              max_steps=args.max_steps, rendering=args.vis,
                              savepath=savepath)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='default')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env_size', type=int, default=20)
    parser.add_argument('--max_steps', type=int, default=100)

    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--vis', default='disk')
    parser.add_argument('--eval_every', type=int, default=10)

    parser.add_argument('--tabular', action='store_true', default=False)
    parser.add_argument('--temperature', type=float, default=1e-1)
    parser.add_argument('--prior_count', type=float, default=1e-3)
    parser.add_argument('--n_update_candidates', type=int, default=64)
    parser.add_argument('--no_optimistic_updates', dest='optimistic_updates',
                        action='store_false', default=True)
    parser.add_argument('--target_network', action='store_true', default=False)

    parser.add_argument('--no_exploration', dest='use_exploration',
                        action='store_false', default=True)
    parser.add_argument('--no_prioritized_update',
                        dest='prioritized_update',
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
