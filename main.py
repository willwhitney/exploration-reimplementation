import time
import os
import math
import pickle
import queue
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

import jax
from jax import numpy as jnp, random
from flax import nn, struct

from dm_control import suite
import point

import dmcontrol_gridworld
import replay_buffer
import q_learning
import utils
from observation_domains import DOMAINS
import jax_specs


R_MAX = 100


@struct.dataclass
class ExplorationState():
    """The pure-JAX components that can be jitted/vmapped.
    """
    novq_state: q_learning.QLearnerState
    target_novq_state: q_learning.QLearnerState
    density_state: Any
    temperature: float
    update_temperature: float
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
    update_target_every: int
    warmup_steps: int
    optimistic_actions: bool
    steps_since_tupdate: int = 0


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
    """The jittable component of the exploration Q function training step."""
    states, actions, next_states, rewards = transitions
    discount = exploration_state.novq_state.discount
    temp = exploration_state.update_temperature

    # if use_target_network:
    #     target_q_state = exploration_state.target_novq_state
    # else:
    #     target_q_state = exploration_state.novq_state

    with jax.profiler.TraceContext("compute next value"):
        if use_optimistic_updates:
            next_values = predict_optimistic_values_batch(
                exploration_state.novq_state,
                exploration_state.density_state,
                exploration_state.prior_count,
                next_states, candidate_next_actions)
            next_values_target = predict_optimistic_values_batch(
                exploration_state.target_novq_state,
                exploration_state.density_state,
                exploration_state.prior_count,
                next_states, candidate_next_actions)
        else:
            next_values = q_learning.predict_action_values_batch(
                exploration_state.novq_state,
                next_states,
                candidate_next_actions)
            next_values_target = q_learning.predict_action_values_batch(
                exploration_state.target_novq_state,
                next_states,
                candidate_next_actions)

    with jax.profiler.TraceContext("compute targets"):
        # double DQN rule:
        # - select next action according to current Q
        # - evaluate it according to target Q
        next_value_probs = nn.softmax(next_values / temp, axis=1)
        next_value_elements = (next_value_probs * next_values_target)
        expected_next_values = next_value_elements.sum(axis=1)
        expected_next_values = expected_next_values.reshape(rewards.shape)

        # compute targets and update
        novelty_reward = compute_novelty_reward(
            exploration_state, states, actions).reshape(rewards.shape)
        q_targets = novelty_reward + discount * expected_next_values

        # clip targets to be within the feasible set
        q_targets = jnp.minimum(q_targets, R_MAX)

    # import ipdb; ipdb.set_trace()

    with jax.profiler.TraceContext("Q update"):
        novq_state, losses = q_functions.train_step(
            exploration_state.novq_state,
            states, actions, q_targets)

    return exploration_state.replace(novq_state=novq_state), losses


@jax.profiler.trace_function
def train_step(agent_state, transitions):
    """A full (optimistic) training step for the exploration Q function."""
    states, actions, next_states, rewards = transitions

    # candidate actions should be (bsize x n_update_candidates x *action_shape)
    with jax.profiler.TraceContext("get n_update_candidates"):
        n_update_candidates = int(agent_state.n_update_candidates)
    with jax.profiler.TraceContext("get candidates"):
        policy_state, candidate_next_actions = policy.action_fn(
            agent_state.policy_state, next_states,
            n_update_candidates, True)
        agent_state = agent_state.replace(policy_state=policy_state)

    # import ipdb; ipdb.set_trace()
    # somehow if I don't cast these to bool JAX will recompile the jitted
    # function train_step_candidates on every call...
    with jax.profiler.TraceContext("train_step_candidates"):
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

    visited = set()

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
                # if transition_id not in visited:
                transition_ids.append(transition_id)
                # visited.add(transition_id)
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
            if preceding_id not in visited:
                pqueue.put((-loss, preceding_id))
                visited.add(preceding_id)

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
    agent_state = agent_state.replace(exploration_state=exploration_state,
                                      steps_since_tupdate=0)
    return agent_state


def uniform_update(agent_state, rng):
    for _ in range(10):
        transitions = tuple((jnp.array(el)
                             for el in agent_state.replay.sample(128)))
        agent_state, losses = train_step(agent_state, transitions)
        agent_state = agent_state.replace(
            steps_since_tupdate=agent_state.steps_since_tupdate + 1)
        if agent_state.steps_since_tupdate >= agent_state.update_target_every:
            agent_state = update_target_q(agent_state)
    return agent_state


@jax.profiler.trace_function
def update_exploration(agent_state, rng, transition_id):
    s, a, sp, r = agent_state.replay.get_transitions(transition_id)

    # update density on new observations
    with jax.profiler.TraceContext("update density"):
        exploration_state = agent_state.exploration_state
        density_state = density.update_batch(exploration_state.density_state,
                                            jnp.expand_dims(s, axis=0),
                                            jnp.expand_dims(a, axis=0))
        exploration_state = exploration_state.replace(
            density_state=density_state)
        agent_state = agent_state.replace(exploration_state=exploration_state)

    if len(agent_state.replay) > agent_state.warmup_steps:
        # update exploration Q to consistency with new density
        rng, novq_rng = random.split(rng)

        if agent_state.prioritized_update:
            agent_state = prioritized_update(agent_state, transition_id)
        else:
            with jax.profiler.TraceContext("uniform update"):
                agent_state = uniform_update(agent_state, novq_rng)
    return agent_state


def compute_weight(prior_count, count):
    root_real_count = count ** 0.5
    # root_prior_count = prior_count ** 0.5
    # return root_real_count / (root_real_count + root_prior_count)
    root_total_count = (count + prior_count) ** 0.5
    return root_real_count / root_total_count


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
def select_candidate_optimistic(exploration_state, rng,
                                state, candidate_actions):
    optimistic_values = predict_optimistic_values(
        exploration_state.novq_state,
        exploration_state.density_state,
        exploration_state.prior_count,
        state, candidate_actions).reshape(-1)

    return q_learning.sample_boltzmann(
        rng, optimistic_values, candidate_actions,
        exploration_state.temperature)


@jax.profiler.trace_function
def sample_exploration_action(agent_state: AgentState, rng, s, train=True):
    # during test, take only one action sample from the task policy
    # -> will follow the task policy
    n = agent_state.n_candidates if train else 1

    with jax.profiler.TraceContext("sample candidate actions"):
        s_batch = jnp.expand_dims(s, axis=0)
        policy_state, candidate_actions = policy.action_fn(
            agent_state.policy_state, s_batch, n, train)

    # policy.action_fn deals with batches and we only have one element
    candidate_actions = candidate_actions[0]
    agent_state = agent_state.replace(policy_state=policy_state)

    with jax.profiler.TraceContext("select from candidates"):
        if agent_state.optimistic_actions:
            a, h = select_candidate_optimistic(agent_state.exploration_state,
                                               rng, s, candidate_actions)
        else:
            a, _, h = q_learning.sample_action_boltzmann(
                agent_state.exploration_state.novq_state, rng,
                s, candidate_actions,
                agent_state.exploration_state.temperature)
    flag = 'train' if train else 'test'
    logger.update(f'{flag}/explore_entropy', h)
    return agent_state, a


def update_agent(agent_state: AgentState, rng, transition):
    # add transition to replay
    transition_id = agent_state.replay.append(*transition)

    # update the exploration policy and density with the observed transition
    agent_state = update_exploration(agent_state, rng, transition_id)
    return agent_state


def run_episode(agent_state: AgentState, rng, env,
                train=True, max_steps=None):
    timestep = env.reset()
    score, novelty_score = 0, 0

    i = 0
    while not timestep.last():
        rng, action_rng = random.split(rng)
        s = utils.flatten_observation(timestep.observation)

        # put some random steps in the replay buffer
        if len(agent_state.replay) < agent_state.warmup_steps:
            action_spec = jax_specs.convert_dm_spec(env.action_spec())
            a = utils.sample_uniform_actions(action_spec, action_rng, 1)[0]
            flag = 'train' if train else 'test'
            logger.update(f'{flag}/policy_entropy', 0)
            logger.update(f'{flag}/explore_entropy', 0)
        else:
            agent_state, a = sample_exploration_action(
                agent_state, action_rng, s, train)
        timestep = env.step(a)

        sp = utils.flatten_observation(timestep.observation)
        r = timestep.reward

        novelty_reward = compute_novelty_reward(agent_state.exploration_state,
                                                jnp.expand_dims(s, axis=0),
                                                jnp.expand_dims(a, axis=0))
        score += r
        novelty_score += float(novelty_reward)

        if train:
            transition = (s, a, sp, r)
            rng, update_rng = random.split(rng)
            agent_state = update_agent(agent_state, update_rng, transition)
        i += 1
        if max_steps is not None and i >= max_steps:
            break
    return agent_state, env, score, novelty_score


# ------------- Visualizations ---------------------------------
@jax.profiler.trace_function
def display_state(agent_state: AgentState, ospec, aspec,
                  max_steps=100, bins=20,
                  rendering='local', savedir=None, episode=None):
    exploration_state = agent_state.exploration_state
    policy_state = agent_state.policy_state

    # min_count_map = dmcontrol_gridworld.render_function(
    #     jax.partial(density.get_count_batch, exploration_state.density_state),
    #     env, reduction=jnp.min)
    count_map = utils.render_function(
        jax.partial(density.get_count_batch, exploration_state.density_state),
        agent_state.replay,
        ospec, aspec, reduction=jnp.max, bins=bins)
    novq_map = utils.render_function(
        jax.partial(q_learning.predict_value, exploration_state.novq_state),
        agent_state.replay,
        ospec, aspec, reduction=jnp.max, bins=bins)
    optimistic_novq_map = utils.render_function(
        jax.partial(predict_optimistic_value_batch,
                    exploration_state.novq_state,
                    exploration_state.density_state,
                    exploration_state.prior_count),
        agent_state.replay,
        ospec, aspec, reduction=jnp.max, bins=bins)
    novelty_reward_map = utils.render_function(
        jax.partial(compute_novelty_reward, exploration_state),
        agent_state.replay,
        ospec, aspec, reduction=jnp.max, bins=bins)
    traj_map = replay_buffer.render_trajectory(
        agent_state.replay, max_steps, ospec, bins=bins)


    subfigs = [
        # (min_count_map, "Visit count (min)"),
        (count_map, "Visit count (max)"),
        (novq_map, "Novelty value (max)"),
        (optimistic_novq_map, "Optimistic novelty value (max)"),
        (novelty_reward_map, "Novelty reward (max)"),
        (traj_map, "Last trajectory"),
    ]

    q_policies = ['policies.deep_q_policy', 'policies.tabular_q_policy']
    if policy.__name__ in q_policies:
        taskq_map = utils.render_function(
            jax.partial(q_learning.predict_value, policy_state.q_state),
            agent_state.replay,
            ospec, aspec, reduction=jnp.max, bins=bins)
        subfigs.append((taskq_map, "Task value (max)"))

    # dump the raw data for later rendering
    raw_path = f"{savedir}/data/{episode}.pkl"
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, 'wb') as f:
        pickle.dump(subfigs, f, protocol=4)

    fig, axs = plt.subplots(1, len(subfigs))
    for ax, subfig in zip(axs, subfigs):
        render, title = subfig
        img = ax.imshow(render)
        fig.colorbar(img, ax=ax)
        ax.set_title(title)
    fig.set_size_inches(4 * len(subfigs), 3)

    fig_path = f"{savedir}/{episode}.png"
    utils.display_figure(fig, rendering, savepath=fig_path)
# -------------------------------------------------------------------


def main(args):
    rng = random.PRNGKey(args.seed)
    if args.env == 'gridworld':
        env = dmcontrol_gridworld.GridWorld(args.env_size, args.max_steps)
        observation_spec = env.observation_spec()
    else:
        env = suite.load(args.env, args.task)
        observation_spec = DOMAINS[args.env][args.task]

    action_spec = env.action_spec()
    j_action_spec = jax_specs.convert_dm_spec(action_spec)
    j_observation_spec = jax_specs.convert_dm_spec(observation_spec)

    state_shape = utils.flatten_spec_shape(j_observation_spec)
    action_shape = action_spec.shape
    batch_size = 128

    # drawing only one candidate action sample from the policy
    # will result in following the policy directly
    n_candidates = 64 if args.use_exploration else 1

    novq_state = q_functions.init_fn(args.seed,
                                     observation_spec,
                                     action_spec,
                                    #  env_size=env.size,
                                     discount=0.97,
                                     max_value=R_MAX)

    density_state = density.new(observation_spec, action_spec,
                                state_bins=args.n_state_bins,
                                action_bins=args.n_action_bins,
                                state_std_scale=args.density_state_scale,
                                action_std_scale=args.density_action_scale)

    # for gridworld we can discretize with one bit per dimension
    replay = replay_buffer.LowPrecisionTracingReplay(
        state_shape, action_shape, min_s=0, max_s=1, n_bins=2)

    policy_state = policy.init_fn(observation_spec, action_spec, args.seed,
                                  lr=args.policy_lr,
                                  update_rule=args.policy_update,
                                  temp=args.policy_temperature,
                                  test_temp=args.policy_test_temperature)

    exploration_state = ExplorationState(
        novq_state=novq_state,
        target_novq_state=novq_state,
        density_state=density_state,
        temperature=args.temperature,
        update_temperature=args.update_temperature,
        prior_count=args.prior_count,
        optimistic_updates=args.optimistic_updates,
        target_network=args.target_network)
    agent_state = AgentState(exploration_state=exploration_state,
                             policy_state=policy_state,
                             replay=replay,
                             n_candidates=n_candidates,
                             n_update_candidates=args.n_update_candidates,
                             prioritized_update=args.prioritized_update,
                             update_target_every=args.update_target_every,
                             warmup_steps=args.warmup_steps,
                             optimistic_actions=args.optimistic_actions,)

    current_time = time.time()
    for episode in range(1, 1000 + 1):
        last_time = current_time
        current_time = time.time()
        logger.update('train/elapsed', current_time - last_time)

        # run an episode
        rng, episode_rng = random.split(rng)
        agent_state, env, score, novelty_score = run_episode(
            agent_state, episode_rng, env, train=True, max_steps=args.max_steps)
        logger.update('train/episode', episode)
        logger.update('train/score', score)
        logger.update('train/novelty_score', novelty_score)

        # update the task policy
        # TODO: pull this loop inside the policy.update_fn
        policy_state = agent_state.policy_state
        for _ in range(args.max_steps // 2):
            transitions = agent_state.replay.sample(batch_size)
            transitions = tuple((jnp.array(el) for el in transitions))
            policy_state = policy.update_fn(
                policy_state, transitions)
        agent_state = agent_state.replace(policy_state=policy_state)

        # output / visualize
        if episode % args.eval_every == 0:
            rng, episode_rng = random.split(rng)
            _, _, test_score, test_novelty_score = run_episode(
                agent_state, episode_rng, env,
                train=False, max_steps=args.max_steps)
            logger.update('test/episode', episode)
            logger.update('test/score', test_score)
            logger.update('test/novelty_score', test_novelty_score)

            density_state = agent_state.exploration_state.density_state
            if hasattr(density_state, "total"):
                logger.update('train/density_size', density_state.total)

            logger.write_all()

            if args.vis != 'none':
                # savepath = f"{args.save_dir}/{episode}"
                display_state(agent_state, observation_spec, action_spec,
                              max_steps=args.max_steps, bins=args.n_state_bins,
                              rendering=args.vis, savedir=args.save_dir,
                              episode=episode)

        if episode % args.save_replay_every == 0:
            replay_path = f"{args.save_dir}/replay.pkl"
            replay_buffer.save(agent_state.replay, replay_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='default')
    parser.add_argument('--env', default='gridworld')
    parser.add_argument('--task', default='default')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env_size', type=int, default=20)
    parser.add_argument('--max_steps', type=int, default=1000)

    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--vis', default='disk')
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--save_replay_every', type=int, default=10)

    parser.add_argument('--policy', type=str, default='deep')
    parser.add_argument('--policy_update', type=str, default='ddqn')
    parser.add_argument('--policy_lr', type=float, default=1e-3)
    parser.add_argument('--policy_temperature', type=float, default=3e-1)
    parser.add_argument('--policy_test_temperature', type=float, default=1e-1)

    parser.add_argument('--density', type=str, default='tabular')
    parser.add_argument('--density_state_scale', type=float, default=1.)
    parser.add_argument('--density_action_scale', type=float, default=1.)

    parser.add_argument('--novelty_q_function', type=str, default='deep')
    parser.add_argument('--temperature', type=float, default=1e-1)
    parser.add_argument('--update_temperature', type=float, default=None)
    parser.add_argument('--prior_count', type=float, default=1e-3)
    parser.add_argument('--n_update_candidates', type=int, default=64)
    parser.add_argument('--n_state_bins', type=int, default=20)
    parser.add_argument('--n_action_bins', type=int, default=2)
    parser.add_argument('--no_optimistic_updates', dest='optimistic_updates',
                        action='store_false', default=True)
    parser.add_argument('--no_optimistic_actions', dest='optimistic_actions',
                        action='store_false', default=True)
    parser.add_argument('--target_network', action='store_true', default=True)
    parser.add_argument('--no_target_network', dest='target_network',
                        action='store_false')
    parser.add_argument('--update_target_every', type=int, default=10)
    parser.add_argument('--warmup_steps', type=int, default=128)

    parser.add_argument('--no_exploration', dest='use_exploration',
                        action='store_false', default=True)
    parser.add_argument('--prioritized_update', dest='prioritized_update',
                        action='store_true', default=False)
    parser.add_argument('--no_prioritized_update', dest='prioritized_update',
                        action='store_false')

    parser.add_subparsers()
    args = parser.parse_args()
    print(args)
    if args.update_temperature is None:
        print("Using --temperature as --update_temperature.")
        args.update_temperature = args.temperature

    args.save_dir = f"results/exploration/{args.name}"
    os.makedirs(args.save_dir, exist_ok=True)
    import experiment_logging
    experiment_logging.setup_default_logger(args.save_dir)
    from experiment_logging import default_logger as logger

    import json
    with open(args.save_dir + '/args.json', 'w') as argfile:
        json.dump(args.__dict__, argfile, indent=4)

    if args.novelty_q_function == 'deep':
        import deep_q_functions as q_functions
    elif args.novelty_q_function == 'sigmoid':
        import sigmoid_q_functions as q_functions
    elif args.novelty_q_function == 'tabular':
        import tabular_q_functions as q_functions
    else:
        raise Exception("Argument --novelty_q_function was invalid.")

    if args.policy == 'deep':
        import policies.deep_q_policy as policy
    elif args.policy == 'uniform':
        import policies.uniform_policy as policy
    elif args.policy == 'tabular':
        import policies.tabular_q_policy as policy
    else:
        raise Exception("Argument --policy was invalid.")

    if args.density == 'tabular':
        import tabular_density as density
    elif args.density == 'kernel':
        import kernel_density as density
    elif args.density == 'kernel_count':
        import kernel_count as density
    else:
        raise Exception("Argument --density was invalid.")

    jit = not args.debug
    if jit:
        main(args)
    else:
        with jax.disable_jit():
            main(args)
