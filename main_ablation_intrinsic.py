# Main differences in this ablation:
# - there is no novelty Q
# - there is no optimism
# - the policy is trained on rewards r + novelty_reward

import time
import os
import math
import pickle
import queue
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

import jax
from jax import numpy as jnp, random, lax

import flax
from flax import nn, optim, struct

from dm_control import suite

import dmcontrol_gridworld
import replay_buffer
import q_learning
import tabular_density as density
import utils
from observation_domains import DOMAINS
import jax_specs
import point


R_MAX = 100


@struct.dataclass
class ExplorationState():
    """The pure-JAX components that can be jitted/vmapped.
    """
    novq_state: q_learning.QLearnerState
    target_novq_state: q_learning.QLearnerState
    density_state: density.DensityState
    temperature: float
    update_temperature: float
    prior_count: float
    optimistic_updates: bool
    target_network: bool
    # density_fns: Any


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
    steps_since_tupdate: int = 0
    # policy_fns: Any = struct.field(pytree_node=False)


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
    return agent_state


@jax.profiler.trace_function
def sample_action(agent_state: AgentState, rng, s, train=True):
    s_batch = jnp.expand_dims(s, axis=0)
    policy_state, actions = policy.action_fn(agent_state.policy_state, s_batch,
                                             n=1, explore=train)
    action = actions[0, 0]
    agent_state = agent_state.replace(policy_state=policy_state)

    flag = 'train' if train else 'test'
    logger.update(f'{flag}/explore_entropy', 0)
    return agent_state, action


def update_agent(agent_state: AgentState, rng, transition):
    # add transition to replay
    transition_id = agent_state.replay.append(*transition)

    # update the density with the observed transition
    agent_state = update_exploration(agent_state, rng, transition_id)
    return agent_state


def run_episode(agent_state: AgentState, rng, env,
                train=True, max_steps=None, explore_only=False):
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
            agent_state, a = sample_action(
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
            if explore_only:
                train_r = novelty_reward
            else:
                train_r = r + novelty_reward
            transition = (s, a, sp, train_r)
            rng, update_rng = random.split(rng)
            agent_state = update_agent(agent_state, update_rng, transition)
        i += 1
        if max_steps is not None and i >= max_steps:
            break
    return agent_state, env, score, novelty_score


# ----- Visualizations for gridworld ---------------------------------
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
    novelty_reward_map = utils.render_function(
        jax.partial(compute_novelty_reward, exploration_state),
        agent_state.replay,
        ospec, aspec, reduction=jnp.max, bins=bins)
    traj_map = replay_buffer.render_trajectory(
        agent_state.replay, max_steps, ospec, bins=bins)


    subfigs = [
        # (min_count_map, "Visit count (min)"),
        (count_map, "Visit count (max)"),
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
    n_candidates = None

    novq_state = None

    density_state = density.new(observation_spec, action_spec,
                                state_bins=args.n_state_bins,
                                action_bins=args.n_action_bins)

    # for gridworld we can discretize with one bit per dimension
    replay = replay_buffer.LowPrecisionTracingReplay(
        state_shape, action_shape, min_s=0, max_s=1, n_bins=2)

    policy_state = policy.init_fn(observation_spec, action_spec, args.seed,
                                  lr=args.policy_lr,
                                  update_rule=args.policy_update)

    exploration_state = ExplorationState(
        novq_state=novq_state,
        target_novq_state=novq_state,
        density_state=density_state,
        temperature=None,
        update_temperature=None,
        prior_count=args.prior_count,
        optimistic_updates=None,
        target_network=None)
    agent_state = AgentState(exploration_state=exploration_state,
                             policy_state=policy_state,
                             replay=replay,
                             n_candidates=n_candidates,
                             n_update_candidates=None,
                             prioritized_update=None,
                             update_target_every=None,
                             warmup_steps=args.warmup_steps)

    for episode in range(1, 1000):
        # run an episode
        rng, episode_rng = random.split(rng)
        agent_state, env, score, novelty_score = run_episode(
            agent_state, episode_rng, env,
            train=True, max_steps=args.max_steps,
            explore_only=args.explore_only)
        logger.update('train/episode', episode)
        logger.update('train/score', score)
        logger.update('train/novelty_score', novelty_score)

        # update the task policy
        # TODO: pull this loop inside the policy.update_fn
        n_updates = args.max_steps // 2
        policy_state = agent_state.policy_state
        for _ in range(n_updates):
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
                train=False, max_steps=args.max_steps,
                explore_only=args.explore_only)
            logger.update('test/episode', episode)
            logger.update('test/score', test_score)
            logger.update('test/novelty_score', test_novelty_score)

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

    parser.add_argument('--prior_count', type=float, default=1e-3)
    parser.add_argument('--n_state_bins', type=int, default=4)
    parser.add_argument('--n_action_bins', type=int, default=2)
    parser.add_argument('--warmup_steps', type=int, default=128)

    parser.add_argument('--explore_only', action='store_true', default=False)
    args = parser.parse_args()
    print(args)

    args.save_dir = f"results/intrinsic/{args.name}"
    os.makedirs(args.save_dir, exist_ok=True)
    import experiment_logging
    experiment_logging.setup_default_logger(args.save_dir)
    from experiment_logging import default_logger as logger

    import json
    with open(args.save_dir + '/args.json', 'w') as argfile:
        json.dump(args.__dict__, argfile, indent=4)

    if args.policy == 'deep':
        import policies.deep_q_policy as policy
    elif args.policy == 'uniform':
        import policies.uniform_policy as policy
    elif args.policy == 'tabular':
        import policies.tabular_q_policy as policy
    else:
        raise Exception("Argument --policy was invalid.")

    jit = not args.debug
    if jit:
        main(args)
    else:
        with jax.disable_jit():
            main(args)
