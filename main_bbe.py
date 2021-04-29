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

import replay_buffer
import q_learning
import utils
from environments.observation_domains import DOMAINS
from environments import jax_specs
from policies.pytorch_sac.video import VideoRecorder


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


@struct.dataclass
class AgentState():
    """A container for the entire state; not jittable.
    """
    exploration_state: ExplorationState
    policy_state: Any = struct.field(pytree_node=False)
    replay: Any = struct.field(pytree_node=False)
    j_action_spec: Any
    n_candidates: int
    n_update_candidates: int
    n_updates_per_step: int
    update_target_every: int
    warmup_steps: int
    optimistic_actions: bool
    uniform_update_candidates: bool
    batch_size: int
    steps_since_tupdate: int = 0


@jax.jit
def _novelty_given_counts(counts):
    ones = jnp.ones(jnp.array(counts).shape)
    rewards = (counts + 1e-8) ** (-0.5)
    options = jnp.stack([ones, rewards], axis=1)

    # Clip rewards to be at most 1 (when count is 0)
    return jnp.min(options, axis=1)


@jax.profiler.trace_function
def compute_novelty_reward(exploration_state, states, actions):
    """Returns a novelty reward in [0, 1] for each (s, a) pair."""
    counts = density.get_count_batch(
        exploration_state.density_state, states, actions)

    return _novelty_given_counts(counts)


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
                train=True, max_steps=None, video_recorder=None,
                explore_only=False, bonus_scale=1):
    timestep = env.reset()
    score, novelty_score = 0, 0

    i = 0
    while not timestep.last():
        if video_recorder is not None:
            video_recorder.record(env)
        rng, action_rng = random.split(rng)
        s = utils.flatten_observation(timestep.observation)

        replay = agent_state.replay
        warmup_steps = agent_state.warmup_steps

        # put some random steps in the replay buffer
        if len(replay) < warmup_steps:
            action_spec = jax_specs.convert_dm_spec(env.action_spec())
            a = utils.sample_uniform_actions(action_spec, action_rng, 1)[0]
            flag = 'train' if train else 'test'
            logger.update(f'{flag}/policy_entropy', np.nan)
            logger.update(f'{flag}/explore_entropy', np.nan)
            logger.update(f'{flag}/alpha', np.nan)
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
                train_r = r + bonus_scale * novelty_reward
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

    if 'object_pos' in ospec:
        vis_elem = {'object_pos'}
    elif 'orientations' in ospec and 'height' in ospec:
        vis_elem = {'height', 'orientations'}
    else:
        vis_elem = None

    render_function = jax.partial(utils.render_function, vis_elem=vis_elem)

    # min_count_map = dmcontrol_gridworld.render_function(
    #     jax.partial(density.get_count_batch, exploration_state.density_state),
    #     env, reduction=jnp.min)
    count_map = render_function(
        jax.partial(density.get_count_batch, exploration_state.density_state),
        agent_state.replay,
        ospec, aspec, reduction=jnp.max, bins=bins)
    novelty_reward_map = render_function(
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
        taskq_map = render_function(
            jax.partial(q_learning.predict_value, policy_state.q_state),
            agent_state.replay,
            ospec, aspec, reduction=jnp.max, bins=bins)
        subfigs.append((taskq_map, "Task value (max)"))
    elif policy.__name__ == 'policies.sac_policy':
        import torch
        def get_task_value(s, a):
            s = torch.FloatTensor(np.array(s)).to(policy_state.device)
            a = torch.FloatTensor(np.array(a)).to(policy_state.device)
            with torch.no_grad():
                v = policy_state.critic.Q1(torch.cat([s, a], dim=-1))
            return v.cpu().detach().numpy()
        taskq_map = render_function(
            get_task_value,
            agent_state.replay,
            ospec, aspec, bins=bins)
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

    fig_path = f"{savedir}/vis/{episode}.png"
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

    density_state = density.new(observation_spec, action_spec,
                                state_bins=args.n_state_bins,
                                action_bins=args.n_action_bins,
                                state_scale=args.density_state_scale,
                                action_scale=args.density_action_scale,
                                max_obs=args.density_max_obs,
                                tolerance=args.density_tolerance,
                                reweight_dropped=args.density_reweight_dropped,
                                conserve_weight=args.density_conserve_weight)

    replay = replay_buffer.Replay(state_shape, action_shape)

    policy_state = policy.init_fn(observation_spec, action_spec, args.seed,
                                  lr=args.policy_lr,
                                  update_rule=args.policy_update,
                                  temp=args.policy_temperature,
                                  test_temp=args.policy_test_temperature)

    exploration_state = ExplorationState(
        novq_state=None,
        target_novq_state=None,
        density_state=density_state,
        temperature=None,
        update_temperature=None,
        prior_count=None)
    agent_state = AgentState(exploration_state=exploration_state,
                             policy_state=policy_state,
                             replay=replay,
                             j_action_spec=j_action_spec,
                             n_candidates=None,
                             n_update_candidates=None,
                             n_updates_per_step=None,
                             update_target_every=None,
                             warmup_steps=args.warmup_steps,
                             optimistic_actions=None,
                             uniform_update_candidates=None,
                             batch_size=None)

    current_time = np.nan
    for episode in range(1, args.max_episodes + 1):
        last_time = current_time
        current_time = time.time()
        logger.update('train/elapsed', current_time - last_time)

        # run an episode
        rng, episode_rng = random.split(rng)
        video_recorder = VideoRecorder(args.save_dir, fps=args.max_steps/10)
        video_recorder.init(enabled=(episode % args.video_every == 0))
        agent_state, env, score, novelty_score = run_episode(
            agent_state, episode_rng, env,
            train=True, max_steps=args.max_steps,
            video_recorder=video_recorder,
            explore_only=args.explore_only, bonus_scale=args.bonus_scale)
        video_recorder.save(f'train_{episode}.mp4')
        logger.update('train/episode', episode)
        logger.update('train/score', score)
        logger.update('train/novelty_score', novelty_score)

        # update the task policy
        # TODO: pull this loop inside the policy.update_fn
        policy_state = agent_state.policy_state
        for _ in range(int(args.max_steps * args.policy_updates_per_step)):
            transitions = agent_state.replay.sample(1024)
            transitions = tuple((jnp.array(el) for el in transitions))
            policy_state = policy.update_fn(
                policy_state, transitions)
        agent_state = agent_state.replace(policy_state=policy_state)

        # output / visualize
        if episode % args.eval_every == 0:
            rng, episode_rng = random.split(rng)
            video_recorder = VideoRecorder(args.save_dir, fps=args.max_steps/10)
            video_recorder.init(enabled=(episode % args.video_every == 0))
            _, _, test_score, test_novelty_score = run_episode(
                agent_state, episode_rng, env,
                train=False, max_steps=args.max_steps,
                video_recorder=video_recorder,
                explore_only=args.explore_only, bonus_scale=args.bonus_scale)
            video_recorder.save(f'test_{episode}.mp4')
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

    # basic environment settings
    parser.add_argument('--name', default='default')
    parser.add_argument('--env', default='gridworld')
    parser.add_argument('--task', default='default')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env_size', type=int, default=20)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--max_episodes', type=int, default=1000)

    # visualization and logging
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--vis', default='disk')
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--video_every', type=int, default=10)
    parser.add_argument('--save_replay_every', type=int, default=10000000)
    parser.add_argument('--warmup_steps', type=int, default=128)

    # policy settings
    parser.add_argument('--policy', type=str, default='deep_q')
    parser.add_argument('--policy_update', type=str, default='ddqn')
    parser.add_argument('--policy_lr', type=float, default=1e-3)
    parser.add_argument('--policy_temperature', type=float, default=3e-1)
    parser.add_argument('--policy_test_temperature', type=float, default=1e-1)
    parser.add_argument('--policy_updates_per_step', type=float, default=1)

    # count settings
    parser.add_argument('--density', type=str, default='tabular')
    parser.add_argument('--density_state_scale', type=float, default=1.)
    parser.add_argument('--density_action_scale', type=float, default=1.)
    parser.add_argument('--density_max_obs', type=float, default=1e5)
    parser.add_argument('--density_tolerance', type=float, default=0.95)
    parser.add_argument('--density_reweight_dropped', action='store_true')
    parser.add_argument('--density_conserve_weight', action='store_true')
    parser.add_argument('--prior_count', type=float, default=1e-3)

    # tabular settings (also for vis)
    parser.add_argument('--n_state_bins', type=int, default=20)
    parser.add_argument('--n_action_bins', type=int, default=2)

    # bbe settings
    parser.add_argument('--explore_only', action='store_true', default=False)
    parser.add_argument('--bonus_scale', type=float, default=1)

    args = parser.parse_args()
    print(args)

    args.save_dir = f"results/bbe/{args.name}"
    os.makedirs(args.save_dir, exist_ok=True)
    import experiment_logging
    experiment_logging.setup_default_logger(args.save_dir)
    from experiment_logging import default_logger as logger

    import json
    with open(args.save_dir + '/args.json', 'w') as argfile:
        json.dump(args.__dict__, argfile, indent=4)

    if args.policy == 'deep_q':
        import policies.deep_q_policy as policy
    elif args.policy == 'sac':
        import policies.sac_policy as policy
    elif args.policy == 'uniform':
        import policies.uniform_policy as policy
    elif args.policy == 'tabular':
        import policies.tabular_q_policy as policy
    else:
        raise Exception("Argument --policy was invalid.")

    if args.density == 'tabular':
        import densities.tabular_density as density
    elif args.density == 'kernel':
        import densities.kernel_density as density
    elif args.density == 'kernel_count':
        import densities.kernel_count as density
    elif args.density == 'faiss_kernel_count':
        from densities import faiss_kernel_count as density
    elif args.density == 'keops_kernel_count':
        from densities import keops_kernel_count as density
    elif args.density == 'dummy':
        import densities.dummy_density as density
    else:
        raise Exception("Argument --density was invalid.")

    print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    jit = not args.debug
    if jit:
        main(args)
    else:
        with jax.disable_jit():
            main(args)
