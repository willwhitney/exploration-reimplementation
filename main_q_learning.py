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

import replay_buffer
import q_learning
import utils
from environments.observation_domains import DOMAINS
from environments import jax_specs


@struct.dataclass
class AgentState():
    policy_state: Any = struct.field(pytree_node=False)
    replay: Any = struct.field(pytree_node=False)

def run_episode(agent_state, rng, env, train=True):
    timestep = env.reset()
    score = 0
    losses = []

    while not timestep.last():
        rng, action_rng = random.split(rng)
        s = utils.flatten_observation(timestep.observation)

        if len(agent_state.replay) < 1000:
            action_spec = jax_specs.convert_dm_spec(env.action_spec())
            a = utils.sample_uniform_actions(action_spec, action_rng, 1)[0]
        else:
            s_batch = jnp.expand_dims(s, axis=0)
            policy_state, candidate_actions = policy.action_fn(
                agent_state.policy_state, s_batch, 1, train)
            a = candidate_actions[0][0]
            agent_state = agent_state.replace(policy_state=policy_state)

        timestep = env.step(a)
        sp = utils.flatten_observation(timestep.observation)
        r = timestep.reward
        score += r

        if train:
            transition = (s, a, sp, r)
            agent_state.replay.append(*transition)
            transitions = agent_state.replay.sample(1024)
            transitions = tuple((jnp.array(el) for el in transitions))

            policy_state = agent_state.policy_state
            policy_state = policy.update_fn(policy_state, transitions)
            agent_state = agent_state.replace(policy_state=policy_state)
            losses.append(jnp.mean(agent_state.policy_state.last_losses))

    loss = np.array(losses).mean()
    return agent_state, env, score, loss

def main(args):
    rng = random.PRNGKey(0)
    if args.env == 'gridworld':
        from environments import dmcontrol_gridworld
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

    policy_state = policy.init_fn(observation_spec, action_spec, args.seed,
                                  lr=args.policy_lr,
                                  update_rule=args.policy_update,
                                  temp=args.policy_temperature,
                                  test_temp=args.policy_test_temperature)

    replay = replay_buffer.Replay(state_shape, action_shape)
    agent_state = AgentState(
        policy_state=policy_state,
        replay=replay)

    for episode in range(args.max_episodes):
        rng, episode_rng = random.split(rng)
        agent_state, env, score, loss = run_episode(
            agent_state, episode_rng, env, train=True)
        print((f'Episode: {episode :4.0f} Score: {score :4.0f} '
               f'Loss: {loss :6.2f}'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # basic environment settings
    parser.add_argument('--name', default='default')
    parser.add_argument('--env', default='gridworld')
    parser.add_argument('--task', default='default')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env_size', type=int, default=20)
    # parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--max_episodes', type=int, default=1000)

    # policy settings
    parser.add_argument('--policy', type=str, default='deep_q')
    parser.add_argument('--policy_update', type=str, default='ddqn')
    parser.add_argument('--policy_lr', type=float, default=1e-3)
    parser.add_argument('--policy_temperature', type=float, default=3e-1)
    parser.add_argument('--policy_test_temperature', type=float, default=1e-1)
    # parser.add_argument('--policy_updates_per_step', type=int, default=1)

    args = parser.parse_args()

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

    main(args)