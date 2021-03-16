import numpy as np
import typing
import matplotlib.pyplot as plt

from dm_control import suite

import jax
from jax import numpy as jnp, random, lax, profiler

from flax import nn, struct

from environments import dmcontrol_gridworld
import utils
import replay_buffer
from environments.observation_domains import DOMAINS


@struct.dataclass
class QLearnerState():
    optimizer: typing.Any
    discount: float

    @property
    def model(self):
        return self.optimizer.target


@jax.jit
def predict_value(q_state: QLearnerState, states, actions):
    """Takes a batch of states and actions and returns the predicted value.
    Returns:
    - an ndarray of dimension `states.shape[0] == actions.shape[0]` containing
        Q_{\theta}(s, a) for each `s` and `a` in the batch.
    """
    return q_state.model(states, actions)


@jax.jit
def predict_action_values(q_state, state, actions):
    """Predict the value of each of the given `actions` in `state`.
    Returns:
    - an ndarray of dimension `actions.shape[0]` containing the value of each
        action
    """
    n_candidates = actions.shape[0]
    values = predict_value(q_state,
                           jnp.repeat(jnp.expand_dims(state, 0),
                                      n_candidates, axis=0),
                           actions.reshape(n_candidates, -1))
    return values.reshape(-1)
# takes a batch of states and a (batch x n_candidates) of actions
# -> a (batch x n_candidates) of values
predict_action_values_batch = jax.vmap(predict_action_values,  # noqa: E305
                                       in_axes=(None, 0, 0))


@jax.jit
def sample_action_egreedy(q_state: QLearnerState, rng, state, actions, epsilon):
    values = predict_action_values(q_state, state, actions)
    action = sample_egreedy(rng, values, actions, epsilon=epsilon)
    return action, values
sample_action_egreedy_n = jax.vmap(sample_action_egreedy,  # noqa: E305
                                   in_axes=(None, 0, None, None, None))
sample_action_egreedy_n_batch = jax.vmap(sample_action_egreedy_n,
                                         in_axes=(None, 0, 0, 0, None))


@jax.jit
def sample_action_boltzmann(q_state: QLearnerState, rng, state, actions, temp):
    values = predict_action_values(q_state, state, actions)
    action, entropy = sample_boltzmann(rng, values, actions, temp)
    return action, values, entropy
sample_action_boltzmann_n = jax.vmap(sample_action_boltzmann,  # noqa: E305
                                     in_axes=(None, 0, None, None, None))
sample_action_boltzmann_n_batch = jax.vmap(sample_action_boltzmann_n,
                                           in_axes=(None, 0, 0, 0, None))
sample_action_boltzmann_batch = jax.vmap(sample_action_boltzmann,
                                         in_axes=(None, 0, 0, 0, None))
sample_action_boltzmann_batch_n = jax.vmap(sample_action_boltzmann,
                                           in_axes=(None, 0, None, None, None))


@jax.jit
def sample_boltzmann(rng, values, actions, temp=1):
    boltzmann_logits = values / temp

    probs = nn.softmax(boltzmann_logits)
    entropy = - jnp.dot(probs, jnp.log(probs + 1e-8))

    # jax categorical is actually categorical(softmax(logits))
    action_index = random.categorical(rng, boltzmann_logits)
    action = actions[action_index]
    return action, entropy
sample_boltzmann_n = jax.vmap(sample_boltzmann,  # noqa: E305
                              in_axes=(0, None, None, None))


@jax.jit
def sample_egreedy(rng, values, actions, epsilon=0.5):
    explore = random.bernoulli(rng, p=epsilon)
    rng = random.split(rng, 1)[0]
    random_index = random.randint(rng, (1,), 0, actions.shape[0])[0]
    max_index = jnp.argmax(values, axis=0)
    action = lax.cond(explore,
                      lambda _: actions[random_index],
                      lambda _: actions[max_index],
                      None)
    return action
sample_egreedy_n = jax.vmap(sample_egreedy,  # noqa: E305
                            in_axes=(0, None, None, None))



# ---------- Utilities for gridworlds --------------
def display_state(q_state: QLearnerState, env: dmcontrol_gridworld.GridWorld,
                  rendering='disk', savepath=None):
    q_map = dmcontrol_gridworld.render_function(
        jax.partial(predict_value, q_state),
        env, reduction=jnp.max)
    subfigs = [
        (q_map, "State values"),
    ]
    utils.display_subfigures(subfigs, rendering, savepath)
# --------------------------------------------------


def main(args):
    rng = random.PRNGKey(0)

    if args.env == 'gridworld':
        env = dmcontrol_gridworld.GridWorld(args.env_size, 100)
        observation_spec = env.observation_spec()
    else:
        env = suite.load(args.env, args.task)
        observation_spec = DOMAINS[args.env][args.task]

    action_spec = env.action_spec()

    state_shape = utils.flatten_spec_shape(observation_spec)
    action_shape = action_spec.shape

    batch_size = 128

    if args.tabular:
        import tabular_q_functions as q_functions
    else:
        import deep_q_functions as q_functions
        # import onehot_deep_q_functions as q_functions

    if args.boltzmann:
        sample_action = jax.partial(sample_action_boltzmann, temp=1)
    else:
        sample_action = jax.partial(sample_action_egreedy, epsilon=0.5)

    action_proposal = jax.partial(utils.sample_uniform_actions,
                                  action_spec)

    q_state = q_functions.init_fn(0, observation_spec, action_spec,
                                  env_size=args.env_size,
                                  discount=0.99)
    targetq_state = q_state
    replay = replay_buffer.Replay(state_shape, action_shape)
    # candidate_actions = jnp.tile(jnp.expand_dims(env.actions, 0),
    #                              (batch_size, 1))
    n_proposal = 16

    def get_action(q_state, targetq_state, rng, s, train=True):
        rng, action_rng = random.split(rng)
        actions = action_proposal(action_rng, n_proposal)
        if train:
            a, v, h = sample_action(q_state, rng, s, actions)
        else:
            a, v = sample_action_egreedy(q_state, rng, s, actions, 0.01)

        return a, v

        # return q_state, env, r

    # @jax.profiler.trace_function
    def run_episode(rng, q_state, targetq_state, env, train=True):
        timestep = env.reset()
        score = 0
        while not timestep.last():
            rng, step_rng = random.split(rng)
            s = utils.flatten_observation(timestep.observation)
            a, _ = get_action(q_state, targetq_state, step_rng, s, train)
            timestep = env.step(a)

            sp = utils.flatten_observation(timestep.observation)
            r = timestep.reward
            if train:
                replay.append(s, a, sp, r)
                if len(replay) > batch_size:
                    transitions = replay.sample(batch_size)

                    rng, update_rng = random.split(rng)
                    candidate_actions = action_proposal(
                        update_rng, batch_size * n_proposal)
                    candidate_actions = candidate_actions.reshape(
                        (batch_size, n_proposal, -1))
                    q_state, losses = q_functions.bellman_train_step(
                        q_state, targetq_state,
                        transitions, candidate_actions)
            score += r
        return q_state, score

    for episode in range(1000):
        rng, episode_rng = random.split(rng)
        q_state, score = run_episode(
            episode_rng, targetq_state, q_state, env)

        if episode % 1 == 0:
            rng, episode_rng = random.split(rng)
            _, test_score = run_episode(
                episode_rng, targetq_state, q_state, env, train=False)
            print((f"Episode {episode:4d}"
                   f", Train score {score:3.0f}"
                   f", Test score {test_score:3.0f}"))
        if args.env == 'gridworld' and episode % 1 == 0:
            savepath = f"results/q_learning/{args.name}/{episode}.png"
            display_state(q_state, env, rendering='disk', savepath=savepath)
            # display_value_map(q_state, env)
            # import time; time.sleep(5)
        if episode % 1 == 0:
            targetq_state = q_state


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="default")
    parser.add_argument('--env', default='gridworld')
    parser.add_argument('--task', default='default')
    parser.add_argument('--tabular', action='store_true', default=False)
    parser.add_argument('--env_size', type=int, default=5)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--boltzmann', action='store_true', default=False)
    args = parser.parse_args()

    jit = not args.debug
    if jit:
        main(args)
    else:
        with jax.disable_jit():
            main(args)

    # profiler.save_device_memory_profile("memory.prof")
