import numpy as np
import typing
import matplotlib.pyplot as plt

import jax
from jax import numpy as jnp, random, lax, profiler

from flax import nn, struct

import gridworld


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


@jax.jit
def sample_action_boltzmann(q_state: QLearnerState, rng, state, actions, temp):
    values = predict_action_values(q_state, state, actions)
    action = sample_boltzmann(rng, values, actions, temp)
    return action, values
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
    boltzmann_probs = nn.softmax(values / temp)
    sampled_index = random.categorical(rng, boltzmann_probs)
    action = actions[sampled_index]
    return action
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
# def location_value(q_state: QLearnerState, env: gridworld.GridWorld, location):
#     env = env.replace(agent=jnp.array(location))
#     s = env.render(env.agent)
#     values = predict_action_values(q_state, s, env.actions)
#     return jnp.max(values)
# location_value_batch = jax.vmap(location_value,  # noqa: E305
#                                 in_axes=(None, None, 0))


# def render_value_map(q_state: QLearnerState, env: gridworld.GridWorld):
#     locations = gridworld.all_coords(env.size)
#     location_values = location_value_batch(q_state, env, locations)
#     value_map = np.zeros((env.size, env.size))
#     for location, value in zip(locations, location_values):
#         value_map[location[0], location[1]] = value
#     return value_map


def render_value_map(q_state: QLearnerState, env: gridworld.GridWorld):
    value_map = gridworld.render_function(
        jax.partial(predict_value, q_state),
        env, reduction=jnp.max)
    return value_map


def display_value_map(q_state, env):
    value_map = render_value_map(q_state, env)
    fig, ax = plt.subplots()
    img = ax.imshow(value_map)
    fig.colorbar(img, ax=ax)
    ax.set_title("State values")
    fig.show()
    plt.close(fig)
# --------------------------------------------------


def main(args):
    import gridworld
    import replay_buffer

    if args.tabular:
        import tabular_q_functions as q_functions
    else:
        import deep_q_functions as q_functions

    rng = random.PRNGKey(0)
    env = gridworld.new(args.env_size)
    state_shape = (2, env.size)
    action_shape = (1,)
    batch_size = 128
    max_steps = 100

    if args.boltzmann:
        sample_action = jax.partial(sample_action_boltzmann, temp=0.1)
    else:
        sample_action = jax.partial(sample_action_egreedy, epsilon=0.5)

    q_state = q_functions.init_fn(0,
                                  (batch_size, *state_shape),
                                  (batch_size, *action_shape),
                                  env_size=args.env_size)
    targetq_state = q_state
    replay = replay_buffer.Replay(state_shape, action_shape)
    candidate_actions = jnp.tile(jnp.expand_dims(env.actions, 0),
                                 (batch_size, 1))

    def full_step(q_state, targetq_state, rng, env, train=True):
        s = gridworld.render(env)
        if train:
            a, v = sample_action(q_state, rng, s, env.actions)
        else:
            a, v = sample_action_egreedy(q_state, rng, s, env.actions, 0.01)

        env, sp, r = gridworld.step(env, int(a))

        if train:
            replay.append(s, a, sp, r)
            if len(replay) > batch_size:
                transitions = replay.sample(batch_size)
                q_state = q_functions.bellman_train_step(
                    q_state, targetq_state,
                    transitions, candidate_actions)

        return q_state, env, r

    # @jax.profiler.trace_function
    def run_episode(rngs, q_state, targetq_state, env, train=True):
        env = gridworld.reset(env)
        score = 0
        for i in range(max_steps):
            q_state, env, r = full_step(
                q_state, targetq_state, rngs[i], env, train)
            score += r
        return q_state, env, score

    for episode in range(1000):
        rngs = random.split(rng, max_steps + 1)
        rng = rngs[0]
        q_state, env, score = run_episode(
            rngs[1:], targetq_state, q_state, env)

        if episode % 10 == 0:
            rngs = random.split(rng, max_steps + 1)
            rng = rngs[0]
            _, _, test_score = run_episode(
                rngs[1:], targetq_state, q_state, env, train=False)
            print((f"Episode {episode:4d}"
                   f", Train score {score:3d}"
                   f", Test score {test_score:3d}"))
        if episode % 1 == 0:
            display_value_map(q_state, env)
            import time; time.sleep(5)
        if episode % 1 == 0:
            targetq_state = q_state


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
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
