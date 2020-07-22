# import copy
# import math
# import time
import numpy as np
from typing import Any
import matplotlib.pyplot as plt

import jax
from jax import numpy as jnp, random, lax, profiler

# import flax
from flax import nn, optim, struct

import gridworld
import replay_buffer
import utils


@struct.dataclass
class QLearnerState():
    optimizer: Any
    # rng: jnp.ndarray = random.PRNGKey(0)

    @property
    def model(self):
        return self.optimizer.target


# def step_rng(q_state: QLearnerState):
#     next_rng = random.split(q_state.rng, 1)[0]
#     return q_state.replace(rng=next_rng)


class DenseQNetwork(nn.Module):
    def apply(self, s, a, hidden_layers, hidden_dim):
        s = jnp.reshape(s, (s.shape[0], -1))
        x = jnp.concatenate([s, a], axis=1)
        for layer in range(hidden_layers):
            x = nn.Dense(x, hidden_dim, name=f'fc{layer}')
            x = nn.relu(x)
        q = nn.Dense(x, 1, name=f'fc{hidden_layers}')
        return q


class TabularQ(nn.Module):
    def apply(self, s, a, env_size):
        def init(key, shape, dtype=jnp.float32):
            return jnp.full(shape, 0, dtype=dtype)

        table = self.param('table', (env_size, env_size, 4), init)

        def lookup(table, s, a):
            x, y = s.argmax(axis=1)
            a = a.astype(jnp.int16)
            return table[x][y][a]
        batched_lookup = jax.vmap(lookup, in_axes=(None, 0, 0))
        return batched_lookup(table, s, a)


def init_fn(seed, state_shape, action_shape, tabular=False, env_size=None):
    rng = random.PRNGKey(seed)
    if tabular:
        q_net = TabularQ.partial(env_size=env_size)
    else:
        q_net = DenseQNetwork.partial(hidden_layers=2,
                                      hidden_dim=512)
    _, initial_params = q_net.init_by_shape(
        rng, [(state_shape, jnp.float32), (action_shape, jnp.float32)])
    rng = random.split(rng, 1)[0]
    initial_model = nn.Model(q_net, initial_params)
    q_opt = optim.Adam(1e-2).create(initial_model)
    return QLearnerState(q_opt)


def loss_fn(model, states, actions, targets):
    preds = model(states, actions)
    loss = jnp.mean((preds - targets)**2)
    return loss
grad_loss_fn = jax.grad(loss_fn)  # noqa: E305


@jax.jit
def train_step(q_state: QLearnerState, states, actions, targets):
    grad = grad_loss_fn(q_state.model, states, actions, targets)
    q_state = q_state.replace(optimizer=q_state.optimizer.apply_gradient(grad))
    return q_state


@jax.partial(jax.profiler.trace_function, name="bellman_train_step")
@jax.jit
def bellman_train_step(q_state: QLearnerState,
                       targetq_state: QLearnerState,
                       transitions,
                       candidate_actions,
                       discount=0.99):
    # transitions should be of form (states, actions, next_states, rewards)
    # candidate_actions should be of form bsize x n_cands x *action_dim
    bsize, n_candidates = candidate_actions.shape[:2]

    # compute max_a' Q_t (s', a')
    targetq_preds = predict_action_values_batch(
        targetq_state, transitions[2], candidate_actions)
    targetq_preds = targetq_preds.max(axis=-1).reshape(transitions[3].shape)
    # import ipdb; ipdb.set_trace()

    # compute targets and update
    q_targets = transitions[3] + discount * targetq_preds
    return train_step(q_state, transitions[0], transitions[1], q_targets)


@jax.jit
def tabular_train_step(q_state: QLearnerState,
                       targetq_state: QLearnerState,
                       transitions,
                       candidate_actions,
                       discount=0.99):
    value_preds = predict_action_values_batch(
        q_state, transitions[2], candidate_actions)
    value_preds = value_preds.max(axis=-1).reshape(transitions[3].shape)
    q_targets = transitions[3] + discount * value_preds

    state_coords = transitions[0].argmax(axis=-1)
    a_s = transitions[1].astype(jnp.int16)
    for coord, a, t in zip(state_coords, a_s, q_targets):
        x, y = coord
        a = a[0]
        t = t[0]
        new_table = jax.ops.index_update(q_state.model.params['table'],
                                         jax.ops.index[x, y, a],
                                         t)
        new_model = q_state.model.replace(params={'table': new_table})
        new_optimizer = q_state.optimizer.replace(target=new_model)
        q_state = q_state.replace(optimizer=new_optimizer)
    return q_state


def predict_value(q_state: QLearnerState, states, actions):
    """Takes a batch of states and actions and returns the predicted value.
    Returns:
    - an ndarray of dimension `states.shape[0] == actions.shape[0]` containing
        Q_{\theta}(s, a) for each `s` and `a` in the batch.
    """
    return q_state.model(states, actions)


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
                                           in_axes=(None, 0, 0, None, None))

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


def location_value(q_state: QLearnerState, env: gridworld.GridWorld, location):
    env = env.replace(agent=jnp.array(location))
    s = env.render(env.agent)
    values = predict_action_values(q_state, s, env.actions)
    return jnp.max(values)
location_value_batch = jax.vmap(location_value,  # noqa: E305
                                in_axes=(None, None, 0))


def render_value_map(q_state: QLearnerState, env: gridworld.GridWorld):
    locations = gridworld.all_coords(env.size)
    location_values = location_value_batch(q_state, env, locations)
    value_map = np.zeros((env.size, env.size))
    for location, value in zip(locations, location_values):
        # import ipdb; ipdb.set_trace()
        value_map[location[0], location[1]] = value
    return value_map


def display_value_map(q_state, env):
    value_map = render_value_map(q_state, env)
    fig, ax = plt.subplots()
    img = ax.imshow(value_map)
    fig.colorbar(img, ax=ax)
    fig.show()
    plt.close(fig)


def main(args):
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

    q_state = init_fn(0, (128, *state_shape), (128, *action_shape),
                      tabular=args.tabular, env_size=args.env_size)
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
                if args.tabular:
                    q_state = tabular_train_step(q_state, targetq_state,
                                                 transitions, candidate_actions)
                else:
                    q_state = bellman_train_step(q_state, targetq_state,
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
        if episode % 50 == 0:
            print("\nQ network values")
            display_value_map(q_state, env)
            # import ipdb; ipdb.set_trace()
            # print("\nTarget Q network values")
            # display_value_map(targetq_state, env)
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
