import copy
import math
import numpy as np

import jax
from jax import numpy as jnp, random, jit, lax

import flax
from flax import nn, optim

import gridworld
import replay


class DenseQNetwork(nn.Module):
    def apply(self, s, a, hidden_layers, hidden_dim):
        s = jnp.reshape(s, (s.shape[0], -1))
        x = jnp.concatenate([s, a], axis=1)
        for layer in range(hidden_layers):
            x = nn.Dense(x, hidden_dim, name=f'fc{layer}')
            x = nn.relu(x)
        q = nn.Dense(x, 1, name=f'fc{hidden_layers}')
        return q


def loss_fn(model, states, actions, targets):
    preds = model(states, actions)
    loss = jnp.mean((preds - targets)**2)
    return loss


grad_loss_fn = jax.grad(loss_fn)


def init_fn(seed, state_shape, action_shape):
    rng = random.PRNGKey(seed)
    q_net = DenseQNetwork.partial(hidden_layers=2,
                                  hidden_dim=512)
    # import ipdb; ipdb.set_trace()
    _, initial_params = q_net.init_by_shape(
        rng, [(state_shape, jnp.float32), (action_shape, jnp.float32)])
    initial_model = nn.Model(q_net, initial_params)
    q_opt = optim.Adam(1e-3).create(initial_model)
    return q_opt


@jax.jit
def train_step(q_opt, states, actions, targets):
    # batch should be of form (states, actions, targets)
    grad = grad_loss_fn(q_opt.target, states, actions, targets)
    q_opt = q_opt.apply_gradient(grad)
    return q_opt


@jax.jit
def bellman_train_step(q_opt, targetq_opt, transitions):
    # transitions should be of form (states, actions, next_states, rewards)
    targetq_preds = targetq_opt.target(transitions[0], transitions[1])
    q_targets = transitions[3] + 0.99 * targetq_preds
    return train_step(q_opt, transitions[0], transitions[1], q_targets)


def predict(q_opt, states, actions):
    return q_opt.target(states, actions)


@jax.jit
def choose_action(rng, q_opt, state, actions):
    # import ipdb; ipdb.set_trace()
    values = predict(q_opt,
                     jnp.repeat(state.reshape(1, *state.shape),
                                actions.shape[0], axis=0),
                     actions.reshape(4, 1))
    values = values.reshape(-1)
    boltzmann = False
    if boltzmann:
        boltzmann_probs = nn.softmax(values)
        sampled_index = random.categorical(rng, boltzmann_probs)
        action = actions[sampled_index]
    else:
        explore = random.bernoulli(rng, p=0.05)
        rng = random.split(rng, 1)[0]
        random_index = random.randint(rng, (1,), 0, actions.shape[0])[0]
        max_index = jnp.argmax(values, axis=0)
        action = lax.cond(explore,
                          lambda _: actions[random_index],
                          lambda _: actions[max_index],
                          None)
    return action


if __name__ == '__main__':
    rng = random.PRNGKey(0)
    env = gridworld.new(2)
    state_shape = (2, env.size)
    action_shape = (1,)
    batch_size = 128
    max_steps = 100
    actions = jnp.arange(4)

    q_opt = init_fn(0, (128, *state_shape), (128, *action_shape))
    targetq_opt = q_opt
    buffer = replay.Replay(state_shape, action_shape)

    def full_step(rng, q_opt, env):
        s = gridworld.render(env)
        a = choose_action(rng, q_opt, s, actions)
        env, sp, r = gridworld.step(env, a)
        buffer.append(s, a, sp, r)

        if len(buffer) > batch_size:
            transitions = buffer.sample(batch_size)
            q_opt = bellman_train_step(q_opt, targetq_opt, transitions)
        return q_opt, env, r

    def run_episode(rngs, q_opt, env):
        env = gridworld.reset(env)
        score = 0
        for i in range(max_steps):
            q_opt, env, r = full_step(rngs[i], q_opt, env)
            score += r
        return q_opt, env, score

    for episode in range(1000):
        # with jax.disable_jit():
        rngs = random.split(rng, max_steps + 1)
        rng = rngs[0]
        q_opt, env, score = run_episode(rngs[1:], q_opt, env)
        print(f"Episode {episode}, Score {score}")
        if episode % 1 == 0:
            targetq_opt = q_opt
