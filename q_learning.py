import copy
import math
import time
import numpy as np
from typing import Any

import jax
from jax import numpy as jnp, random, jit, lax, profiler

import flax
from flax import nn, optim, struct

import gridworld
import replay_buffer


@struct.dataclass
class QLearnerState():
    optimizer: Any
    rng: jnp.ndarray = random.PRNGKey(0)

    @property
    def model(self):
        return self.optimizer.target


def step_rng(q_state: QLearnerState):
    next_rng = random.split(q_state.rng, 1)[0]
    return q_state.replace(rng=next_rng)


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
grad_loss_fn = jax.grad(loss_fn)  # noqa: E305


def init_fn(seed, state_shape, action_shape):
    rng = random.PRNGKey(seed)
    q_net = DenseQNetwork.partial(hidden_layers=2,
                                  hidden_dim=512)
    _, initial_params = q_net.init_by_shape(
        rng, [(state_shape, jnp.float32), (action_shape, jnp.float32)])
    rng = random.split(rng, 1)[0]
    initial_model = nn.Model(q_net, initial_params)
    q_opt = optim.Adam(1e-3).create(initial_model)
    return QLearnerState(q_opt, rng)


@jax.jit
def train_step(q_state: QLearnerState, states, actions, targets):
    # batch should be of form (states, actions, targets)
    grad = grad_loss_fn(q_state.model, states, actions, targets)
    q_state = q_state.replace(optimizer=q_state.optimizer.apply_gradient(grad))
    return q_state


@jax.partial(jax.profiler.trace_function, name="bellman_train_step")
@jax.jit
def bellman_train_step(q_state: QLearnerState,
                       targetq_state: QLearnerState,
                       transitions):
    # transitions should be of form (states, actions, next_states, rewards)
    targetq_preds = targetq_state.model(transitions[0], transitions[1])
    q_targets = transitions[3] + 0.99 * targetq_preds
    return train_step(q_state, transitions[0], transitions[1], q_targets)


def predict(q_state: QLearnerState, states, actions):
    return q_state.model(states, actions)


@jax.partial(jax.profiler.trace_function, name="sample_action")
@jax.jit
def sample_action(q_state: QLearnerState, rng, state, actions):
    values = predict(q_state,
                     jnp.repeat(state.reshape(1, *state.shape),
                                actions.shape[0], axis=0),
                     actions.reshape(4, 1))
    values = values.reshape(-1)
    boltzmann = True
    if boltzmann:
        action = sample_boltzmann(rng, values, actions)
    else:
        action = sample_egreedy(rng, values, actions)
    return action
sample_action_n = jax.vmap(sample_action,  # noqa: E305
                           in_axes=(0, None, None, None))


def sample_boltzmann(rng, values, actions, temperature=1):
    boltzmann_probs = nn.softmax(values / temperature)
    sampled_index = random.categorical(rng, boltzmann_probs)
    action = actions[sampled_index]
    return action
sample_boltzmann_n = jax.vmap(sample_boltzmann,  # noqa: E305
                              in_axes=(0, None, None))


def sample_egreedy(rng, values, actions, epsilon=0.05):
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
                            in_axes=(0, None, None))


if __name__ == '__main__':
    rng = random.PRNGKey(0)
    env = gridworld.new(5)
    state_shape = (2, env.size)
    action_shape = (1,)
    batch_size = 128
    max_steps = 100

    q_state = init_fn(0, (128, *state_shape), (128, *action_shape))
    targetq_state = q_state
    replay = replay_buffer.Replay(state_shape, action_shape)

    # @jax.jit
    # @jax.profiler.trace_function
    def full_step(q_state, rng, env):
        s = gridworld.render(env)
        a = sample_action(q_state, rng, s, env.actions)
        # with profiler.TraceContext("env step"):
        env, sp, r = gridworld.step(env, int(a))
        # with profiler.TraceContext("replay append"):
        replay.append(s, a, sp, r)

        if len(replay) > batch_size:
            # with profiler.TraceContext("replay sample"):
            transitions = replay.sample(batch_size)
            # with profiler.TraceContext("bellman step"):
            q_state = bellman_train_step(q_state, targetq_state, transitions)
        return q_state, env, r

    # @jax.profiler.trace_function
    def run_episode(rngs, q_state, env):
        env = gridworld.reset(env)
        score = 0
        for i in range(max_steps):
            q_state, env, r = full_step(q_state, rngs[i], env)
            score += r
        return q_state, env, score

    for episode in range(200):
        rngs = random.split(rng, max_steps + 1)
        rng = rngs[0]
        q_state, env, score = run_episode(rngs[1:], q_state, env)
        print(f"Episode {episode}, Score {score}")
        if episode % 1 == 0:
            targetq_state = q_state

    profiler.save_device_memory_profile("memory.prof")
