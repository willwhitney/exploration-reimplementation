import jax
from jax import numpy as jnp, random, profiler  # noqa: F401

from flax import nn, optim

import q_learning
from deep_q_functions import train_step, bellman_train_step  # noqa: F401


class DenseQNetwork(nn.Module):
    def apply(self, s, a, hidden_layers, hidden_dim):
        env_size = s.shape[2]

        # should be bsize x 2
        locs = jnp.argmax(s, axis=-1)
        xs, ys = locs[:, 0], locs[:, 1]
        new_s = xs * env_size + ys
        s = jax.nn.one_hot(new_s, env_size * env_size)

        s = jnp.reshape(s, (s.shape[0], -1))
        a = jnp.reshape(a, (a.shape[0], -1))
        x = jnp.concatenate([s, a], axis=1)
        for layer in range(hidden_layers):
            x = nn.Dense(x, hidden_dim, name=f'fc{layer}')
            x = nn.relu(x)
        q = nn.Dense(x, 1, name=f'fc{hidden_layers}')
        return q


def init_fn(seed, state_shape, action_shape, discount, **kwargs):
    rng = random.PRNGKey(seed)
    q_net = DenseQNetwork.partial(hidden_layers=2,
                                  hidden_dim=512)
    _, initial_params = q_net.init_by_shape(
        rng, [(state_shape, jnp.float32), (action_shape, jnp.float32)])
    rng = random.split(rng, 1)[0]
    initial_model = nn.Model(q_net, initial_params)
    q_opt = optim.Adam(1e-3).create(initial_model)
    return q_learning.QLearnerState(q_opt, discount)


# def loss_fn(model, states, actions, targets):
#     preds = model(states, actions)
#     loss = jnp.mean((preds - targets)**2)
#     return loss
# grad_loss_fn = jax.grad(loss_fn)  # noqa: E305


# @jax.jit
# def train_step(q_state: q_learning.QLearnerState, states, actions, targets):
#     grad = grad_loss_fn(q_state.model, states, actions, targets)
#     q_state = q_state.replace(optimizer=q_state.optimizer.apply_gradient(grad))
#     return q_state


# @jax.partial(jax.profiler.trace_function, name="bellman_train_step")
# @jax.jit
# def bellman_train_step(q_state: q_learning.QLearnerState,
#                        targetq_state: q_learning.QLearnerState,
#                        transitions,
#                        candidate_actions):
#     # transitions should be of form (states, actions, next_states, rewards)
#     # candidate_actions should be of form bsize x n_cands x *action_dim
#     bsize, n_candidates = candidate_actions.shape[:2]

#     # compute max_a' Q_t (s', a')
#     targetq_preds = q_learning.predict_action_values_batch(
#         targetq_state, transitions[2], candidate_actions)
#     targetq_preds = targetq_preds.max(axis=-1).reshape(transitions[3].shape)

#     # compute targets and update
#     q_targets = transitions[3] + q_state.discount * targetq_preds
#     return train_step(
#         q_state, transitions[0], transitions[1], q_targets)
