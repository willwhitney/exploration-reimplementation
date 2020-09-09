import jax
from jax import numpy as jnp, random, profiler  # noqa: F401

from flax import nn, optim

import q_learning


class DenseQNetwork(nn.Module):
    def apply(self, s, a, hidden_layers, hidden_dim):
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
    q_opt = optim.Adam(1e-2).create(initial_model)
    return q_learning.QLearnerState(q_opt, discount)


def single_loss_fn(model, state, action, target):
    pred = model(jnp.expand_dims(state, axis=0),
                 jnp.expand_dims(action, axis=0)).reshape(())
    loss = (target - pred)**2
    return loss.mean()
single_loss_and_grad = jax.value_and_grad(single_loss_fn)  # noqa: E305
batch_loss_and_grad = jax.vmap(single_loss_and_grad, in_axes=(None, 0, 0, 0))


def loss_fn(model, states, actions, targets):
    preds = model(states, actions)
    loss = jnp.mean((preds - targets)**2)
    return loss
grad_loss_fn = jax.grad(loss_fn)  # noqa: E305
loss_and_grad_fn = jax.value_and_grad(loss_fn)


@jax.jit
def losses_and_grad_fn(model, states, actions, targets):
    losses, grads = batch_loss_and_grad(model, states, actions, targets)
    grad = jax.tree_map(jax.partial(jnp.mean, axis=0), grads)
    return losses, grad


@jax.jit
def train_step(q_state: q_learning.QLearnerState, states, actions, targets):
    # loss, grad = value_and_grad_loss_fn(q_state.model, states, actions, targets)
    losses, grad = losses_and_grad_fn(q_state.model, states, actions, targets)
    q_state = q_state.replace(optimizer=q_state.optimizer.apply_gradient(grad))
    return q_state, losses


@jax.partial(jax.profiler.trace_function, name="bellman_train_step")
@jax.jit
def bellman_train_step(q_state: q_learning.QLearnerState,
                       targetq_state: q_learning.QLearnerState,
                       transitions,
                       candidate_actions):
    # transitions should be of form (states, actions, next_states, rewards)
    # candidate_actions should be of form bsize x n_cands x *action_dim
    bsize, n_candidates = candidate_actions.shape[:2]

    # compute max_a' Q_t (s', a')
    targetq_preds = q_learning.predict_action_values_batch(
        targetq_state, transitions[2], candidate_actions)
    targetq_preds = targetq_preds.max(axis=-1).reshape(transitions[3].shape)

    # compute targets and update
    q_targets = transitions[3] + q_state.discount * targetq_preds
    return train_step(
        q_state, transitions[0], transitions[1], q_targets)
