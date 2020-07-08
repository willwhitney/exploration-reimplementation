import math
import numpy as np

import jax
from jax import numpy as jnp, random, jit, lax

import flax
from flax import nn, optim


class ExplorationPolicy:
    def __init__(self, novelty_q, policy,
                 n_action_samples=64, temperature=1, seed=0):
        self.novelty_q = novelty_q
        self.policy = policy
        self.n_action_samples = n_action_samples
        self.temperature = temperature
        self.rng = random.PRNGKey(seed)

        def _sample_reweighted(novelty_q_state, rng, s, actions):
            q_values = novelty_q(novelty_q_state, s, actions)
            logits = q_values / temperature
            action_index = random.categorical(rng, logits)
            return actions[action_index]

        def _sample_action(novelty_q_state, policy_state, rng, s):
            actions = self.policy(policy_state, s, n=self.n_action_samples)
            return _sample_reweighted(novelty_q_state, rng, s, actions)

        self._sample_actions = jax.vmap(_sample_action,
                                        in_axes=(None, None, 0, 0))

    def sample_actions(self, novelty_q_state, policy_state, states):
        new_rngs = random.split(self.rng, states.shape[0] + 1)
        self.rng = new_rngs[0]
        actions = self._sample_actions(
            novelty_q_state, policy_state, new_rngs[1:], states)
        return actions


class DensityEstimator:
    def log_p(self, states, actions):
        raise NotImplementedError

    def update(self, states, actions):
        raise NotImplementedError


class DenseQNetwork(nn.Module):
    def apply(self, s, a, hidden_layers, hidden_dim):
        s = jnp.reshape(s, (s.shape[0], -1))
        x = jnp.concatenate([s, a], axis=1)
        for layer in range(hidden_layers):
            x = nn.Dense(x, hidden_dim, name=f'fc{layer}')
            x = nn.relu(x)
        q = nn.Dense(1, hidden_dim, name=f'fc{hidden_layers}')
        return q


def batch_to_jax(batch):
    return (jnp.array(batch[0]), jnp.array(batch[1]))


def loss_fn(model, batch):
    preds = model(batch[0])
    loss = jnp.mean((preds - batch[1])**2)
    return loss


grad_loss_fn = jax.grad(loss_fn)


def init_fn(seed):
    rng = random.PRNGKey(seed)
    q_net = DenseQNetwork.partial(hidden_layers=2,
                                  hidden_dim=512)
    _, initial_params = q_net.init_by_shape(rng, [(128, 784)])
    initial_model = nn.Model(q_net, initial_params)
    optimizer = optim.Adam(1e-3).create(initial_model)
    return optimizer


@jax.jit
def train_step(optimizer, batch):
    batch = batch_to_jax(batch)
    grad = grad_loss_fn(optimizer.target, batch)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer


def eval_fn(optimizer, batch):
    batch = batch_to_jax(batch)
    return loss_fn(optimizer.target, batch)


if __name__ == '__main__':
    pass
