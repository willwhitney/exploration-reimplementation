import jax
from jax import numpy as jnp, random

from flax import nn, optim

import q_learning


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


def init_fn(seed, state_shape, action_shape, discount, **kwargs):
    rng = random.PRNGKey(seed)
    q_net = TabularQ.partial(env_size=kwargs['env_size'])
    _, initial_params = q_net.init_by_shape(
        rng, [(state_shape, jnp.float32), (action_shape, jnp.float32)])
    rng = random.split(rng, 1)[0]
    initial_model = nn.Model(q_net, initial_params)
    q_opt = optim.Adam(1e-2).create(initial_model)
    return q_learning.QLearnerState(q_opt, discount)


@jax.jit
def train_step(q_state: q_learning.QLearnerState, states, actions, targets):
    state_coords = states.argmax(axis=-1)
    for coord, a, t in zip(state_coords, actions, targets):
        x, y = coord
        a = a.astype(jnp.int16)[0]
        t = jnp.array(t).reshape(tuple())
        new_table = jax.ops.index_update(q_state.model.params['table'],
                                         jax.ops.index[x, y, a],
                                         t)
        new_model = q_state.model.replace(params={'table': new_table})
        new_optimizer = q_state.optimizer.replace(target=new_model)
        q_state = q_state.replace(optimizer=new_optimizer)
    return q_state


@jax.jit
def bellman_train_step(q_state: q_learning.QLearnerState,
                       targetq_state: q_learning.QLearnerState,
                       transitions,
                       candidate_actions):
    value_preds = q_learning.predict_action_values_batch(
        q_state, transitions[2], candidate_actions)
    value_preds = value_preds.max(axis=-1).reshape(transitions[3].shape)
    q_targets = transitions[3] + q_state.discount * value_preds

    states = transitions[0]
    actions = transitions[1].astype(jnp.int16)
    return train_step(q_state, states, actions, q_targets)
