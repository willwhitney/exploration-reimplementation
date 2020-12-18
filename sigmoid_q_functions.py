import jax
from jax import numpy as jnp, random, profiler  # noqa: F401

from flax import nn, optim

import q_learning
import utils
import jax_specs

from deep_q_functions import *


class SigmoidQNetwork(nn.Module):
    def apply(self, s, a, hidden_layers, hidden_dim, flat_ospec, flat_aspec,
              max_value):
        s = jnp.reshape(s, (s.shape[0], -1))
        a = jnp.reshape(a, (a.shape[0], -1))
        s = utils.normalize(s, flat_ospec) * 2 - 1
        # s = fourier_code(s)
        a = utils.normalize(a, flat_aspec) * 2 - 1
        x = jnp.concatenate([s, a], axis=1)
        for layer in range(hidden_layers):
            x = nn.Dense(x, hidden_dim, name=f'fc{layer}')
            x = nn.relu(x)
        q = nn.Dense(x, 1, name=f'fc{hidden_layers}')
        q = (max_value + 2) * nn.sigmoid(q / 10) - 1
        return q

def init_fn(seed, state_spec, action_spec, discount, max_value=100, lr=1e-2,
            **kwargs):
    flat_state_spec = utils.flatten_observation_spec(state_spec)
    j_state_spec = jax_specs.convert_dm_spec(flat_state_spec)
    j_action_spec = jax_specs.convert_dm_spec(action_spec)
    state_shape = flat_state_spec.shape
    action_shape = action_spec.shape
    rng = random.PRNGKey(seed)
    q_net = SigmoidQNetwork.partial(hidden_layers=2,
                                  hidden_dim=512,
                                  flat_ospec=j_state_spec,
                                  flat_aspec=j_action_spec,
                                  max_value=max_value)
    _, initial_params = q_net.init_by_shape(
        rng, [((128, *state_shape), jnp.float32),
              ((128, *action_shape), jnp.float32)])
    rng = random.split(rng, 1)[0]
    initial_model = nn.Model(q_net, initial_params)
    q_opt = optim.Adam(lr).create(initial_model)
    return q_learning.QLearnerState(q_opt, discount)

