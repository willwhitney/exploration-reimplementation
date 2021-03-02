import numpy as np
import math
import torch
from typing import Any
import jax
from jax import numpy as jnp

from torch.utils import dlpack as tdlpack
from jax import dlpack as jdlpack

from flax import struct
from pykeops.torch import LazyTensor

import jax_specs
import utils


@struct.dataclass
class DensityState:
    observations: Any
    weights: Any
    device: Any
    state_rescale: jnp.ndarray
    action_rescale: jnp.ndarray
    state_shift: jnp.ndarray
    action_shift: jnp.ndarray
    max_obs: int = 100000
    tolerance: float = 0.95
    total: int = 0
    next_slot: int = 0


def new(observation_spec, action_spec, max_obs=100000,
        state_scale=1, action_scale=1, tolerance=0.95,
        **kwargs):
    flat_ospec = utils.flatten_observation_spec(observation_spec)
    j_flat_ospec = jax_specs.convert_dm_spec(flat_ospec)
    j_aspec = jax_specs.convert_dm_spec(action_spec)

    state_rescale = state_scale * (j_flat_ospec.maximum - j_flat_ospec.minimum)
    action_rescale = action_scale * (j_aspec.maximum - j_aspec.minimum)
    state_shift = j_flat_ospec.minimum
    action_shift = j_aspec.minimum

    key_dim = j_flat_ospec.shape[0] + j_aspec.shape[0]

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    # initialize this to some reasonable size
    # starting_size = 65536
    starting_size = 1024
    observations = torch.zeros((starting_size, key_dim))
    observations = observations.type(torch.float32).to(device)
    weights = torch.zeros((starting_size,))
    weights = weights.type(torch.float32).to(device)

    return DensityState(observations, weights, device,
                        state_rescale, action_rescale,
                        state_shift, action_shift,
                        max_obs=max_obs, tolerance=tolerance)


@jax.profiler.trace_function
def update_batch(density_state: DensityState, states, actions):
    # increase the size of weights vector if needed
    observations = density_state.observations
    weights = density_state.weights
    needed_size = density_state.next_slot + states.shape[0]
    while ((needed_size >= observations.shape[0]) and
           (observations.shape[0] < density_state.max_obs)):
        observations, weights = _grow_observations(observations, weights,
                                                   density_state.max_obs)
    density_state = density_state.replace(observations=observations,
                                          weights=weights)

    # compute which states are new and which weights to update
    keys = _make_key_batch(density_state, states, actions)
    new_keys, weight_updates = _compute_updates(density_state, keys)

    # add all the new observations to the index
    if len(new_keys) > 0:
        density_state = _add_observations(density_state, new_keys)

    # update weights
    if weight_updates.sum() > 0:
        weights = density_state.weights + weight_updates.to(density_state.device)
        density_state = density_state.replace(weights=weights)

    return density_state


@jax.profiler.trace_function
def _compute_updates(density_state: DensityState, keys):
    if density_state.total <= 0:
        weight_update = torch.zeros_like(density_state.weights)
        return keys, weight_update

    obs = density_state.observations

    x_o = LazyTensor( obs[:, None, :] )  # obs_size x 1 x dim
    x_q = LazyTensor( keys[None, :, :] )  # 1 x batch_size x dim

    D_oq = ((x_o - x_q)**2).sum(dim=2)  # obs_size x batch_size
    K_oq = (-0.5 * D_oq).exp()

    # want:
    #   (1) the keys that have no close neighbors,
    #   (2) the close neighbors & distances of the others

    mins, inds = (-K_oq).Kmin_argKmin(16, dim=1)
    sims_per_neighbor = -mins.cpu().numpy()
    new_keys = []
    weight_updates = torch.zeros((obs.shape[0],))

    for (key, sims, ind) in zip(keys, sims_per_neighbor, inds):
        similar_mask = sims > density_state.tolerance
        n_similar_obs = similar_mask.sum()

        if n_similar_obs >= 1:
            similar_meta_indices = np.flatnonzero(similar_mask)
            similar_indices = ind[similar_meta_indices]
            weight_updates[similar_indices] += 1 / n_similar_obs
        else:
            new_keys.append(key)

    if len(new_keys) > 0:
        new_keys = torch.stack(new_keys)
    return new_keys, weight_updates


@jax.profiler.trace_function
def _add_observations(density_state: DensityState, keys):
    bsize = keys.shape[0]
    next_slot = density_state.next_slot
    observations = density_state.observations
    weights = density_state.weights

    if density_state.total == density_state.max_obs:
        indices = torch.randint(low=0, high=int(density_state.max_obs - 1),
                                size=(bsize,))
    else:
        indices = torch.arange(next_slot, next_slot + bsize)
        indices = indices % density_state.max_obs

    # update the observations
    observations[indices] = keys
    weights[indices] = 1
    total = min(density_state.total + bsize, observations.shape[0])
    next_slot = (next_slot + bsize) % observations.shape[0]
    return density_state.replace(observations=observations, weights=weights,
                                 total=total, next_slot=next_slot)


@jax.profiler.trace_function
def _grow_observations(observations, weights, max_size):
    current_size = observations.shape[0]
    print(f"Growing KDE observations from {current_size}.")

    observations = torch.cat([observations, torch.zeros_like(observations)],
                             dim=0)
    weights = torch.cat([weights, torch.zeros_like(weights)],
                        dim=0)
    return observations, weights


@jax.profiler.trace_function
def get_count(density_state: DensityState, state, action):
    states = np.expand_dims(state, axis=0)
    actions = np.expand_dims(action, axis=0)
    return get_count_batch(density_state, states, actions)[0]


@jax.profiler.trace_function
def get_count_batch(density_state: DensityState, states, actions):
    # prevent the index from segfaulting if queried when empty
    if density_state.total <= 0:
        return np.zeros((states.shape[0],))

    obs = density_state.observations
    weights = density_state.weights
    keys = _make_key_batch(density_state, states, actions)

    x_o = LazyTensor( obs[:, None, :] )  # obs_size x 1 x dim
    x_q = LazyTensor( keys[None, :, :] )  # 1 x batch_size x dim
    x_w = LazyTensor( weights[:, None], axis=0 )  # obs_size x 1

    D_oq = ((x_o - x_q)**2).sum(dim=2)  # obs_size x batch_size
    K_oq = (-0.5 * D_oq).exp()
    C_oq = x_w * K_oq  # multiply the row for each obs by its weight
    counts = C_oq.sum(dim=0)  # batch_size
    return counts.cpu().reshape(-1).numpy()


@jax.profiler.trace_function
@jax.partial(jax.jit, backend='cpu')
def _make_key_jax(state_rescale, action_rescale, state_shift, action_shift,
                  s, a):
    flat_s = utils.flatten_observation(s)
    flat_a = jnp.array(a).reshape((-1,))
    normalized_s = (flat_s - state_shift) / state_rescale
    normalized_a = (flat_a - action_shift) / action_rescale
    return jnp.concatenate([normalized_s, normalized_a], axis=0).astype(jnp.float32)
_make_key_jax_batch = jax.vmap(_make_key_jax, in_axes=(None, None, None, None,
                                                       0, 0))


def j_to_t(x):
    return tdlpack.from_dlpack(jdlpack.to_dlpack(x))


def _make_key_batch(density_state: DensityState, s, a):
    j_key = _make_key_jax_batch(density_state.state_rescale,
                                density_state.action_rescale,
                                density_state.state_shift,
                                density_state.action_shift,
                                s, a)
    return j_to_t(j_key).to(density_state.device)


if __name__ == "__main__":
    from dm_control import suite
    from observation_domains import DOMAINS
    import jax_specs
    import point

    env_name = 'point'
    task_name = 'velocity'
    env = suite.load(env_name, task_name)
    ospec = DOMAINS[env_name][task_name]

    aspec = env.action_spec()
    j_aspec = jax_specs.convert_dm_spec(aspec)
    j_ospec = jax_specs.convert_dm_spec(ospec)
    density_state = new(ospec, aspec, state_scale=0.01, action_scale=1)

    timestep = env.reset()
    state = utils.flatten_observation(timestep.observation)
    actions = utils.sample_uniform_actions(j_aspec, jax.random.PRNGKey(0), 1)
    action = actions[0]


    # ---------- sanity checking counts --------------------
    timestep2 = env.step(jnp.ones(aspec.shape))
    state2 = utils.flatten_observation(timestep2.observation)

    print("S1 count:", get_count(density_state, state, action))

    print("S2 count:", get_count(density_state, state2, action))
    density_state_updated = update_batch(density_state,
                                         jnp.expand_dims(state2, axis=0),
                                         jnp.expand_dims(action, axis=0))
    print("S2 count after self update:", get_count(density_state_updated,
                                                state2, action))

    print("Batch of counts:", get_count_batch(density_state_updated,
                                              jnp.stack([state, state2]),
                                              jnp.stack([action, action])))
