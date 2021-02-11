import typing
import numpy as np

import jax
from jax import numpy as jnp, random, lax

import faiss
from flax import struct

import utils


@struct.dataclass
class DensityState:
    kernel_cov: jnp.ndarray
    index: typing.Any
    weights: jnp.ndarray
    max_obs: int = 100000
    scale_factor: float = 1.0
    total: int = 0
    next_slot: int = 0
    k: int = 16


# class KnnKernelCount():
#     def __init__(self, observation_spec, action_spec, max_obs=100000,
#                  state_std_scale=1, action_std_scale=1, **kwargs):
#         flat_ospec = utils.flatten_observation_spec(observation_spec)
#         state_std = flat_ospec.maximum - flat_ospec.minimum
#         action_std = np.array(action_spec.maximum - action_spec.minimum)
#         action_std = action_std.reshape((-1,))

#         state_std = state_std_scale * state_std
#         action_std = action_std_scale * action_std
#         concat_std = jnp.concatenate([state_std, action_std], axis=0)
#         self.kernel_cov = jnp.diag(concat_std ** 2)
#         self.key_dim = self.kernel_cov.shape[0]

#         # calculate a scalar to shift the max to 1
#         max_pdf = _max_pdf_value(kernel_cov)
#         self.scale_factor = 1 / max_pdf

#         # initialize this to some reasonable size
#         # starting_size = 65536
#         starting_size = 1024
#         self.weights = jnp.zeros((starting_size,))
#         self.total = 0
#         self.next_slot = 0
#         self.max_obs = max_obs

#         self.index = faiss.IndexFlatL2(self.key_dim)


def new(observation_spec, action_spec, max_obs=100000,
        state_std_scale=1, action_std_scale=1, **kwargs):
        flat_ospec = utils.flatten_observation_spec(observation_spec)
        state_std = flat_ospec.maximum - flat_ospec.minimum
        action_std = np.array(action_spec.maximum - action_spec.minimum)
        action_std = action_std.reshape((-1,))

        state_std = state_std_scale * state_std
        action_std = action_std_scale * action_std
        concat_std = jnp.concatenate([state_std, action_std], axis=0)
        kernel_cov = jnp.diag(concat_std ** 2)
        key_dim = kernel_cov.shape[0]

        # calculate a scalar to shift the max to 1
        max_pdf = _max_pdf_value(kernel_cov)
        scale_factor = 1 / max_pdf

        # initialize this to some reasonable size
        # starting_size = 65536
        starting_size = 1024
        weights = np.zeros((starting_size,))
        max_obs = max_obs
        # import ipdb; ipdb.set_trace()
        # index = faiss.IndexFlatL2(key_dim)
        index = faiss.IndexHNSWFlat(key_dim, 32)
        return DensityState(kernel_cov, index, weights,
                            max_obs=max_obs, scale_factor=scale_factor)


def _max_pdf_value(cov):
    k = cov.shape[0]
    return (2 * jnp.pi) ** (-k / 2) * jnp.linalg.det(cov) ** (- 1/2)


def update_batch(density_state: DensityState, states, actions):
    # increase the size of weights vector if needed
    weights = density_state.weights
    weights_size = weights.shape[0]
    while ((density_state.next_slot + states.shape[0] >= weights_size) and
           (weights_size < density_state.max_obs)):
        weights = _grow_weights(weights, density_state.max_obs)
        weights_size = weights.shape[0]
    density_state = density_state.replace(weights=weights)

    # import ipdb; ipdb.set_trace()
    new_states, new_actions, weight_updates = _compute_updates(
        density_state, states, actions)

    # add all the new observations to the index
    if len(new_states) > 0:
        new_states = np.stack(new_states)
        new_actions = np.stack(new_actions)
        density_state = _add_observations(density_state, states, actions)

    # update weights
    if weight_updates.sum() > 0:
        weights = density_state.weights + weight_updates
        density_state = density_state.replace(weights=weights)

    return density_state

def _compute_updates(density_state: DensityState, states, actions):
    weights_size = density_state.weights.shape[0]
    if density_state.index.ntotal <= 0:
        return list(states), list(actions), np.zeros((weights_size,))

    # compute which observations are similar to existing
    sims_per_neighbor, indx = _find_similarities_batch(density_state, states, actions)
    new_states, new_actions = [], []
    weight_updates = np.zeros((weights_size,))

    for (state, action, sims, ind) in zip(states, actions, sims_per_neighbor, indx):
        similar_mask = sims > 0.95
        n_similar_obs = similar_mask.sum()

        if n_similar_obs >= 1:
            similar_meta_indices = np.flatnonzero(similar_mask)
            similar_indices = ind[similar_meta_indices]
            weight_updates[similar_indices] += 1 / n_similar_obs
        else:
            new_states.append(state)
            new_actions.append(action)
    return new_states, new_actions, weight_updates


def _add_observations(density_state: DensityState, states, actions):
    bsize = states.shape[0]
    next_slot = density_state.next_slot
    weights = density_state.weights
    w_size = weights.shape[0]

    # add new observations to the index
    keys = _make_key_batch(states, actions)
    density_state.index.add(np.array(keys))

    # update the weights vector
    indices = jnp.linspace(next_slot, next_slot + bsize - 1, bsize)
    indices = indices.round().astype(int)
    # don't use arange because then you can't jit — size depends on next_slot

    weights = jax.ops.index_update(weights, indices, 1)
    total = jnp.minimum(density_state.total + bsize, w_size)
    next_slot = (next_slot + bsize) % w_size
    return density_state.replace(weights=weights, total=total,
                                 next_slot=next_slot)


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(1,))
def _grow_weights(weights, max_size):
    """An un-jittable function which grows the size of the weights array"""
    w_size = weights.shape[0]
    new_w_size = jnp.minimum(max_size, jnp.ceil(2 * w_size)).astype(int)
    print(f"Growing KDE weights from {w_size} to {new_w_size}.")

    # growing weights
    new_weight_shape = (new_w_size,)
    new_weights = jnp.zeros(new_weight_shape)
    weights = jax.ops.index_update(new_weights,
                                   jax.ops.index[:w_size],
                                   weights)
    return weights


@jax.jit
def _add_weights(density_state: DensityState, new_weights):
    new_weights = density_state.weights + new_weights
    return density_state.replace(weights=new_weights)


@jax.profiler.trace_function
def _find_similarities_batch(density_state: DensityState, states, actions):
    """Put one unit of count on the space at every observation.
    Fall-off from that is specified by the covariance."""
    keys = np.array(_make_key_batch(states, actions))

    # dists is a len(states) x density_state.k array of distances
    # indices is the same shape and contains the indices of the matching keys
    dists, indices, neighbors = density_state.index.search_and_reconstruct(
        keys, density_state.k)

    probs = _normal_diag_pdf_batchedmeanx(
        neighbors, density_state.kernel_cov, keys)
    similarities = probs * density_state.scale_factor

    # a similarity of nan is invalid, e.g. we asked for more neighbors than exist
    similarities = np.nan_to_num(similarities, nan=0.0)
    return similarities, indices


@jax.profiler.trace_function
def get_count(density_state: DensityState, state, action):
    states = jnp.expand_dims(state, axis=0)
    actions = jnp.expand_dims(action, axis=0)
    return get_count_batch(density_state, states, actions)[0]


@jax.profiler.trace_function
def get_count_batch(density_state: DensityState, states, actions):
    # sims_per_neighbor is len(states) x density_state.k
    sims_per_neighbor, indx = _find_similarities_batch(density_state, states, actions)
    neighbor_weights = density_state.weights[indx]
    counts_per_neighbor = neighbor_weights * sims_per_neighbor
    counts = counts_per_neighbor.sum(axis=1)
    return counts


def _make_key(s, a):
    flat_s = utils.flatten_observation(s)
    flat_a = jnp.array(a).reshape((-1,))
    return jnp.concatenate([flat_s, flat_a], axis=0)
_make_key_batch = jax.vmap(_make_key)


# @jax.jit
# def _normal_pdf(mean, cov, x):
#     return jax.scipy.stats.multivariate_normal.pdf(x, mean, cov)
# # for evaluating one x at many means:
# _normal_pdf_batchedmean = jax.vmap(_normal_pdf, in_axes=(0, None, None))
# # for evaluating many xs at a single mean:
# _normal_pdf_batchedx = jax.vmap(_normal_pdf, in_axes=(None, None, 0))
# # for many xs and many means
# _normal_pdf_batchedxmean = jax.vmap(_normal_pdf, in_axes=(0, None, 0))



_batched_norm_pdf = jax.vmap(jax.scipy.stats.norm.pdf)
@jax.jit
def _normal_diag_pdf(mean, cov, x):
    diag_indices = jnp.diag_indices(cov.shape[0])
    variances = cov[diag_indices]
    axis_probs = _batched_norm_pdf(x, mean, variances**0.5)
    return axis_probs.prod()
# for evaluating one x at many means:
_normal_diag_pdf_batchedmean = jax.vmap(_normal_diag_pdf, in_axes=(0, None, None))
# for evaluating many xs at a single mean:
_normal_diag_pdf_batchedx = jax.vmap(_normal_diag_pdf, in_axes=(None, None, 0))

# for many xs and a batch of means per x
_normal_diag_pdf_batchedmeanx = jax.vmap(_normal_diag_pdf_batchedmean,
                                         in_axes=(0, None, 0))


# @jax.jit
# def _centered_normal_diag_pdf(cov, x):
#     diag_indices = jnp.diag_indices(cov.shape[0])
#     stddevs = cov[diag_indices] ** 0.5
#     import ipdb; ipdb.set_trace()
#     axis_probs = _batched_norm_pdf(x, jnp.zeros(x.shape), stddevs)
#     return axis_probs.prod()
# _centered_normal_diag_pdf_batched = jax.vmap(
#     _centered_normal_diag_pdf, in_axes=(None, 0))
# _centered_normal_diag_pdf_doublebatched = jax.vmap(
#     _centered_normal_diag_pdf_batched, in_axes=(None, 0))


if __name__ == "__main__":
    from dm_control import suite
    from observation_domains import DOMAINS
    import jax_specs
    import point

    # env = suite.load('point_mass', 'easy')
    # ospec = DOMAINS['point_mass']['easy']
    env = suite.load('point', 'velocity')
    ospec = DOMAINS['point']['velocity']

    aspec = env.action_spec()
    j_aspec = jax_specs.convert_dm_spec(aspec)
    density_state = new(ospec, aspec, state_std_scale=1e-2, action_std_scale=1)
    # import ipdb; ipdb.set_trace()

    timestep = env.reset()
    state = utils.flatten_observation(timestep.observation)
    actions = utils.sample_uniform_actions(j_aspec, jax.random.PRNGKey(0), 1)
    action = actions[0]

    # states = jnp.expand_dims(state, axis=0)
    # density_state = update_batch(density_state, states, actions)

    visited_states = []
    visited_actions = []
    for i in range(128):
        timestep = env.step(action)
        state = utils.flatten_observation(timestep.observation)
        action = utils.sample_uniform_actions(j_aspec, jax.random.PRNGKey(i), 1)[0]
        visited_states.append(state)
        visited_actions.append(action)

    density_state = update_batch(density_state,
                                 jnp.stack(visited_states),
                                 jnp.stack(visited_actions))
    env.reset()


    # ---------- profiling speed ----------------------
    # density_state = update_batch(density_state,
    #                              jnp.repeat(states, 20000, axis=0),
    #                              jnp.repeat(actions, 20000, axis=0))

    import time
    start = time.time()
    for i in range(1000):
        print(i)
        timestep = env.step(action)
        state = utils.flatten_observation(timestep.observation)
        states = jnp.expand_dims(state, axis=0)

        actions = utils.sample_uniform_actions(j_aspec, jax.random.PRNGKey(i), 1)
        action = actions[0]
        density_state = update_batch(density_state, states, actions)
    end = time.time()
    elapsed_building = end - start

    N_extra = 50000
    extra_states = jnp.repeat(states, N_extra, axis=0)
    extra_actions = jnp.repeat(actions, N_extra, axis=0)
    density_state = update_batch(density_state, extra_states, extra_actions)

    start = time.time()
    N = 128 * 32
    query_states = jnp.repeat(states, N, axis=0)
    query_actions = jnp.repeat(actions, N, axis=0)
    for _ in range(1000):
        count = get_count_batch(density_state, query_states, query_actions)
        _ = float(count.sum())
    end = time.time()
    print(f"Elapsed building: {elapsed_building :.2f}")
    print(f"Elapsed evaluating: {end - start :.2f}")

    # ---------- sanity checking counts --------------------
    # timestep2 = env.step(jnp.ones(aspec.shape))
    # state2 = utils.flatten_observation(timestep2.observation)

    # # print("S1 pdf:", pdf(density_state, state, action))
    # print("S1 count:", get_count(density_state, state, action))

    # # print("S2 pdf:", pdf(density_state, state2, action))
    # print("S2 count:", get_count(density_state, state2, action))
    # density_state_updated = update_batch(density_state,
    #                                      jnp.expand_dims(state2, axis=0),
    #                                      jnp.expand_dims(action, axis=0))
    # # print("S2 pdf after self update:", pdf(density_state_updated, state2, action))
    # print("S2 count after self update:", get_count(density_state_updated,
    #                                                state2, action))

    # print("Batch of counts:", get_count_batch(density_state_updated,
    #                                           jnp.stack([state, state2]),
    #                                           jnp.stack([action, action])))

    # --------- plotting the kernel -----------------
    # from mpl_toolkits import mplot3d
    # import numpy as np
    # import matplotlib.pyplot as plt
    # import numpy as np

    # n_points = 200
    # x = np.linspace(-0.3, 0.3, n_points)
    # y = np.linspace(-0.3, 0.3, n_points)

    # X, Y = np.meshgrid(x, y)
    # states = np.stack([X, Y]).transpose().reshape((-1, 2))
    # actions = jnp.repeat(jnp.expand_dims(action, axis=0), n_points**2, axis=0)
    # Z = get_count_batch(density_state, states, actions)
    # Z = np.array(Z).reshape((n_points, n_points))
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.contour3D(X, Y, Z, 50, cmap='binary')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # fig.savefig('kernel_3d.png')






