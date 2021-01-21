import jax
from jax import numpy as jnp, random

from flax import struct

import utils


@struct.dataclass
class DensityState:
    kernel_cov: jnp.ndarray
    observations: jnp.ndarray
    total: int = 0
    next_slot: int = 0


def new(observation_spec, action_spec, max_obs=100000, kernel_cov_scale=1,
        **kwargs):
    flat_ospec = utils.flatten_observation_spec(observation_spec)
    concat_min = jnp.concatenate([flat_ospec.minimum, action_spec.minimum],
                                 axis=0)
    concat_max = jnp.concatenate([flat_ospec.maximum, action_spec.maximum],
                                 axis=0)
    kernel_cov = jnp.diag((concat_max - concat_min) ** 2)

    # normalize covariance so the max value of the Gaussian is 1
    max_pdf = _max_pdf_value(kernel_cov)
    kernel_det_pow = jnp.linalg.det(kernel_cov) ** (1 / kernel_cov.shape[0])
    normalizing_scalar = 1 / (2 * jnp.pi * kernel_det_pow)
    kernel_cov = kernel_cov * normalizing_scalar * kernel_cov_scale
    print("Normalizing kernel covariance by", normalizing_scalar)
    print("New maximum of kernel PDF:", _max_pdf_value(kernel_cov))

    observations = jnp.zeros((max_obs, *concat_min.shape))
    return DensityState(kernel_cov, observations)


def _max_pdf_value(cov):
    k = cov.shape[0]
    return (2 * jnp.pi) ** (-k / 2) * jnp.linalg.det(cov) ** (- 1/2)


def pdf(density_state: DensityState, state, action):
    size_bound = smallest_greater_pow(density_state.total)
    return _jittable_pdf(density_state, state, action, size_bound)


def pdf_batch(density_state: DensityState, states, actions):
    size_bound = smallest_greater_pow(density_state.total)
    return _jittable_pdf_batch(density_state, states, actions, size_bound)


@jax.partial(jax.jit, static_argnums=(3,))
def _jittable_pdf(density_state: DensityState, state, action, size_bound):
    """Jittable function to calculate the pdf under the KDE despite
    variable length of observations.
    Works by taking an extra, static argument size_bound and masking.
    """
    key = _make_key(state, action)
    included_obs = density_state.observations[:size_bound]
    mask = jnp.linspace(0, size_bound - 1, size_bound) < density_state.total
    mask = mask.astype(jnp.int32)

    probs_per_obs = _normal_pdf_batchedmean(included_obs, density_state.kernel_cov, key)
    masked_obs = mask * probs_per_obs
    normalized_prob = masked_obs.sum() / mask.sum()
    return normalized_prob
_jittable_pdf_batch = jax.vmap(_jittable_pdf, in_axes=(None, 0, 0, None))


def smallest_greater_pow(n):
    """Computes the smallest power of 2 which is at least n.
    Useful for dispatching jitted functions using static_argnums
    without too many recompilations."""
    if n == 0:
        return 0
    elif n <= 128:
        return 128
    else:
        return int(math.pow(2, math.ceil(math.log(n, 2))))


@jax.jit
def update_batch(density_state: DensityState, states, actions):
    bsize = states.shape[0]
    next_slot = density_state.next_slot
    observations = density_state.observations

    keys = _make_key_batch(states, actions)
    indices = jnp.linspace(next_slot, next_slot + bsize - 1, bsize)
    indices = indices.astype(jnp.int32) % len(observations)
    observations = jax.ops.index_update(observations, indices, keys)
    total = jnp.minimum(density_state.total + bsize, len(observations))
    next_slot = (next_slot + bsize) % len(observations)
    return density_state.replace(observations=observations,
                                 total=total, next_slot=next_slot)


def get_count(density_state: DensityState, state, action):
    key = _make_key(state, action)
    size_bound = smallest_greater_pow(density_state.total)
    prior_pdf = _jittable_pdf(density_state, state, action, size_bound)
    self_pdf = _normal_pdf(key, density_state.kernel_cov, key)
    n = density_state.total
    updated_pdf = (n * prior_pdf + self_pdf) / (n + 1)
    numerator = prior_pdf * (1 - updated_pdf)
    denominator = updated_pdf - prior_pdf
    pseudocount = numerator / denominator
    return pseudocount


def get_count_batch(density_state: DensityState, states, actions):
    keys = _make_key_batch(states, actions)
    size_bound = smallest_greater_pow(density_state.total)
    prior_pdfs = _jittable_pdf_batch(density_state, states, actions, size_bound)

    # this is a constant, so it's a little silly to do this, but <shrug>
    self_pdfs = _normal_pdf_batchedxmean(keys, density_state.kernel_cov, keys)

    n = density_state.total
    updated_pdfs = (n * prior_pdfs + self_pdfs) / (n + 1)
    pseudocounts = (prior_pdfs * (1 - updated_pdfs)) / (updated_pdfs - prior_pdfs)
    return pseudocounts


def _make_key(s, a):
    flat_s = utils.flatten_observation(s)
    return jnp.concatenate([flat_s, a], axis=0)
_make_key_batch = jax.vmap(_make_key)


def _normal_pdf(mean, cov, x):
    return jax.scipy.stats.multivariate_normal.pdf(x, mean, cov)
# for evaluating one x at many means:
_normal_pdf_batchedmean = jax.vmap(_normal_pdf, in_axes=(0, None, None))
# for evaluating many xs at a single mean:
_normal_pdf_batchedx = jax.vmap(_normal_pdf, in_axes=(None, None, 0))
_normal_pdf_batchedxmean = jax.vmap(_normal_pdf, in_axes=(0, None, 0))


def _mog_pdf(cov, weights, kernel_locs, x):
    """Returns the PDF of a mixture of Gaussians model evaluated at x.

    Arguments:
    - `cov`: a covariance matrix which will be used at every point
    - `weights`: the weight of each point in the mixture model
    - `kernel_locs`: the locations of the observed points which define the model
    - `x`: a query point to evaluate
    """
    normalizer = weights.sum()
    weights = weights / normalizer
    per_kernel_densities = _normal_pdf_batchedmean(kernel_locs, cov, x)
    weighted_densities = weights * per_kernel_densities
    return weighted_densities.sum(axis=0)
_mog_pdf_batchedx = jax.vmap(_mog_pdf, in_axes=(None, None, None, 0))


def kde_pdf(cov, kernel_locs, x):
    weights = jnp.ones(x.shape[0])
    return _mog_pdf(cov, weights, kernel_locs, x)



if __name__ == "__main__":
    from dm_control import suite
    from observation_domains import DOMAINS
    import jax_specs

    env = suite.load('point_mass', 'easy')
    ospec = DOMAINS['point_mass']['easy']

    aspec = env.action_spec()
    j_aspec = jax_specs.convert_dm_spec(aspec)
    density_state = new(ospec, aspec, kernel_std_scale=1e0)

    timestep = env.reset()
    state = utils.flatten_observation(timestep.observation)
    actions = utils.sample_uniform_actions(j_aspec, jax.random.PRNGKey(0), 1)
    action = actions[0]

    states = jnp.expand_dims(state, axis=0)
    density_state = update_batch(density_state, states, actions)

    timestep2 = env.reset()
    state2 = utils.flatten_observation(timestep2.observation)

    print("S1 pdf:", pdf(density_state, state, action))
    print("S1 count:", get_count(density_state, state, action))
    density_state_updated = update_batch(density_state,
                                         jnp.expand_dims(state, axis=0),
                                         jnp.expand_dims(action, axis=0))
    print("S1 pdf after self update:", pdf(density_state_updated, state, action))

    print("S2 pdf:", pdf(density_state, state2, action))
    print("S2 count:", get_count(density_state, state2, action))
    density_state_updated = update_batch(density_state,
                                         jnp.expand_dims(state2, axis=0),
                                         jnp.expand_dims(action, axis=0))
    print("S2 pdf after self update:", pdf(density_state_updated, state2, action))
    print("S2 count after self update:", get_count(density_state_updated,
                                                   state2, action))
