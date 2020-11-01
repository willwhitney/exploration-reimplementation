import matplotlib.pyplot as plt
import typing

import jax
from jax import numpy as jnp
from flax import struct

import gridworld
import jax_specs
import utils


@struct.dataclass
class DensitySettings():
    """Represents the un-jittable fixed settings of the density."""
    observation_spec: jax_specs.Array
    action_spec: jax_specs.Array
    state_bins: int = 4
    action_bins: int = 2


@struct.dataclass
class DensityState():
    settings: DensitySettings
    observations: jnp.ndarray = jnp.array((4, 4, 2))
    total: int = 1
    eps: float = 1e-8


# def make_density(observation_spec, action_spec, state_bins=4, action_bins=2):
#     """Creates a new DensityState."""
#     observation_spec = utils.flatten_observation_spec(observation_spec)
#     observation_spec = jax_specs.convert_dm_spec(observation_spec)
#     action_spec = jax_specs.convert_dm_spec(action_spec)
#     state_shape = observation_spec.shape
#     if isinstance(action_spec, jax_specs.DiscreteArray):
#         action_shape = (1,)
#     else:
#         action_shape = action_spec.shape
#     state_values_per_dim = [state_bins
#                             for s in state_shape for _ in range(s)]
#     action_values_per_dim = [action_bins
#                              for a in action_shape for _ in range(a)]
#     total_dimension_list = state_values_per_dim + action_values_per_dim
#     storage = jnp.zeros(tuple(total_dimension_list), dtype=jnp.uint32)

#     return DensityState(observations=storage)


#     @jax.jit
#     def _make_key(s, a):
#         ospec = observation_spec
#         aspec = action_spec
#         state_bins = state_bins
#         action_bins = action_bins
#         discrete_state = utils.discretize_observation(s, ospec, state_bins)
#         discrete_action = utils.discretize_observation(a, aspec, action_bins)
#         discrete_trans = [discrete_state.flatten(), discrete_action.flatten()]
#         return tuple(jnp.concatenate(discrete_trans).astype(int))
#     _make_key_batch = jax.vmap(_make_key)  # noqa: E305


#     @jax.jit
#     def update_batch(density_state: DensityState, states, actions):
#         keys = _make_key_batch(states, actions)
#         # keys = _make_key_gridworld_batch(states, actions)
#         new_observations = jax.ops.index_add(density_state.observations, keys, 1)
#         new_total = density_state.total + states.shape[0]
#         density_state = density_state.replace(observations=new_observations,
#                                               total=new_total)
#         return density_state


#     def get_count(density_state: DensityState, state, action):
#         key = _make_key(state, action)
#         # key = _make_key_gridworld(state, action)
#         return density_state.observations[key]


#     @jax.jit
#     def get_count_batch(density_state: DensityState, states, actions):
#         keys = _make_key_batch(states, actions)
#         # keys = _make_key_gridworld_batch(states, actions)
#         return density_state.observations[keys]


# def log_p(density_state: DensityState, state, action):
#     count = get_count(density_state, state, action)
#     return jnp.log(count / density_state.total + density_state.eps)
# log_p_batch = jax.vmap(log_p, in_axes=(None, 0, 0))  # noqa: E305


def _process_gridworld_state(s):
    return jnp.argmax(s, axis=-1).astype(jnp.int16)


def _process_gridworld_action(a):
    return a.astype(jnp.int16)


def _make_key_gridworld(s, a):
    s = _process_gridworld_state(s)
    a = _process_gridworld_action(a)
    return tuple(jnp.concatenate([s.flatten(), a.flatten()]))
_make_key_gridworld_batch = jax.vmap(_make_key_gridworld)  # noqa: E305


def new(observation_spec, action_spec, state_bins=4, action_bins=2):
    """Creates a new DensityState."""
    observation_spec = utils.flatten_observation_spec(observation_spec)
    observation_spec = jax_specs.convert_dm_spec(observation_spec)
    action_spec = jax_specs.convert_dm_spec(action_spec)
    state_shape = observation_spec.shape
    if isinstance(action_spec, jax_specs.DiscreteArray):
        action_shape = (1,)
    else:
        action_shape = action_spec.shape
    state_values_per_dim = [state_bins
                            for s in state_shape for _ in range(s)]
    action_values_per_dim = [action_bins
                             for a in action_shape for _ in range(a)]
    total_dimension_list = state_values_per_dim + action_values_per_dim
    storage = jnp.zeros(tuple(total_dimension_list), dtype=jnp.uint32)

    density_settings = DensitySettings(observation_spec, action_spec,
                                       state_bins, action_bins)
    return DensityState(density_settings, observations=storage)


def update_batch(density_state: DensityState, states, actions):
    # keys = _make_key_batch(density_state.settings, states, actions)
    # # keys = _make_key_gridworld_batch(states, actions)
    # new_observations = jax.ops.index_add(density_state.observations, keys, 1)
    # new_total = density_state.total + states.shape[0]
    # density_state = density_state.replace(observations=new_observations,
    #                                       total=new_total)
    # return density_state
    return _update_batch(density_state, density_state.settings, states, actions)


@jax.jit
def _update_batch(density_state: DensityState,
                  density_settings: DensitySettings,
                  states, actions):
    keys = _make_key_batch(density_settings, states, actions)
    # keys = _make_key_gridworld_batch(states, actions)
    new_observations = jax.ops.index_add(density_state.observations, keys, 1)
    new_total = density_state.total + states.shape[0]
    density_state = density_state.replace(observations=new_observations,
                                          total=new_total)
    return density_state


def get_count(density_state: DensityState, state, action):
    key = _make_key(density_state.settings, state, action)
    # key = _make_key_gridworld(state, action)
    return density_state.observations[key]


@jax.jit
def get_count_batch(density_state: DensityState, states, actions):
    keys = _make_key_batch(density_state.settings, states, actions)
    # keys = _make_key_gridworld_batch(states, actions)
    return density_state.observations[keys]


def log_p(density_state: DensityState, state, action):
    count = get_count(density_state, state, action)
    return jnp.log(count / density_state.total + density_state.eps)
log_p_batch = jax.vmap(log_p, in_axes=(None, 0, 0))  # noqa: E305


def _process_gridworld_state(s):
    return jnp.argmax(s, axis=-1).astype(jnp.int16)


def _process_gridworld_action(a):
    return a.astype(jnp.int16)


def _make_key_gridworld(s, a):
    s = _process_gridworld_state(s)
    a = _process_gridworld_action(a)
    return tuple(jnp.concatenate([s.flatten(), a.flatten()]))
_make_key_gridworld_batch = jax.vmap(_make_key_gridworld)  # noqa: E305


@jax.jit
def _make_key(density_settings: DensitySettings, s, a):
    ospec = density_settings.observation_spec
    aspec = density_settings.action_spec
    state_bins = density_settings.state_bins
    action_bins = density_settings.action_bins
    discrete_state = utils.discretize_observation(s, ospec, state_bins)
    discrete_action = utils.discretize_observation(a, aspec, action_bins)
    discrete_trans = [discrete_state.flatten(), discrete_action.flatten()]
    return tuple(jnp.concatenate(discrete_trans).astype(int))
_make_key_batch = jax.vmap(_make_key, in_axes=(None, 0, 0))  # noqa: E305


def _flatten(s, a):
    return jnp.concatenate([s.flatten(), a.flatten()])


# -------- Visualization tools for gridworld -------------------------------
def render_density_map(density_state: DensityState,
                       env: gridworld.GridWorld):
    count_map = gridworld.render_function(
        jax.partial(get_count_batch, density_state),
        env, reduction=jnp.min)
    return count_map


def display_density_map(density_state: DensityState,
                        env: gridworld.GridWorld):
    density_map = render_density_map(density_state, env)
    fig, ax = plt.subplots()
    img = ax.imshow(density_map)
    fig.colorbar(img, ax=ax)
    ax.set_title("State visitation counts (min)")
    fig.show()
    plt.close(fig)
# --------------------------------------------------------------------------


if __name__ == "__main__":
    # import random
    # density_state = new([10, 10], [4])
    # gw = gridworld.new(10)
    # for i in range(10):
    #     a = random.randint(0, 3)
    #     a_vec = jnp.array(a, dtype=jnp.int16)
    #     gw, obs, r = gridworld.step(gw, a)
    #     print(obs)
    #     density_state = update_batch(density_state,
    #                                  jnp.expand_dims(obs, axis=0),
    #                                  jnp.expand_dims(a, axis=0))
    # display_density_map(density_state, gw)

    from dm_control import suite
    from observation_domains import DOMAINS
    env = suite.load('point_mass', 'easy')
    ospec = DOMAINS['point_mass']['easy']
    # ospec = jax_specs.convert_dm_spec(ospec)
    aspec = env.action_spec()
    density_state = new(ospec, aspec, state_bins=10, action_bins=2)

    timestep = env.reset()
    state = utils.flatten_observation(timestep.observation)
    actions = utils.sample_uniform_actions(aspec, jax.random.PRNGKey(0), 1)
    action = actions[0]
    key = _make_key(density_state.settings, state, action)

    states = jnp.expand_dims(state, axis=0)
    density_state = update_batch(density_state, states, actions)


    pass
