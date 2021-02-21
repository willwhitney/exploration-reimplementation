import math
import time
import numpy as np

import jax
from jax import numpy as jnp, random, lax


from dm_control import suite

from observation_domains import DOMAINS
import jax_specs
import point
import utils


env_name = 'point'
task_name = 'velocity'
env = suite.load(env_name, task_name)
ospec = DOMAINS[env_name][task_name]

aspec = env.action_spec()
j_aspec = jax_specs.convert_dm_spec(aspec)
j_ospec = jax_specs.convert_dm_spec(ospec)
flat_ospec = utils.flatten_observation_spec(ospec)
j_flat_ospec = jax_specs.convert_dm_spec(flat_ospec)


def profile_write(density, density_state, n=1000):
    rng = random.PRNGKey(0)
    states = utils.sample_uniform_actions(j_flat_ospec, rng, n)
    actions = utils.sample_uniform_actions(j_aspec, rng, n)
    states = jnp.expand_dims(states, axis=1)
    actions = jnp.expand_dims(actions, axis=1)

    start = time.time()
    for i in range(n):
        density_state = density.update_batch(density_state, states[i], actions[i])
    end = time.time()
    elapsed = end - start
    return density_state, elapsed


def profile_read(density, density_state, n=1000, qsize=128*64):
    rng = random.PRNGKey(0)
    query_states = utils.sample_uniform_actions(j_flat_ospec, rng, qsize)
    query_actions = utils.sample_uniform_actions(j_aspec, rng, qsize)
    # states = states.reshape((n, qsize, states.shape[1:]))
    # actions = actions.reshape((n, qsize, actions.shape[1:]))

    start = time.time()
    for i in range(n):
        # query_states = states[i]
        # query_actions = actions[i]
        count = density.get_count_batch(density_state, query_states, query_actions)
        _ = float(count.sum())
    end = time.time()
    elapsed = end - start
    return elapsed


def fill(density, density_state, n=50000, bsize=1):
    for i in range(math.ceil(n / bsize)):
        rng = random.PRNGKey(i)
        extra_states = utils.sample_uniform_actions(j_flat_ospec, rng, bsize)
        extra_actions = utils.sample_uniform_actions(j_aspec, rng, bsize)
        density_state = density.update_batch(density_state, extra_states, extra_actions)
    return density_state


from densities import kernel_count
from densities import knn_kernel_count

for density in [kernel_count]:
    fill_bsize = 1024
    tolerance = 1.0
    for max_obs in [1024, 4096, 16384, 65536]:
    # max_obs =
    # for tolerance in [0.99]:
        print(f"{density.__name__} with max_obs={max_obs} and tolerance={tolerance}")
        density_state = density.new(ospec, aspec,
                                        state_std_scale=1e-1, action_std_scale=1,
                                        max_obs=max_obs, tolerance=tolerance)

        density_state, elapsed_write_empty = profile_write(density,
                                                        density_state)
        elapsed_read_empty = profile_read(density, density_state)

        density_state = fill(density, density_state,
                            n=min(50000, max_obs), bsize=fill_bsize)
        # print(density_state.total)

        density_state, elapsed_write_full = profile_write(density,
                                                        density_state)
        elapsed_read_full = profile_read(density, density_state)


        print(f"Write empty: {elapsed_write_empty :.2f}")
        print(f"Read empty: {elapsed_read_empty :.2f}")
        print(f"Write full: {elapsed_write_full :.2f}")
        print(f"Read full: {elapsed_read_full :.2f}")
        print()

"""
# Point env
## Size of obs table

kernel_count with max_obs=1024
Write empty: 11.11
Read empty: 1.59
Write full: 8.52
Read full: 1.10

kernel_count with max_obs=4096
Growing KDE observations from 1024 to 2048.
Growing KDE observations from 2048 to 4096.
Write empty: 8.17
Read empty: 0.94
Write full: 8.82
Read full: 1.81

kernel_count with max_obs=16384
Growing KDE observations from 1024 to 2048.
Growing KDE observations from 2048 to 4096.
Growing KDE observations from 4096 to 8192.
Growing KDE observations from 8192 to 16384.
Write empty: 7.97
Read empty: 0.98
Write full: 8.58
Read full: 4.33

kernel_count with max_obs=65536
Growing KDE observations from 1024 to 2048.
Growing KDE observations from 2048 to 4096.
Growing KDE observations from 4096 to 8192.
Growing KDE observations from 8192 to 16384.
Growing KDE observations from 16384 to 32768.
Growing KDE observations from 32768 to 65536.
Write empty: 8.61
Read empty: 1.10
Write full: 9.20
Read full: 20.02


## Tolerance

kernel_count with max_obs=65536 and tolerance=0.6
Write empty: 11.87
Read empty: 1.57
Write full: 8.19
Read full: 0.96

kernel_count with max_obs=65536 and tolerance=0.9
Write empty: 8.14
Read empty: 0.98
Write full: 8.98
Read full: 1.01

kernel_count with max_obs=65536 and tolerance=0.95
Write empty: 8.69
Read empty: 0.99
Write full: 8.72
Read full: 0.98

kernel_count with max_obs=65536 and tolerance=0.97
Growing KDE observations from 1024 to 2048.
Write empty: 8.16
Read empty: 0.98
Write full: 8.36
Read full: 1.38

kernel_count with max_obs=65536 and tolerance=0.99
Growing KDE observations from 1024 to 2048.
Growing KDE observations from 2048 to 4096.
Write empty: 8.72
Read empty: 0.98
Write full: 8.81
Read full: 1.85


# Cartpole
## Tolerance

kernel_count with max_obs=65536 and tolerance=0.3
Growing KDE observations from 1024 to 2048.
Growing KDE observations from 2048 to 4096.
Growing KDE observations from 4096 to 8192.
Write empty: 13.42
Read empty: 1.69
Write full: 8.57
Read full: 4.69

kernel_count with max_obs=65536 and tolerance=0.4
Growing KDE observations from 1024 to 2048.
Growing KDE observations from 2048 to 4096.
Growing KDE observations from 4096 to 8192.
Growing KDE observations from 8192 to 16384.
Write empty: 8.36
Read empty: 0.99
Write full: 8.06
Read full: 8.08

kernel_count with max_obs=65536 and tolerance=0.5
Growing KDE observations from 1024 to 2048.
Growing KDE observations from 2048 to 4096.
Growing KDE observations from 4096 to 8192.
Growing KDE observations from 8192 to 16384.
Growing KDE observations from 16384 to 32768.
Write empty: 8.28
Read empty: 0.99
Write full: 7.87
Read full: 15.90

kernel_count with max_obs=65536 and tolerance=0.6
Growing KDE observations from 1024 to 2048.
Growing KDE observations from 2048 to 4096.
Growing KDE observations from 4096 to 8192.
Growing KDE observations from 8192 to 16384.
Growing KDE observations from 16384 to 32768.
Write empty: 13.04
Read empty: 1.45
Write full: 8.81
Read full: 15.86

kernel_count with max_obs=65536 and tolerance=0.7
Growing KDE observations from 1024 to 2048.
Growing KDE observations from 2048 to 4096.
Growing KDE observations from 4096 to 8192.
Growing KDE observations from 8192 to 16384.
Growing KDE observations from 16384 to 32768.
Growing KDE observations from 32768 to 65536.
Write empty: 8.15
Read empty: 1.04
Write full: 8.72
Read full: 40.13

kernel_count with max_obs=65536 and tolerance=0.8
Growing KDE observations from 1024 to 2048.
Growing KDE observations from 2048 to 4096.
Growing KDE observations from 4096 to 8192.
Growing KDE observations from 8192 to 16384.
Growing KDE observations from 16384 to 32768.
Growing KDE observations from 32768 to 65536.
Write empty: 8.51
Read empty: 1.04
Write full: 8.90
Read full: 38.80

kernel_count with max_obs=65536 and tolerance=0.9
Growing KDE observations from 1024 to 2048.
Growing KDE observations from 2048 to 4096.
Growing KDE observations from 4096 to 8192.
Growing KDE observations from 8192 to 16384.
Growing KDE observations from 16384 to 32768.
Growing KDE observations from 32768 to 65536.
Write empty: 8.03
Read empty: 1.05
Write full: 8.85
Read full: 38.85

kernel_count with max_obs=65536 and tolerance=0.95
Growing KDE observations from 1024 to 2048.
Growing KDE observations from 2048 to 4096.
Growing KDE observations from 4096 to 8192.
Growing KDE observations from 8192 to 16384.
Growing KDE observations from 16384 to 32768.
Growing KDE observations from 32768 to 65536.
Write empty: 8.24
Read empty: 1.06
Write full: 9.43
Read full: 38.87
"""