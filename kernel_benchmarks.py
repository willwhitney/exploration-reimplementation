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


env_name = 'hopper'
task_name = 'hop'
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
    states = np.array(jnp.expand_dims(states, axis=1))
    actions = np.array(jnp.expand_dims(actions, axis=1))

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
    query_states = np.array(query_states)
    query_actions = np.array(query_actions)

    counts = []
    start = time.time()
    for i in range(n):
        count = density.get_count_batch(density_state, query_states, query_actions)
        counts.append(float(count.sum()))
    end = time.time()
    elapsed = end - start
    return elapsed, np.array(counts).mean()


def fill(density, density_state, n=50000, bsize=1):
    for i in range(math.ceil(n / bsize)):
        rng = random.PRNGKey(i)
        extra_states = utils.sample_uniform_actions(j_flat_ospec, rng, bsize)
        extra_actions = utils.sample_uniform_actions(j_aspec, rng, bsize)
        density_state = density.update_batch(density_state, extra_states, extra_actions)
    return density_state


from densities import kernel_count
from densities import knn_kernel_count

for density in [knn_kernel_count]:
    fill_bsize = 4096
    tolerance = 1.0
    for max_obs in [4096, 16384]:
    # max_obs =
    # for tolerance in [0.99]:
        print((f"{env_name} {task_name} {density.__name__} "
               f"with max_obs={max_obs} and tolerance={tolerance}"))
        density_state = density.new(ospec, aspec,
                                        state_std_scale=1e-1, action_std_scale=1,
                                        max_obs=max_obs, tolerance=tolerance)

        density_state, elapsed_write_empty = profile_write(density,
                                                        density_state)
        elapsed_read_empty, mean_count_empty = profile_read(density, density_state)

        density_state = fill(density, density_state,
                            n=min(50000, max_obs), bsize=fill_bsize)
        # print(density_state.total)

        density_state, elapsed_write_full = profile_write(density,
                                                        density_state)
        elapsed_read_full, mean_count_full = profile_read(density, density_state)


        print(f"Write empty: {elapsed_write_empty :.2f}")
        print(f"Read empty: {elapsed_read_empty :.2f}, Count: {mean_count_empty :.8f}")
        print(f"Write full: {elapsed_write_full :.2f}")
        print(f"Read full: {elapsed_read_full :.2f}, Count: {mean_count_full :.8f}")
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