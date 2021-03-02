from dm_control import suite
import jax
from jax import numpy as jnp

from observation_domains import DOMAINS
import jax_specs
import point
import utils

from densities import keops_kernel_count as density


env_name = 'point'
task_name = 'velocity'
env = suite.load(env_name, task_name)
ospec = DOMAINS[env_name][task_name]

aspec = env.action_spec()
j_aspec = jax_specs.convert_dm_spec(aspec)
j_ospec = jax_specs.convert_dm_spec(ospec)
density_state = density.new(ospec, aspec, state_scale=0.01, action_scale=1)

timestep = env.reset()
state = utils.flatten_observation(timestep.observation)
actions = utils.sample_uniform_actions(j_aspec, jax.random.PRNGKey(0), 1)
action = actions[0]


# ---------- sanity checking counts --------------------
timestep2 = env.step(jnp.ones(aspec.shape))
state2 = utils.flatten_observation(timestep2.observation)

print("S1 count:", density.get_count(density_state, state, action))

print("S2 count:", density.get_count(density_state, state2, action))
density_state_updated = density.update_batch(density_state,
                                        jnp.expand_dims(state2, axis=0),
                                        jnp.expand_dims(action, axis=0))
print("S2 count after self update:", density.get_count(density_state_updated,
                                            state2, action))

print("Batch of counts:", density.get_count_batch(density_state_updated,
                                            jnp.stack([state, state2]),
                                            jnp.stack([action, action])))
