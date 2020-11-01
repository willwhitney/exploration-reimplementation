import time
import jax
from jax import numpy as jnp, random
from dm_control import suite

import policies.deep_q_policy as policy
from observation_domains import DOMAINS
import jax_specs
import utils

import q_learning


env = suite.load("point_mass", "easy")
observation_spec = DOMAINS["point_mass"]["easy"]

action_spec = env.action_spec()
action_spec = jax_specs.convert_dm_spec(action_spec)
j_observation_spec = jax_specs.convert_dm_spec(observation_spec)

policy_state = policy.init_fn(0, j_observation_spec, action_spec)
policy_settings = policy_state.settings
timestep = env.reset()
s = utils.flatten_observation(timestep.observation)
s_batch = jnp.expand_dims(s, axis=0)

# for i in range(5):
#     start = time.time()
#     with jax.profiler.TraceContext(f"sampling action_fn {i}"):
#         policy_state, candidate_actions = policy.action_fn(
#             policy_state, s_batch, 32, True)
#     end = time.time()
#     print(f"Time for action_fn {i :3}: {end - start :.2f}s")

# for i in range(5):
#     start = time.time()
#     with jax.profiler.TraceContext(f"sampling redundant _action_fn {i}"):
#         policy_state, candidate_actions = policy._action_fn(
#             policy_state, policy_state.settings, s_batch, 32, True)
#     end = time.time()
#     print(f"Time for redundant _action_fn {i :3}: {end - start :.2f}s")

# for i in range(5):
#     start = time.time()
#     with jax.profiler.TraceContext(f"sampling _action_fn {i}"):
#         policy_state, candidate_actions = policy._action_fn(
#             policy_state, policy_settings, s_batch, 32, True)
#     end = time.time()
#     print(f"Time for _action_fn {i :3}: {end - start :.2f}s")

# for i in range(5):
#     start = time.time()
#     with jax.profiler.TraceContext(f"sampling uses_raw_action_fn {i}"):
#         policy_state, candidate_actions = policy.uses_raw_action_fn(
#             policy_state, s_batch, 32, True)
#     end = time.time()
#     print(f"Time for uses_raw_action_fn {i :3}: {end - start :.2f}s")




@jax.profiler.trace_function
def action_fn(policy_state, s, n=1, explore=True):
    return _action_fn(policy_state, policy_state.settings,
                      s, n, explore)

@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(1, 3, 4))
def _action_fn(policy_state, policy_settings,
               s, n=1, explore=True):
    """A dummy function to allow settings to be static."""
    bsize = s.shape[0]
    rngs = random.split(policy_state.rng, bsize * n + 1)
    policy_rng = rngs[0]
    policy_rng, candidate_rng = random.split(policy_rng)
    action_rngs = rngs[1:].reshape((bsize, n, -1))

    with jax.profiler.TraceContext("sample uniform actions"):
        candidate_actions = utils.sample_uniform_actions(
            policy_settings.action_spec, candidate_rng,
            32 * bsize)
            # policy_state.n_candidates * bsize)

    # candidate_actions = jnp.zeros((bsize * n, 2))
    with jax.profiler.TraceContext("all but sample uniform"):
        candidate_shape = (bsize,
                        32,
                        #    policy_state.n_candidates,
                        *candidate_actions.shape[1:])
        candidate_actions = candidate_actions.reshape(candidate_shape)
        if explore:
            with jax.profiler.TraceContext("sample boltzmann"):
                actions, values = q_learning.sample_action_boltzmann_n_batch(
                    policy_state.q_state, action_rngs, s, candidate_actions, 0.1)
        else:
            with jax.profiler.TraceContext("sample egreedy"):
                actions, values = q_learning.sample_action_egreedy_n_batch(
                    policy_state.q_state, action_rngs, s, candidate_actions, 0.01)
        policy_state = policy_state.replace(rng=policy_rng)
    return policy_state, actions

# @jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(1, 3, 4))
def _aspec_action_fn(policy_state, action_spec,
               s, n=1, explore=True):
    """A dummy function to allow settings to be static."""
    bsize = s.shape[0]
    rngs = random.split(policy_state.rng, bsize * n + 1)
    policy_rng = rngs[0]
    policy_rng, candidate_rng = random.split(policy_rng)
    action_rngs = rngs[1:].reshape((bsize, n, -1))

    # with jax.profiler.TraceContext("sample uniform actions"):
    candidate_actions = utils.sample_uniform_actions(
        action_spec, candidate_rng,
        32 * bsize)
            # policy_state.n_candidates * bsize)

    # candidate_actions = jnp.zeros((bsize * n, 2))
    # with jax.profiler.TraceContext("all but sample uniform"):
    candidate_shape = (bsize,
                    32,
                    #    policy_state.n_candidates,
                    *candidate_actions.shape[1:])
    candidate_actions = candidate_actions.reshape(candidate_shape)
    if explore:
        with jax.profiler.TraceContext("sample boltzmann"):
            actions, values = q_learning.sample_action_boltzmann_n_batch(
                policy_state.q_state, action_rngs, s, candidate_actions, 0.1)
    else:
        with jax.profiler.TraceContext("sample egreedy"):
            actions, values = q_learning.sample_action_egreedy_n_batch(
                policy_state.q_state, action_rngs, s, candidate_actions, 0.01)
    policy_state = policy_state.replace(rng=policy_rng)
    return policy_state, actions

@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(1, 3, 4))
def _aspec_return_action_fn(policy_state, action_spec,
               s, n=1, explore=True):
    """A dummy function to allow settings to be static."""
    bsize = s.shape[0]
    rngs = random.split(policy_state.rng, bsize * n + 1)
    policy_rng = rngs[0]
    policy_rng, candidate_rng = random.split(policy_rng)
    action_rngs = rngs[1:].reshape((bsize, n, -1))

    with jax.profiler.TraceContext("sample uniform actions"):
        candidate_actions = utils.sample_uniform_actions(
            action_spec, candidate_rng,
            32 * bsize)
            # policy_state.n_candidates * bsize)

    # candidate_actions = jnp.zeros((bsize * n, 2))
    with jax.profiler.TraceContext("all but sample uniform"):
        candidate_shape = (bsize,
                        32,
                        #    policy_state.n_candidates,
                        *candidate_actions.shape[1:])
        candidate_actions = candidate_actions.reshape(candidate_shape)
        if explore:
            with jax.profiler.TraceContext("sample boltzmann"):
                actions, values = q_learning.sample_action_boltzmann_n_batch(
                    policy_state.q_state, action_rngs, s, candidate_actions, 0.1)
        else:
            with jax.profiler.TraceContext("sample egreedy"):
                actions, values = q_learning.sample_action_egreedy_n_batch(
                    policy_state.q_state, action_rngs, s, candidate_actions, 0.01)
        policy_state = policy_state.replace(rng=policy_rng)
    return policy_state, action_spec, actions


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def _only_aspec_action_fn(action_spec):
    """A dummy function to allow settings to be static."""
    s = s_batch
    n = 32
    explore = True
    bsize = s.shape[0]
    rngs = random.split(policy_state.rng, bsize * n + 1)
    policy_rng = rngs[0]
    policy_rng, candidate_rng = random.split(policy_rng)
    action_rngs = rngs[1:].reshape((bsize, n, -1))

    with jax.profiler.TraceContext("sample uniform actions"):
        candidate_actions = utils.sample_uniform_actions(
            action_spec, candidate_rng,
            32 * bsize)
            # policy_state.n_candidates * bsize)

    # candidate_actions = jnp.zeros((bsize * n, 2))
    with jax.profiler.TraceContext("all but sample uniform"):
        candidate_shape = (bsize,
                        32,
                        #    policy_state.n_candidates,
                        *candidate_actions.shape[1:])
        candidate_actions = candidate_actions.reshape(candidate_shape)
        if explore:
            with jax.profiler.TraceContext("sample boltzmann"):
                actions, values = q_learning.sample_action_boltzmann_n_batch(
                    policy_state.q_state, action_rngs, s, candidate_actions, 0.1)
        else:
            with jax.profiler.TraceContext("sample egreedy"):
                actions, values = q_learning.sample_action_egreedy_n_batch(
                    policy_state.q_state, action_rngs, s, candidate_actions, 0.01)
        # policy_state = policy_state.replace(rng=policy_rng)
    return action_spec, actions


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def _only_aspec_uniform_action_fn(action_spec):
    """A dummy function to allow settings to be static."""
    s = s_batch
    n = 32
    explore = True
    bsize = 1
    # rngs = random.split(policy_state.rng, bsize * n + 1)
    # policy_rng = rngs[0]
    # policy_rng, candidate_rng = random.split(policy_rng)
    # action_rngs = rngs[1:].reshape((bsize, n, -1))
    candidate_rng = random.PRNGKey(0)

    with jax.profiler.TraceContext("sample uniform actions"):
        candidate_actions = utils.sample_uniform_actions(
            action_spec, candidate_rng,
            32 * bsize)
            # policy_state.n_candidates * bsize)

    return action_spec, candidate_actions


@jax.partial(jax.jit, static_argnums=(0,))
def _notrace_only_aspec_uniform_action_fn(action_spec):
    """A dummy function to allow settings to be static."""
    s = s_batch
    n = 32
    explore = True
    bsize = 1
    candidate_rng = random.PRNGKey(0)

    candidate_actions = utils.sample_uniform_actions(
        action_spec, candidate_rng,
        32 * bsize)

    return action_spec, candidate_actions


@jax.partial(jax.jit, static_argnums=(0,))
def fspec_sample(aspec: jax_specs.BoundedArray):
    candidate_actions = utils.sample_uniform_actions(
        action_spec, random.PRNGKey(0),
        32 * 1)
    return aspec, candidate_actions

# print("action_fn")
# old_policy_state = policy_state
# for i in range(5):
#     start = time.time()
#     # print((#f"Policy Equality: {policy_state == old_policy_state}  |  "
#     #        f"Policy Identity: {policy_state is old_policy_state}"))
#     print((f"Settings Equality: {policy_state.settings == policy_settings}  |  "
#            f"Settings Identity: {policy_state.settings is policy_settings}"))
    
#     policy_state, candidate_actions = action_fn(
#         policy_state, s_batch, 32, True)
#     end = time.time()
#     print(f"Time for action_fn {i :3}: {end - start :.2f}s")

# print()
# print("_action_fn")
# for i in range(5):
#     start = time.time()
#     # print((#f"Policy Equality: {policy_state == old_policy_state}  |  "
#     #        f"Policy Identity: {policy_state is old_policy_state}"))
#     print((f"Settings Equality: {policy_state.settings == policy_settings}  |  "
#            f"Settings Identity: {policy_state.settings is policy_settings}"))

#     policy_state, candidate_actions = _action_fn(
#         policy_state, policy_state.settings, s_batch, 32, True)
#     end = time.time()
#     print(f"Time for _action_fn {i :3}: {end - start :.2f}s")

print()
print("_aspec_action_fn")
for i in range(5):
    start = time.time()
    # print((#f"Policy Equality: {policy_state == old_policy_state}  |  "
    #        f"Policy Identity: {policy_state is old_policy_state}"))
    print((f"Aspec Equality: {policy_state.settings.action_spec == action_spec}  |  "
           f"Aspec Identity: {policy_state.settings.action_spec is action_spec}"))

    policy_state, candidate_actions = _aspec_action_fn(
        policy_state, policy_state.settings.action_spec, s_batch, 32, True)
    end = time.time()
    print(f"Time for _aspec_action_fn {i :3}: {end - start :.2f}s")

print()
print("_aspec_return_action_fn")
old_action_spec = action_spec
for i in range(5):
    start = time.time()
    # print((#f"Policy Equality: {policy_state == old_policy_state}  |  "
    #        f"Policy Identity: {policy_state is old_policy_state}"))
    print((f"Aspec Equality: {old_action_spec == action_spec}  |  "
           f"Aspec Identity: {old_action_spec is action_spec}"))
    last_action_spec = action_spec
    policy_state, action_spec, candidate_actions = _aspec_return_action_fn(
        policy_state, action_spec, s_batch, 32, True)
    print((f"Last Aspec Equality: {last_action_spec == action_spec}  |  "
           f"Last Aspec Identity: {last_action_spec is action_spec}"))
    # import ipdb; ipdb.set_trace()
    end = time.time()
    print(f"Time for _aspec_return_action_fn {i :3}: {end - start :.2f}s")

print()
print("_only_aspec_action_fn")
old_action_spec = action_spec
for i in range(5):
    start = time.time()
    # print((#f"Policy Equality: {policy_state == old_policy_state}  |  "
    #        f"Policy Identity: {policy_state is old_policy_state}"))
    print((f"Aspec Equality: {old_action_spec == action_spec}  |  "
           f"Aspec Identity: {old_action_spec is action_spec}"))
    last_action_spec = action_spec
    action_spec, candidate_actions = _only_aspec_action_fn(action_spec)
    print((f"Last Aspec Equality: {last_action_spec == action_spec}  |  "
           f"Last Aspec Identity: {last_action_spec is action_spec}"))
    # import ipdb; ipdb.set_trace()
    end = time.time()
    print(f"Time for _only_aspec_action_fn {i :3}: {end - start :.2f}s")

print()
print("_only_aspec_uniform_action_fn")
old_action_spec = action_spec
for i in range(5):
    start = time.time()
    # print((#f"Policy Equality: {policy_state == old_policy_state}  |  "
    #        f"Policy Identity: {policy_state is old_policy_state}"))
    print((f"Aspec Equality: {old_action_spec == action_spec}  |  "
           f"Aspec Identity: {old_action_spec is action_spec}"))
    last_action_spec = action_spec
    action_spec, candidate_actions = _only_aspec_uniform_action_fn(action_spec)
    print((f"Last Aspec Equality: {last_action_spec == action_spec}  |  "
           f"Last Aspec Identity: {last_action_spec is action_spec}"))
    # import ipdb; ipdb.set_trace()
    end = time.time()
    print(f"Time for _only_aspec_uniform_action_fn {i :3}: {end - start :.2f}s")

print()
print("_notrace_only_aspec_uniform_action_fn")
old_action_spec = action_spec
for i in range(5):
    start = time.time()
    # print((#f"Policy Equality: {policy_state == old_policy_state}  |  "
    #        f"Policy Identity: {policy_state is old_policy_state}"))
    print((f"Aspec Equality: {old_action_spec == action_spec}  |  "
           f"Aspec Identity: {old_action_spec is action_spec}"))
    last_action_spec = action_spec
    action_spec, candidate_actions = _notrace_only_aspec_uniform_action_fn(action_spec)
    print((f"Last Aspec Equality: {last_action_spec == action_spec}  |  "
           f"Last Aspec Identity: {last_action_spec is action_spec}"))
    # import ipdb; ipdb.set_trace()
    end = time.time()
    print(f"Time for _notrace_only_aspec_uniform_action_fn {i :3}: {end - start :.2f}s")


print()
print("fspec_sample")
old_action_spec = action_spec
for i in range(5):
    start = time.time()
    # print((#f"Policy Equality: {policy_state == old_policy_state}  |  "
    #        f"Policy Identity: {policy_state is old_policy_state}"))
    print((f"Aspec Equality: {old_action_spec == action_spec}  |  "
           f"Aspec Identity: {old_action_spec is action_spec}"))
    last_action_spec = action_spec
    action_spec, candidate_actions = fspec_sample(action_spec)
    print((f"Last Aspec Equality: {last_action_spec == action_spec}  |  "
           f"Last Aspec Identity: {last_action_spec is action_spec}"))
    # import ipdb; ipdb.set_trace()
    end = time.time()
    print(f"Time for fspec_sample {i :3}: {end - start :.2f}s")

print()
print("fspec_sample identity")
old_action_spec = action_spec
for i in range(5):
    start = time.time()
    # print((#f"Policy Equality: {policy_state == old_policy_state}  |  "
    #        f"Policy Identity: {policy_state is old_policy_state}"))
    print((f"Aspec Equality: {old_action_spec == action_spec}  |  "
           f"Aspec Identity: {old_action_spec is action_spec}"))
    last_action_spec = action_spec
    _, candidate_actions = fspec_sample(action_spec)
    print((f"Last Aspec Equality: {last_action_spec == action_spec}  |  "
           f"Last Aspec Identity: {last_action_spec is action_spec}"))
    # import ipdb; ipdb.set_trace()
    end = time.time()
    print(f"Time for fspec_sample {i :3}: {end - start :.2f}s")
