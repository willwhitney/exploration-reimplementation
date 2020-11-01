import time
import jax
from jax import numpy as jnp, random
from flax import struct

from dm_env import specs
from dm_control import suite
import jax_specs
import utils


fixed_rng = random.PRNGKey(0)


def test_fn(fn):
    name = fn.__name__
    env = suite.load("point_mass", "easy")
    action_spec = env.action_spec()
    action_spec = jax_specs.convert_dm_spec(action_spec)
    old_action_spec = action_spec
    with jax.profiler.TraceContext(name):
        print(name)
        for i in range(5):
            start = time.time()
            last_action_spec = action_spec

            with jax.profiler.TraceContext(f"Round {i}"):
                action_spec, candidate_actions = fn(action_spec)

            end = time.time()
            print((f"Time {i :3}: {end - start :.4f}s  |  "
                   f"Equality: {action_spec == old_action_spec}  |  "
                   f"Identity: {action_spec is old_action_spec}  |  "
                   f"Last Equality: {action_spec == last_action_spec}  |  "
                   f"Last Identity: {action_spec is last_action_spec}"))
    print()


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def sample_uniform_actions_simple_no_n(action_spec, rng):
    n = 32
    shape = (n, 2)
    minval = jnp.expand_dims(action_spec.minimum, axis=0).tile((n, 1))
    maxval = jnp.expand_dims(action_spec.maximum, axis=0).tile((n, 1))
    actions = random.uniform(rng, shape=shape,
                             minval=minval, maxval=maxval)
    return actions


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def fspec_sample_simple_no_n(action_spec: jax_specs.BoundedArray):
    candidate_actions = sample_uniform_actions_simple_no_n(
        action_spec, random.PRNGKey(0))
    return action_spec, candidate_actions


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def fspec_sample_simple_no_n_fixed_rng(action_spec: jax_specs.BoundedArray):
    candidate_actions = sample_uniform_actions_simple_no_n(
        action_spec, fixed_rng)
    return action_spec, candidate_actions



@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def sample_uniform_actions_simple_no_n_no_rng(action_spec):
    n = 32
    shape = (n, 2)
    minval = jnp.expand_dims(action_spec.minimum, axis=0).tile((n, 1))
    maxval = jnp.expand_dims(action_spec.maximum, axis=0).tile((n, 1))
    actions = random.uniform(random.PRNGKey(0), shape=shape,
                             minval=minval, maxval=maxval)
    return actions


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def fspec_sample_simple_no_n_no_rng(action_spec: jax_specs.BoundedArray):
    candidate_actions = sample_uniform_actions_simple_no_n_no_rng(
        action_spec)
    return action_spec, candidate_actions


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def sample_return_spec(action_spec):
    n = 32
    shape = (n, 2)
    minval = jnp.expand_dims(action_spec.minimum, axis=0).tile((n, 1))
    maxval = jnp.expand_dims(action_spec.maximum, axis=0).tile((n, 1))
    actions = random.uniform(random.PRNGKey(0), shape=shape,
                             minval=minval, maxval=maxval)
    return action_spec, actions


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def sample_return_spec_fixed_rng(action_spec):
    n = 32
    shape = (n, 2)
    minval = jnp.expand_dims(action_spec.minimum, axis=0).tile((n, 1))
    maxval = jnp.expand_dims(action_spec.maximum, axis=0).tile((n, 1))
    actions = random.uniform(fixed_rng, shape=shape,
                             minval=minval, maxval=maxval)
    return action_spec, actions


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def dummy_fn(action_spec, rng):
    return jnp.zeros((3, 3))


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def dummy_wrapper(action_spec):
    return action_spec, dummy_fn(action_spec, fixed_rng)


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def less_dummy_fn(action_spec, rng):
    return random.uniform(rng, shape=(3, 3))


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def less_dummy_wrapper(action_spec):
    return action_spec, less_dummy_fn(action_spec, fixed_rng)



# jobs = [
#     #---- good ----
#     # fspec_sample_simple_no_n_no_rng,
#     # sample_return_spec,
#     # sample_return_spec_fixed_rng,
#     dummy_wrapper,

#     #---- bad ----
#     less_dummy_wrapper,
#     # fspec_sample_simple_no_n_fixed_rng,
# ]


# for fn in jobs:
#     test_fn(fn)



# @jax.profiler.trace_function
# @jax.partial(jax.jit, static_argnums=(0,))
# def dummy_nospec_fn(not_a_spec, rng):
#     return random.uniform(rng, shape=(3, 3))


# @jax.profiler.trace_function
# @jax.partial(jax.jit, static_argnums=(0,))
# def dummy_nospec_wrapper(not_a_spec):
#     return not_a_spec, dummy_nospec_fn(not_a_spec, fixed_rng)


# fn = dummy_nospec_wrapper
# name = fn.__name__
# dummy_arg = [True]
# old_dummy_arg = dummy_arg
# with jax.profiler.TraceContext(name):
#     print(name)
#     for i in range(5):
#         start = time.time()
#         last_dummy_arg = dummy_arg

#         with jax.profiler.TraceContext(f"Round {i}"):
#             dummy_arg, candidate_actions = fn(dummy_arg)
#             dummy_arg = [True]

#         end = time.time()
#         print((f"Time {i :3}: {end - start :.4f}s  |  "
#                 f"Equality: {dummy_arg == old_dummy_arg}  |  "
#                 f"Identity: {dummy_arg is old_dummy_arg}  |  "
#                 f"Last Equality: {dummy_arg == last_dummy_arg}  |  "
#                 f"Last Identity: {dummy_arg is last_dummy_arg}"))
# print()


# fn = dummy_nospec_fn
# name = fn.__name__
# dummy_arg = [True]
# old_dummy_arg = dummy_arg
# with jax.profiler.TraceContext(name):
#     print(name)
#     for i in range(5):
#         start = time.time()
#         last_dummy_arg = dummy_arg

#         with jax.profiler.TraceContext(f"Round {i}"):
#             _ = fn(dummy_arg, fixed_rng)
#             dummy_arg = [True]

#         end = time.time()
#         print((f"Time {i :3}: {end - start :.4f}s  |  "
#                 f"Equality: {dummy_arg == old_dummy_arg}  |  "
#                 f"Identity: {dummy_arg is old_dummy_arg}  |  "
#                 f"Last Equality: {dummy_arg == last_dummy_arg}  |  "
#                 f"Last Identity: {dummy_arg is last_dummy_arg}"))
# print()


# @jax.partial(jax.jit, static_argnums=(0,))
# def dummy_add_fn(dummy, x):
#     return x + 1


# fn = dummy_add_fn
# name = fn.__name__
# dummy_arg = [3]
# fixed_arg = jnp.zeros((3,))
# with jax.profiler.TraceContext(name):
#     print(name)
#     for i in range(5):
#         start = time.time()
#         last_dummy_arg = dummy_arg

#         with jax.profiler.TraceContext(f"Round {i}"):
#             _ = fn(dummy_arg, fixed_arg)
#             dummy_arg = [3]

#         end = time.time()
#         print((f"Time {i :3}: {end - start :.4f}s  |  "
#                f"Last Equality: {dummy_arg == last_dummy_arg}  |  "
#                f"Last Identity: {dummy_arg is last_dummy_arg}"))
# print()


@jax.partial(jax.jit, static_argnums=(0,))
def dummy_add_fn(dummy, x):
    return x + 1

dummy_arg = [0]
real_arg = jnp.zeros((3,))
with jax.profiler.TraceContext("Run 1"):
    dummy_add_fn(dummy_arg, real_arg)  # <- JIT compilation
with jax.profiler.TraceContext("Run 2"):
    dummy_add_fn(dummy_arg, real_arg)  # <- no compilation

dummy_arg = [0]
with jax.profiler.TraceContext("Run 3"):
    dummy_add_fn(dummy_arg, real_arg)  # <- compiles again



@jax.partial(jax.jit, static_argnums=(0,))
def dummy_add_fn(dummy, x):
    return x + 1

dummy_arg = jnp.zeros((1,))
real_arg = jnp.zeros((3,))
with jax.profiler.TraceContext("Run 1"):
    dummy_add_fn(dummy_arg, real_arg)  # <- JIT compilation
with jax.profiler.TraceContext("Run 2"):
    dummy_add_fn(dummy_arg, real_arg)  # <- no compilation

dummy_arg = jnp.zeros((1,))
with jax.profiler.TraceContext("Run 3"):
    dummy_add_fn(dummy_arg, real_arg)  # <- compiles again