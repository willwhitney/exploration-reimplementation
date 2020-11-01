import time
import jax
from jax import numpy as jnp, random
from flax import struct

from dm_env import specs
from dm_control import suite
import jax_specs
import utils


fixed_key = random.PRNGKey(0)


@jax.partial(jax.jit, static_argnums=(0,))
def sample_uniform_actions(action_spec):
    rng = random.PRNGKey(0)
    n = 32
    if len(action_spec.shape) > 0:
        shape = (n, *action_spec.shape)
    else:
        shape = (n, 1)
    minval = jnp.expand_dims(action_spec.minimum, axis=0).tile((n, 1))
    maxval = jnp.expand_dims(action_spec.maximum, axis=0).tile((n, 1))

    # if jnp.issubdtype(action_spec.dtype, jnp.integer):
    if (isinstance(action_spec, jax_specs.DiscreteArray) or
        isinstance(action_spec, specs.DiscreteArray)):
        sampler = random.randint
        maxval += 1  # maxval is exclusive for randint but not uniform
    else:
        sampler = random.uniform

    actions = sampler(rng, shape=shape, # dtype=action_spec.dtype,
                      minval=minval, maxval=maxval)
    return action_spec, actions.reshape((n, *action_spec.shape))


@jax.partial(jax.jit, static_argnums=(0, 1))
def sample_uniform_actions_n(action_spec, n):
    rng = random.PRNGKey(0)
    n = 32
    if len(action_spec.shape) > 0:
        shape = (n, *action_spec.shape)
    else:
        shape = (n, 1)
    minval = jnp.expand_dims(action_spec.minimum, axis=0).tile((n, 1))
    maxval = jnp.expand_dims(action_spec.maximum, axis=0).tile((n, 1))

    # if jnp.issubdtype(action_spec.dtype, jnp.integer):
    if (isinstance(action_spec, jax_specs.DiscreteArray) or
        isinstance(action_spec, specs.DiscreteArray)):
        sampler = random.randint
        maxval += 1  # maxval is exclusive for randint but not uniform
    else:
        sampler = random.uniform

    actions = sampler(rng, shape=shape, # dtype=action_spec.dtype,
                      minval=minval, maxval=maxval)
    return action_spec, actions.reshape((n, *action_spec.shape))


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def fspec_sample(action_spec: jax_specs.BoundedArray):
    candidate_actions = sample_uniform_actions(action_spec)
    return action_spec, candidate_actions


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def fspec_sample_n(action_spec: jax_specs.BoundedArray):
    candidate_actions = sample_uniform_actions_n(action_spec, 32 * 1)
    return action_spec, candidate_actions


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def fspec_sample_orig(action_spec: jax_specs.BoundedArray):
    candidate_actions = utils.sample_uniform_actions(
        action_spec, random.PRNGKey(0),
        32 * 1)
    return action_spec, candidate_actions


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0, 2))
def sample_uniform_actions_dup(action_spec, rng, n):
    if len(action_spec.shape) > 0:
        shape = (n, *action_spec.shape)
    else:
        shape = (n, 1)
    minval = jnp.expand_dims(action_spec.minimum, axis=0).tile((n, 1))
    maxval = jnp.expand_dims(action_spec.maximum, axis=0).tile((n, 1))

    # if jnp.issubdtype(action_spec.dtype, jnp.integer):
    if (isinstance(action_spec, jax_specs.DiscreteArray) or
        isinstance(action_spec, specs.DiscreteArray)):
        sampler = random.randint
        maxval += 1  # maxval is exclusive for randint but not uniform
    else:
        sampler = random.uniform

    actions = sampler(rng, shape=shape, # dtype=action_spec.dtype,
                      minval=minval, maxval=maxval)
    return actions.reshape((n, *action_spec.shape))


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def fspec_sample_dup(action_spec: jax_specs.BoundedArray):
    candidate_actions = sample_uniform_actions_dup(
        action_spec, random.PRNGKey(0),
        32 * 1)
    return action_spec, candidate_actions


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def sample_uniform_actions_dup_no_n(action_spec, rng):
    n = 32 * 1
    if len(action_spec.shape) > 0:
        shape = (n, *action_spec.shape)
    else:
        shape = (n, 1)
    minval = jnp.expand_dims(action_spec.minimum, axis=0).tile((n, 1))
    maxval = jnp.expand_dims(action_spec.maximum, axis=0).tile((n, 1))

    # if jnp.issubdtype(action_spec.dtype, jnp.integer):
    if (isinstance(action_spec, jax_specs.DiscreteArray) or
        isinstance(action_spec, specs.DiscreteArray)):
        sampler = random.randint
        maxval += 1  # maxval is exclusive for randint but not uniform
    else:
        sampler = random.uniform

    actions = sampler(rng, shape=shape, # dtype=action_spec.dtype,
                      minval=minval, maxval=maxval)
    return actions.reshape((n, *action_spec.shape))


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def fspec_sample_dup_no_n(action_spec: jax_specs.BoundedArray):
    candidate_actions = sample_uniform_actions_dup_no_n(
        action_spec, random.PRNGKey(0))
    return action_spec, candidate_actions


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def sample_uniform_actions_dup_no_n_no_rng(action_spec):
    rng = random.PRNGKey(0)
    n = 32 * 1
    if len(action_spec.shape) > 0:
        shape = (n, *action_spec.shape)
    else:
        shape = (n, 1)
    minval = jnp.expand_dims(action_spec.minimum, axis=0).tile((n, 1))
    maxval = jnp.expand_dims(action_spec.maximum, axis=0).tile((n, 1))

    # if jnp.issubdtype(action_spec.dtype, jnp.integer):
    if (isinstance(action_spec, jax_specs.DiscreteArray) or
        isinstance(action_spec, specs.DiscreteArray)):
        sampler = random.randint
        maxval += 1  # maxval is exclusive for randint but not uniform
    else:
        sampler = random.uniform

    actions = sampler(rng, shape=shape, # dtype=action_spec.dtype,
                      minval=minval, maxval=maxval)
    return actions.reshape((n, *action_spec.shape))


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def fspec_sample_dup_no_n_no_rng(action_spec: jax_specs.BoundedArray):
    candidate_actions = sample_uniform_actions_dup_no_n_no_rng(action_spec)
    return action_spec, candidate_actions


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0, 1))
def sample_uniform_actions_dup_no_rng(action_spec, n):
    rng = random.PRNGKey(0)
    if len(action_spec.shape) > 0:
        shape = (n, *action_spec.shape)
    else:
        shape = (n, 1)
    minval = jnp.expand_dims(action_spec.minimum, axis=0).tile((n, 1))
    maxval = jnp.expand_dims(action_spec.maximum, axis=0).tile((n, 1))

    # if jnp.issubdtype(action_spec.dtype, jnp.integer):
    if (isinstance(action_spec, jax_specs.DiscreteArray) or
        isinstance(action_spec, specs.DiscreteArray)):
        sampler = random.randint
        maxval += 1  # maxval is exclusive for randint but not uniform
    else:
        sampler = random.uniform

    actions = sampler(rng, shape=shape, # dtype=action_spec.dtype,
                      minval=minval, maxval=maxval)
    return actions.reshape((n, *action_spec.shape))


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def fspec_sample_dup_no_rng(action_spec: jax_specs.BoundedArray):
    candidate_actions = sample_uniform_actions_dup_no_rng(
        action_spec, 32 * 1)
    return action_spec, candidate_actions


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0, 2))
def sample_uniform_actions_simple(action_spec, rng, n):
    shape = (n, 2)
    minval = jnp.expand_dims(action_spec.minimum, axis=0).tile((n, 1))
    maxval = jnp.expand_dims(action_spec.maximum, axis=0).tile((n, 1))

    actions = random.uniform(rng, shape=shape,
                             minval=minval, maxval=maxval)
    return actions


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def fspec_sample_simple(action_spec: jax_specs.BoundedArray):
    candidate_actions = sample_uniform_actions_simple(
        action_spec, random.PRNGKey(0),
        32 * 1)
    return action_spec, candidate_actions


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0, 2))
def identity(action_spec, rng, n):
    return action_spec, rng, n


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def fspec_identity(action_spec: jax_specs.BoundedArray):
    rng = random.PRNGKey(0)
    n = 32
    action_spec, rng, n = identity(action_spec, rng, n)
    return action_spec, rng + n


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
        action_spec, fixed_key)
    return action_spec, candidate_actions


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0, 1))
def sample_uniform_actions_simple_no_rng(action_spec, n):
    shape = (n, 2)
    minval = jnp.expand_dims(action_spec.minimum, axis=0).tile((n, 1))
    maxval = jnp.expand_dims(action_spec.maximum, axis=0).tile((n, 1))

    actions = random.uniform(random.PRNGKey(0), shape=shape,
                             minval=minval, maxval=maxval)
    return actions


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def fspec_sample_simple_no_rng(action_spec: jax_specs.BoundedArray):
    candidate_actions = sample_uniform_actions_simple_no_rng(
        action_spec, 32 * 1)
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


# @jax.profiler.trace_function
# def fspec_sample_simple_no_n_no_rng_nofjit(action_spec: jax_specs.BoundedArray):
#     candidate_actions = sample_uniform_actions_simple_no_n_no_rng(action_spec)
#     return action_spec, candidate_actions

@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def sample_fixed_rng(action_spec):
    n = 32
    shape = (n, 2)
    minval = jnp.expand_dims(action_spec.minimum, axis=0).tile((n, 1))
    maxval = jnp.expand_dims(action_spec.maximum, axis=0).tile((n, 1))
    actions = random.uniform(fixed_key, shape=shape,
                             minval=minval, maxval=maxval)
    return action_spec, actions


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def sample_no_rng(action_spec):
    n = 32
    shape = (n, 2)
    minval = jnp.expand_dims(action_spec.minimum, axis=0).tile((n, 1))
    maxval = jnp.expand_dims(action_spec.maximum, axis=0).tile((n, 1))
    actions = random.uniform(random.PRNGKey(0), shape=shape,
                             minval=minval, maxval=maxval)
    return action_spec, actions


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def sample_fixed_rng_dummy(action_spec, dummy):
    n = 32
    shape = (n, 2)
    minval = jnp.expand_dims(action_spec.minimum, axis=0).tile((n, 1))
    maxval = jnp.expand_dims(action_spec.maximum, axis=0).tile((n, 1))
    actions = random.uniform(fixed_key, shape=shape,
                             minval=minval, maxval=maxval)
    return action_spec, actions + dummy


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def sample_no_rng_dummy(action_spec, dummy):
    n = 32
    shape = (n, 2)
    minval = jnp.expand_dims(action_spec.minimum, axis=0).tile((n, 1))
    maxval = jnp.expand_dims(action_spec.maximum, axis=0).tile((n, 1))
    actions = random.uniform(random.PRNGKey(0), shape=shape,
                             minval=minval, maxval=maxval)
    return action_spec, actions + dummy

@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def f_sample_fixed_rng(action_spec):
    return sample_fixed_rng(action_spec)

@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def f_sample_no_rng(action_spec):
    return sample_no_rng(action_spec)

@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def f_sample_fixed_rng_dummy(action_spec):
    return sample_fixed_rng_dummy(action_spec, jnp.zeros(1))

@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def f_sample_no_rng_dummy(action_spec):
    return sample_no_rng_dummy(action_spec, jnp.zeros(1))


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
            print((f"Time {i :3}: {end - start :.2f}s  |  "
                   f"Equality: {action_spec == old_action_spec}  |  "
                   f"Identity: {action_spec is old_action_spec}  |  "
                   f"Last Equality: {action_spec == last_action_spec}  |  "
                   f"Last Identity: {action_spec is last_action_spec}"))
    print()


jobs = [
    #---- good ----
    # sample_uniform_actions,
    # fspec_sample,
    # fspec_sample_n,
    # fspec_sample_dup_no_n_no_rng,
    # fspec_sample_dup_no_rng,
    # fspec_identity,
    # fspec_sample_simple_no_rng,
    # fspec_sample_simple_no_n_no_rng,
    sample_no_rng,
    f_sample_no_rng,



    #---- bad ----
    # fspec_sample_orig,
    # fspec_sample_dup,
    # fspec_sample_dup_no_n,
    # fspec_sample_simple,
    # fspec_sample_simple_no_n,
    # fspec_sample_simple_no_n_fixed_rng,
    f_sample_no_rng_dummy,
    sample_fixed_rng,
    f_sample_fixed_rng,
    f_sample_fixed_rng_dummy,
]


for fn in jobs:
    test_fn(fn)
