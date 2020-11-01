import time
import jax
from jax import numpy as jnp, random
from flax import struct


from dm_control import suite
import jax_specs
import utils


@struct.dataclass
class Dummy():
    x: jnp.ndarray
    
    def __eq__(self, other):
        return jnp.allclose(self.x, other.x)
    
    
@struct.dataclass
class DummySpec():
    aspec: jax_specs.BoundedArray
    
    def __eq__(self, other):
        return self.aspec == other.aspec
    

@jax.profiler.trace_function  
@jax.partial(jax.jit, static_argnums=(0,))
def f(d: Dummy):
    m = jnp.ones((100, 100))
    return d, m @ d.x


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def fspec(aspec: jax_specs.BoundedArray):
    x = aspec.minimum
    m = jnp.ones((aspec.shape[0], aspec.shape[0]))
    return aspec, m @ x


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def fspec_sample(aspec: jax_specs.BoundedArray):
    candidate_actions = utils.sample_uniform_actions(
        action_spec, random.PRNGKey(0),
        32 * 1)
    return aspec, candidate_actions


@jax.profiler.trace_function
@jax.partial(jax.jit, static_argnums=(0,))
def f_dummyspec(d: DummySpec):
    aspec = d.aspec
    x = aspec.minimum
    m = jnp.ones((aspec.shape[0], aspec.shape[0]))
    return d, m @ x



# d = Dummy(jnp.ones((100,)))
# old_d = d
# with jax.profiler.TraceContext("Dummy function"):
#     print("Dummy function")
#     for i in range(5):
#         start = time.time()
#         print(f"Equality: {d == old_d}  |  Identity: {d is old_d}")
        
#         with jax.profiler.TraceContext(f"Round {i}"):
#             d, _ = f(d)
        
#         end = time.time()
#         print(f"Time for f {i :3}: {end - start :.2f}s")


# env = suite.load("point_mass", "easy")
# action_spec = env.action_spec()
# action_spec = jax_specs.convert_dm_spec(action_spec)
# old_aspec = action_spec
# with jax.profiler.TraceContext("Aspec function"):
#     print()
#     print("Aspec function")
#     for i in range(5):
#         start = time.time()
#         print((f"Equality: {action_spec == old_aspec}  |  "
#             f"Identity: {action_spec is old_aspec}"))
#         last_aspec = action_spec
        
#         with jax.profiler.TraceContext(f"Round {i}"):
#             action_spec, _ = fspec(action_spec)
            
#         print((f"Last Equality: {action_spec == last_aspec}  |  "
#             f"Last Identity: {action_spec is last_aspec}"))

#         # import ipdb; ipdb.set_trace()
#         end = time.time()
#         print(f"Time for fspec {i :3}: {end - start :.2f}s")


env = suite.load("point_mass", "easy")
action_spec = env.action_spec()
action_spec = jax_specs.convert_dm_spec(action_spec)
old_aspec = action_spec
with jax.profiler.TraceContext("Aspec sample function"):
    print()
    print("Aspec sample function")
    for i in range(5):
        start = time.time()
        last_aspec = action_spec
        print((f"Equality: {action_spec == old_aspec}  |  "
            f"Identity: {action_spec is old_aspec}"))
        
        with jax.profiler.TraceContext(f"Round {i}"):
            action_spec, candidate_actions = fspec_sample(action_spec)
            
        print((f"Last Equality: {action_spec == last_aspec}  |  "
            f"Last Identity: {action_spec is last_aspec}"))

        # import ipdb; ipdb.set_trace()
        end = time.time()
        print(f"Time for fspec_sample {i :3}: {end - start :.2f}s")


env = suite.load("point_mass", "easy")
action_spec = env.action_spec()
action_spec = jax_specs.convert_dm_spec(action_spec)
old_aspec = action_spec
dummy_spec = DummySpec(old_aspec)
old_dummy_spec = dummy_spec
with jax.profiler.TraceContext("Aspec dummy function"):
    print()
    print("Aspec dummy function")
    for i in range(5):
        start = time.time()
        print((f"Equality: {dummy_spec == old_dummy_spec}  |  "
            f"Identity: {dummy_spec is old_dummy_spec}"))
        
        with jax.profiler.TraceContext(f"Round {i}"):
            dummy_spec, _ = f_dummyspec(dummy_spec)
            
        end = time.time()
        print(f"Time for f_dummyspec {i :3}: {end - start :.2f}s")


