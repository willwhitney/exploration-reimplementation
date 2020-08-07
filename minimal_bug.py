from typing import Any
import jax
from jax import numpy as jnp

from flax import nn, struct


class DenseNetwork(nn.Module):
    def apply(self, x):
        for layer in range(1):
            x = nn.Dense(x, 10, name=f'fc{layer}')
            x = nn.relu(x)
        x = nn.Dense(x, 1, name=f'fc_out')
        return x


@struct.dataclass
class ModelContainer():
    model: Any = struct.field(pytree_node=False)


_, initial_params = DenseNetwork.init_by_shape(
    jax.random.PRNGKey(0),
    [((1, 1,), jnp.float32)])
initial_model = nn.Model(DenseNetwork, initial_params)


def apply_fn(model, x):
    values = model(x)
    return model, values
jit_apply_fn = jax.jit(apply_fn)


def other_fn(model_container, x):
    values = model_container.model(x)
    return model, values
jit_other_fn = jax.jit(other_fn)


model = initial_model
print("Equality before use:", model == initial_model)

model, value = apply_fn(model, jnp.ones((128, 1)))
print("Equality after use:", model == initial_model)

model_container = ModelContainer(model)
jit_other_fn(model_container, jnp.ones((128, 1)))

model, value = jit_apply_fn(model, jnp.ones((128, 1)))
# print("Equality after JITted use:", model == initial_model)

model_container = model_container.replace(model=model)
jit_other_fn(model_container, jnp.ones((128, 1)))
