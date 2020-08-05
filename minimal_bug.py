import jax
from jax import numpy as jnp

from flax import nn


class DenseNetwork(nn.Module):
    def apply(self, x):
        for layer in range(1):
            x = nn.Dense(x, 10, name=f'fc{layer}')
            x = nn.relu(x)
        x = nn.Dense(x, 1, name=f'fc_out')
        return x


_, initial_params = DenseNetwork.init_by_shape(
    jax.random.PRNGKey(0),
    [((1, 1,), jnp.float32)])
initial_model = nn.Model(DenseNetwork, initial_params)


def apply_fn(model, x):
    values = model(x)
    return model, values


jit_apply_fn = jax.jit(apply_fn)


model = initial_model
print("Equality before use:", model == initial_model)

model, value = apply_fn(model, jnp.ones((128, 1)))
print("Equality after use:", model == initial_model)

model, value = jit_apply_fn(model, jnp.ones((128, 1)))
print("Equality after JITted use:", model == initial_model)
