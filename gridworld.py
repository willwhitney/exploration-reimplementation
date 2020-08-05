import types

import jax
from jax import numpy as jnp

from flax import struct

import utils


DTYPE = jnp.int16
SIZE = 10
one_hot_10 = jax.partial(utils.one_hot, k=SIZE)

ACTION_MAP = jnp.stack([
    jnp.array((1, 0), dtype=DTYPE),   # visually DOWN
    jnp.array((0, 1), dtype=DTYPE),   # visually RIGHT
    jnp.array((-1, 0), dtype=DTYPE),  # visually UP
    jnp.array((0, -1), dtype=DTYPE),  # visually LEFT
])


@struct.dataclass
class GridWorld:
    size: int
    render: types.FunctionType = struct.field(pytree_node=False)
    agent: jnp.ndarray = jnp.array((0, 0), dtype=DTYPE)
    actions: jnp.ndarray = jnp.arange(4)


def goal(s):
    return jnp.array((s - 1, s - 1), dtype=DTYPE)


def new(size, render_onehot=True):
    if render_onehot:
        _render = jax.partial(utils.one_hot, k=size)
    else:
        _render = lambda x: x
    return GridWorld(size, _render)


def new_batch(n, size):
    return utils.tree_stack(new(size) for _ in range(n))


def reset(gridworld):
    return gridworld.replace(agent=jnp.array((0, 0), dtype=DTYPE))
reset_batch = jax.vmap(reset)  # noqa: E305


def render(gridworld):
    return gridworld.render(gridworld.agent)
render_batch = jax.vmap(render)  # noqa: E305


def step(gridworld, action):
    new_agent = gridworld.agent + ACTION_MAP[action]
    new_agent = jnp.clip(new_agent, 0, gridworld.size - 1)
    gridworld = gridworld.replace(agent=new_agent)
    reward = (gridworld.agent == goal(gridworld.size)).all()
    return gridworld, render(gridworld), reward
step = jax.jit(step, static_argnums=(1,))  # noqa: E305
step_batch = jax.vmap(step)


def all_coords(size):
    grid = jnp.stack(jnp.meshgrid(jnp.linspace(0, size - 1, size),
                                  jnp.linspace(0, size - 1, size)))
    return grid.transpose().reshape(-1, 2).astype(DTYPE)


if __name__ == "__main__":
    import random
    gw = new(10)
    for i in range(10):
        gw, obs, r = step(gw, random.randint(0, 3))
        print(obs)

    gws = new_batch(3, 10)
    print(gws)
    gws, obss, rs = step_batch(gws, jnp.array(range(3)))
