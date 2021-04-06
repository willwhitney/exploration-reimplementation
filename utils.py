import os
import time
import numpy as np
import matplotlib.pyplot as plt
import collections

from torch.utils import dlpack as tdlpack
from jax import dlpack as jdlpack

import jax
from jax import numpy as jnp, random
from jax.lib import pytree
from dm_env import specs
from environments import jax_specs


def one_hot(x, k, dtype=jnp.int16):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


def tree_stack(trees):
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).

    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = pytree.flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def tree_unstack(tree):
    """Takes a tree and turns it into a list of trees. Inverse of tree_stack.
    For example, given a tree ((a, b), c), where a, b, and c all have first
    dimension k, will make k trees
    [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]

    Useful for turning the output of a vmapped function into normal objects.
    """
    leaves, treedef = pytree.flatten(tree)
    n_trees = leaves[0].shape[0]
    new_leaves = [[] for _ in range(n_trees)]
    for leaf in leaves:
        for i in range(n_trees):
            new_leaves[i].append(leaf[i])
    new_trees = [treedef.unflatten(l) for l in new_leaves]
    return new_trees


def display_figure(fig, rendering, savepath=None):
    if rendering == 'local':
        plt.show(fig)
    elif rendering == 'remote':
        plt.show(fig)
        plt.close(fig)
        time.sleep(3)
    elif rendering == 'disk':
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath)
        plt.close(fig)
    else:
        raise ValueError((f"Value of `{rendering}` for `display_figure`"
                          f"is not valid."))


def display_subfigures(subfigs_with_names, rendering='disk', savepath=None):
    fig, axs = plt.subplots(1, len(subfigs_with_names))

    def make_subfig(ax, subfig):
        render, title = subfig
        img = ax.imshow(render)
        fig.colorbar(img, ax=ax)
        ax.set_title(title)

    if len(subfigs_with_names) > 1:
        for ax, subfig in zip(axs, subfigs_with_names):
            make_subfig(ax, subfig)
    else:
        ax = axs
        make_subfig(ax, subfigs_with_names[0])

    fig.set_size_inches(4 * len(subfigs_with_names), 3)
    display_figure(fig, rendering, savepath=savepath)


def super_flatten(tree):
    """Take a Pytree-structured input and make a flat array.

    Returns the flattened array, the treedef for the Pytree,
    and the shapes of each of the original leaf arrays; this
    allows it to be rehydrated later.
    """
    array_list, treedef = jax.tree_util.tree_flatten(tree)
    shapes = [a.shape for a in array_list]
    flat_arrays = [a.flatten() for a in array_list]
    big_array = jnp.concatenate(flat_arrays, axis=0)
    return big_array, treedef, shapes


def super_unflatten(big_array, treedef, shapes):
    """Take a super_flattened array and structure and rehydrate.

    Arguments should be structured as the output of super_flatten.
    Returns a rehydrated Pytree.
    """
    sizes = [jnp.prod(s) for s in shapes]
    flat_arrays = []
    edges = [0, *jnp.cumsum(jnp.array(sizes))]
    for i in range(len(edges) - 1):
        start = edges[i]
        end = edges[i + 1]
        flat_array = big_array[start:end]
        flat_arrays.append(flat_array)

    arrays = []
    for flat_array, shape in zip(flat_arrays, shapes):
        arrays.append(flat_array.reshape(shape))

    return jax.tree_util.tree_unflatten(treedef, arrays)


def discretize(x, spec, bins):
    """Discretize an input ndarray `x` into `bins` values.

    First rescales `x` to have every value in [0, 1] using the spec min and max.
    Then scales `x` up to [0, bins - 1].
    Finally casts `x` to an integer type.
    """
    x = (x - spec.minimum) / (spec.maximum - spec.minimum)
    x = jnp.floor(x * bins)
    x = jnp.clip(x, 0, bins - 1)
    return x.astype(jnp.uint32)


def normalize(obs_el, spec_el):
    return (obs_el - spec_el.minimum) / (spec_el.maximum - spec_el.minimum)


def flatten_observation(obs, preserve_batch=False):
    flat_tree = jax.tree_leaves(obs)
    if preserve_batch:
        flat_elements = [el.reshape((el.shape[0], -1)) for el in flat_tree]
        return jnp.concatenate(flat_elements, axis=1)
    else:
        flat_elements = [el.flatten() for el in flat_tree]
        return jnp.concatenate(flat_elements, axis=0)


@jax.partial(jax.jit, static_argnums=(2, 3))
def discretize_observation(obs, spec, bins, preserve_batch=False):
    discretize_bins = jax.partial(discretize, bins=bins)
    discrete_tree_obs = jax.tree_multimap(discretize_bins, obs, spec)
    flat_obs = flatten_observation(discrete_tree_obs, preserve_batch)
    return flat_obs


def flatten_spec_shape(spec):
    return (sum([np.prod(v.shape) for v in spec.values()]),)


def flatten_observation_spec(spec):
    """Takes a dm_env spec and flattens it.
    Preserves the minimum and maximum vectors by flattening them as well.
    CANNOT be called with a jax_specs spec!
    """
    flat_tree = jax.tree_leaves(spec)
    assert isinstance(flat_tree[0], specs.BoundedArray)
    mins = jnp.concatenate([el.minimum.flatten() for el in flat_tree])
    maxs = jnp.concatenate([el.maximum.flatten() for el in flat_tree])
    shape = flatten_spec_shape(spec)
    return specs.BoundedArray(shape, dtype=np.float32,
                              minimum=mins, maximum=maxs)


@jax.partial(jax.jit, static_argnums=(2,))
def sample_uniform_actions(action_spec, rng, n):
    aspec_shape = action_spec.minimum.shape
    if len(aspec_shape) > 0:
        shape = (n, *aspec_shape)
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
    return actions.reshape((n, *aspec_shape))

@jax.partial(jax.jit, static_argnums=(2, 3))
def sample_uniform_actions_batch(action_spec, rng, bsize, n):
    total_actions = bsize * n
    actions = sample_uniform_actions(
        action_spec, rng, total_actions)
    actions = actions.reshape((bsize, n, *actions.shape[1:]))
    return actions

# sample_uniform_actions_batch = jax.vmap(sample_uniform_actions,
#                                         in_axes=(None, 0, None))

# @jax.partial(jax.jit, static_argnums=(2,))
def sample_uniform_single(spec, rng, n):
    aspec_shape = spec.minimum.shape
    if len(aspec_shape) > 0:
        shape = (n, *aspec_shape)
    else:
        shape = (n, 1)
    minval = jnp.expand_dims(spec.minimum, axis=0).tile((n, 1))
    maxval = jnp.expand_dims(spec.maximum, axis=0).tile((n, 1))

    if (isinstance(spec, jax_specs.DiscreteArray) or
        isinstance(spec, specs.DiscreteArray)):
        sampler = random.randint
        maxval += 1  # maxval is exclusive for randint but not uniform
    else:
        sampler = random.uniform

    samples = sampler(rng, shape=shape,
                      minval=minval, maxval=maxval)
    return samples.reshape(shape)


def j_to_t(x):
    return tdlpack.from_dlpack(jdlpack.to_dlpack(x))


def t_to_j(x):
    return jdlpack.from_dlpack(tdlpack.to_dlpack(x))



def sample_flat_uniform(spec, rng, n):
    samples = jax.tree_map(jax.partial(sample_uniform_single, rng=rng, n=n),
                           spec)
    return flatten_observation(samples, preserve_batch=True)


def sample_grid(spec, dims, bins):
    xmin, xmax = spec.minimum[dims[0]], spec.maximum[dims[0]]
    ymin, ymax = spec.minimum[dims[1]], spec.maximum[dims[1]]
    grid = np.stack(np.meshgrid(np.linspace(xmin, xmax, bins),
                                np.linspace(ymin, ymax, bins)))
    grid = grid.transpose().reshape(-1, 2)
    return grid


def select_observations(ospec, elements, flat_obs):
    dims = []
    start_dim = 0
    for name, component in ospec.items():
        end_dim = start_dim + np.prod(component.shape)
        if name in elements:
            dims += list(range(start_dim, end_dim))
        start_dim = end_dim
    return flat_obs[..., dims]


@jax.profiler.trace_function
def render_function(fn, replay, ospec, aspec, reduction=np.max, vis_elem=None,
                    vis_dims=(1, 0), bins=20, use_uniform_states=False):
    """Renders a given function at sampled (state, action) pairs.

    Arguments:
    - fn: a function that takes (batch of states, batch of actions) as its
        arguments
    - ospec: an observation spec
    - aspec: an action spec
    - reduction: a function mapping from jnp.ndarray -> float. maps from the
        vector of values for each action at a particular state to a single
        value which will represent that state.
    """
    rng = random.PRNGKey(0)
    j_aspec = jax_specs.convert_dm_spec(aspec)
    n_samples = 2000

    action_shape = aspec.shape
    if len(action_shape) == 0:
        action_shape = (1,)

    with jax.profiler.TraceContext("render sample states"):
        transitions = replay.sample(5 * n_samples)
        states = transitions[0]
        actions = transitions[1]
        actions = actions.reshape((-1, *action_shape))

    with jax.profiler.TraceContext("render sample actions"):
        n_random_actions = 4 * n_samples
        uniform_actions = sample_uniform_actions(j_aspec, rng, n_random_actions)
        actions[:n_random_actions] = uniform_actions

    if use_uniform_states:
        sampled_flat_states = np.array(
            sample_flat_uniform(ospec, rng, n_samples))
        sampled_actions = sample_uniform_single(aspec, rng, n_samples)
        states = jnp.concatenate([sampled_flat_states,
                                  states], axis=0)
        actions = jnp.concatenate([sampled_actions,
                                   actions], axis=0)

    values_list = []
    bsize = 10000
    for i in range(0, states.shape[0], bsize):
        with jax.profiler.TraceContext("render fn"):
            value = fn(states[i: i + bsize], actions[i: i + bsize])
            values_list.append(np.array(value))

    with jax.profiler.TraceContext("render reshape discretize"):
        values = np.concatenate(values_list, axis=0)

        flat_ospec = flatten_observation_spec(ospec)
        discrete_states = np.array(discretize(states, flat_ospec, bins))
        discrete_actions = np.array(discretize(actions, aspec, bins))

        if vis_elem is not None:
            discrete_states = select_observations(ospec, vis_elem,
                                                  discrete_states)

        xs = np.concatenate([discrete_states, discrete_actions], axis=1)

    with jax.profiler.TraceContext("render value lists"):
        value_lists = [[[] for _ in range(bins)] for _ in range(bins)]
        for (x, value) in zip(xs, values):
            i, j = x[vis_dims[0]], x[vis_dims[1]]
            value_lists[i][j].append(value)

    with jax.profiler.TraceContext("render image"):
        rendered_values = np.zeros((bins, bins))
        for i in range(bins):
            for j in range(bins):
                l = value_lists[i][j]
                if len(l) > 0:
                    with jax.profiler.TraceContext("render assign reduction"):
                        flip_i = (bins - 1) - i
                        rendered_values[flip_i, j] = reduction(np.stack(l))
    return rendered_values


if __name__ == "__main__":
    def make_tree():
        sizes = ((1, 2), (3, 1), (3,))
        make_leaf_np = lambda i: np.random.uniform(size=sizes[i])
        make_leaf = lambda i: jnp.array(make_leaf_np(i))
        return ((make_leaf(0), make_leaf(1)), make_leaf(2))
    trees = [make_tree() for _ in range(3)]

    print("Before")
    print(trees)

    print("\nStacked")
    stacked = tree_stack(trees)
    print(stacked)

    print("\nUnstacked")
    unstacked = tree_unstack(stacked)
    print(unstacked)
