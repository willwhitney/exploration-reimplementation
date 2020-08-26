import os
import time
import numpy as np
import matplotlib.pyplot as plt

from jax import numpy as jnp
from jax.lib import pytree


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
