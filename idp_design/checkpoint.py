"""Control flow functions."""
import functools

import jax
import jax.numpy as jnp
from jax import lax


def _split_n_stack(x, n):
    """Splits `x` into `n` parts along axis 0 and stackes the resulting arrays."""
    return jax.tree_map(lambda y: jnp.stack(jnp.split(y, n)), x)


def _flatten_n(x, n):
    """Flattens the first `n` dimensions of `xs`."""
    return jax.tree_map(lambda y: jnp.reshape(y, (-1,) + y.shape[n:]), x)


def checkpoint_scan(f, init, xs, checkpoint_every):
    """Replicates `lax.scan` but checkpoints grads every `checkpoint_every` steps."""
    flat_xs, _ = jax.tree_util.tree_flatten(xs)
    length = flat_xs[0].shape[0]
    outer_iterations, residual = divmod(length, checkpoint_every)
    if residual:
        raise ValueError('`checkpoint_every` must evenly divide the length of `xs`. '
                         f'Got {checkpoint_every} and {length}.')
    reshaped_xs = _split_n_stack(xs, outer_iterations)

    @jax.checkpoint
    def inner_loop(_init, _xs):
        return lax.scan(f, _init, _xs)

    final, result = lax.scan(inner_loop, init, reshaped_xs)
    return final, _flatten_n(result, 2)


def get_scan(checkpoint_every=None):
    """Wraps `checkpoint_scan` to allow no checkpointing."""
    if checkpoint_every is None:
        scan = lax.scan
    else:
        scan = functools.partial(checkpoint_scan, checkpoint_every=checkpoint_every)
    return scan
