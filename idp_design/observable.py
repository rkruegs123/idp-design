import jax.numpy as jnp
from jax import vmap
from jax_md import space


def com(R: jnp.ndarray, mass: jnp.ndarray) -> jnp.ndarray:
    """Computes the center of mass (COM) for a set of weighted positions.

    The center of mass is calculated as:

    .. math::

        \\text{COM} = \\frac{\\sum_i m_i \\mathbf{R}_i}{\\sum_i m_i}

    where :math:`m_i` are the masses, and :math:`\\mathbf{R}_i` are the corresponding
    positions.

    :param jnp.ndarray R: A ``(n, 3)`` JAX array representing ``n`` positions in
      3D space.
    :param jnp.ndarray mass: A ``(n,)`` JAX array containing the mass of each particle.

    :returns: The computed center of mass as a ``(3,)`` JAX array.
    :rtype: jnp.ndarray

    Example:
        >>> R = jnp.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        >>> mass = jnp.array([1.0, 2.0])
        >>> com(R, mass)
        Array([0.6666667, 1.3333334, 2. ], dtype=float32)
    """
    weighted_sum = jnp.sum(jnp.multiply(R, jnp.expand_dims(mass, axis=1)), axis=0)
    com_ = weighted_sum / jnp.sum(mass)
    return com_


def rg(R, mass, displacement_fn):
    """Computes the radius of gyration (Rg) for a set of positions and masses.

    The radius of gyration is a measure of the spatial distribution of a system
    relative to its center of mass. It is computed as:

    .. math::

        \\text{Rg} = \\sqrt{\\sum_i{m_i r_i^2 / M}}

    where `m_i` are the masses, `r_i` are the distances from the center of mass,
    and `M` is the total mass.

    Args:
        R (jnp.ndarray): A `(n, 3)` JAX array representing `n` positions in 3D space.
        mass (jnp.ndarray): A `(n,)` JAX array containing the mass of each particle.
        displacement_fn (Callable): A function that computes displacement vectors
            given two sets of positions, following JAX-MD's conventions.

    Returns:
        jnp.ndarray: A scalar JAX array representing the computed radius of gyration.

    Example:
        >>> R = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        >>> mass = jnp.array([1.0, 1.0])
        >>> displacement_fn = lambda r1, r2: r2 - r1  # Simple Euclidean displacement
        >>> rg(R, mass, displacement_fn)
        Array(0.5, dtype=float32)
    """
    com_ = com(R, mass)
    drs = vmap(displacement_fn, (None, 0))(com_, R)
    rs = space.distance(drs)

    inertia = jnp.sum(mass * rs**2)
    M = jnp.sum(mass)
    rg = jnp.sqrt(inertia / M)

    return rg


def end_to_end_dist(R, displacement_fn):
    """Computes the end-to-end distance of a structure.

    The end-to-end distance is defined as the Euclidean distance between the first
    and last positions in `R`, considering periodic boundary conditions if applicable.

    Args:
        R (jnp.ndarray): A `(n, 3)` JAX array representing `n` positions in 3D space.
        displacement_fn (Callable): A function that computes displacement vectors
            between two positions, following JAX-MD's conventions.

    Returns:
        jnp.ndarray: A scalar JAX array representing the computed end-to-end distance.

    Example:
        >>> R = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
        >>> displacement_fn = lambda r1, r2: r2 - r1  # Simple Euclidean displacement
        >>> end_to_end_dist(R, displacement_fn)
        Array(5.196152, dtype=float32)  # sqrt(3^2 + 3^2 + 3^2)
    """
    e2e_dist = space.distance(displacement_fn(R[0], R[-1]))
    return e2e_dist
