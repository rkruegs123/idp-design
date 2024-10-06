import jax.numpy as jnp
from jax import vmap

from jax_md import space


def compute_com(R, mass):
    com = jnp.sum(jnp.multiply(R, jnp.expand_dims(mass, axis=1)), axis=0) / jnp.sum(mass) # note: assumes free boundary conditions
    return com

def rg(R, mass, displacement_fn):
    # com = jnp.sum(jnp.multiply(R, jnp.expand_dims(mass, axis=1)), axis=0) / jnp.sum(mass) # note: assumes free boundary conditions
    com = compute_com(R, mass)
    drs = vmap(displacement_fn, (None, 0))(com, R)
    rs = space.distance(drs)

    I = jnp.sum(mass * rs**2)
    M = jnp.sum(mass)
    rg = jnp.sqrt(I/M)

    return rg


def end_to_end_dist(R, displacement_fn):
    e2e_dist = space.distance(displacement_fn(R[0], R[-1]))
    return e2e_dist
