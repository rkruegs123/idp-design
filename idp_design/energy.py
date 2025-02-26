
from typing import Callable, Tuple
import jax
import jax.numpy as jnp
from jax import vmap
from jax_md import space

import idp_design.utils as utils

jax.config.update("jax_enable_x64", True)




def get_energy_fn(
    bonded_nbrs: jnp.ndarray,
    base_unbonded_nbrs: jnp.ndarray,
    displacement_fn: Callable,
    use_gg: bool = True
) -> Tuple[Callable, Callable]:
    """
    Generates energy functions for bonded and unbonded interactions.

    This function constructs two energy functions:

    - `subterms_fn`: Computes individual energy contributions, including
      total bonded, total unbonded, Wang-Frenkel, and Coulomb interactions.
    - `energy_fn`: Computes the total energy of the system.

    The function supports two parameter sets (`use_gg=True/False`), which determine
    the values used for Debye screening and Wang-Frenkel potentials.


    Args:
      bonded_nbrs: A `(m, 2)` JAX array specifying `m` bonded neighbor pairs.
      base_unbonded_nbrs: A `(p, 2)` JAX array specifying `p` unbonded neighbor pairs.
      displacement_fn: A function that computes displacement vectors between two positions,
        following JAX-MD conventions.
      use_gg: Whether to use the Mpipi-GG force field (`True`) or standard Mpipi force field (`False`).
        Defaults to `True`.

    Returns:
      A tuple of two functions
        - `subterms_fn`, which takes `(R, seq, unbonded_nbrs)` as input and computes total energy and individual energy terms
        - `energy_fn`, which takes `(R, seq, unbonded_nbrs)` as input and computes only the total energy.

    Example:
        >>> bonded_nbrs = jnp.array([[0, 1], [1, 2]])
        >>> unbonded_nbrs = jnp.array([[0, 2]])
        >>> disp_fn = lambda r1, r2: r2 - r1  # Simple Euclidean displacement
        >>> subterms_fn, energy_fn = get_energy_fn(bonded_nbrs, unbonded_nbrs, disp_fn)
        >>> R = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        >>> seq = jnp.array([0, 1, 2])  # Example amino acid indices
        >>> total_energy = energy_fn(R, seq)
    """
    if use_gg:
        debye_kappa = utils.DEBYE_KAPPA_GG
        debye_path = utils.DEBYE_GG_PATH
        wf_path = utils.WF_GG_PATH
    else:
        debye_kappa = utils.DEBYE_KAPPA
        debye_path = utils.DEBYE_PATH
        wf_path = utils.WF_PATH

    eps_table, sigma_table, nu_table, mu_table, rc_table = utils._read_wf(fpath=wf_path)
    cutoff_table = utils._read_debye(fpath=debye_path)

    # spring_k = 8.03 * 10 / 4.184 # FIXME: double check
    spring_k = 9.6 * 2

    def subterms_fn(R, seq, unbonded_nbrs=base_unbonded_nbrs):

        def pairwise_unbonded(i, j):
            ipos = R[i]
            jpos = R[j]

            r = space.distance(displacement_fn(ipos, jpos))

            ires = seq[i]
            jres = seq[j]

            wf_val = utils._wang_frenkel(
                r,
                r_c=rc_table[ires, jres],
                sigma=sigma_table[ires, jres],
                nu=nu_table[ires, jres],
                mu=mu_table[ires, jres],
                eps=eps_table[ires, jres]
            )
            coul_val = utils._coul(
                r,
                qij=jnp.array([utils.charges[ires], utils.charges[jres]]),
                eps=utils.debye_relative_dielectric,
                k=debye_kappa,
                r_c=cutoff_table[ires, jres]
            )

            val = wf_val + coul_val
            # val = jnp.where(r > 35, 0.0, val)
            return val, (wf_val, coul_val)

        def pairwise_bonded(i, j):
            ipos = R[i]
            jpos = R[j]

            r = space.distance(displacement_fn(ipos, jpos))

            return utils._harmonic_spring(r, r0=utils.spring_r0, k=spring_k)

        bnd_i = bonded_nbrs[:, 0]
        bnd_j = bonded_nbrs[:, 1]
        total_bonded_val = jnp.sum(vmap(pairwise_bonded)(bnd_i, bnd_j))
        ub_i = unbonded_nbrs[:, 0]
        ub_j = unbonded_nbrs[:, 1]
        all_unbonded_vals, (wf_val, coul_val) = vmap(pairwise_unbonded)(ub_i, ub_j)
        total_unbonded_val = jnp.sum(all_unbonded_vals)
        total_wf_val = jnp.sum(wf_val)
        total_coul_val = jnp.sum(coul_val)

        total_energy = total_bonded_val + total_unbonded_val
        aux = (total_bonded_val, total_unbonded_val, total_wf_val, total_coul_val)
        return total_energy, aux

    def energy_fn(R, seq, unbonded_nbrs=base_unbonded_nbrs):
        total_energy, aux = subterms_fn(R, seq, unbonded_nbrs)
        (total_bonded_val, total_unbonded_val, total_wf_val, total_coul_val) = aux
        return total_energy

    return subterms_fn, energy_fn
