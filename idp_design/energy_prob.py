import jax
import jax.numpy as jnp
from jax import vmap
from jax_md import space

import idp_design.utils as utils

jax.config.update("jax_enable_x64", True)


mapped_wang_frenkel = vmap(utils._wang_frenkel, in_axes=(None, 0, 0, 0, 0 ,0))
mapped_coul = vmap(utils._coul, in_axes=(None, 0, None, None, 0))

def get_energy_fn(bonded_nbrs, base_unbonded_nbrs, displacement_fn, use_gg=True):
    """Generates energy functions for a probabilistic sequence representation.

    This function constructs two energy functions:
    - `subterms_fn`: Computes individual energy contributions, including
      total bonded, total unbonded, Wang-Frenkel, and Coulomb interactions
      based on a **probabilistic sequence** representation.
    - `energy_fn`: Computes the total expected energy of the system.

    Unlike standard energy functions for discrete sequences, this function
    computes the **expected energy** by marginalizing over all possible
    amino acid pairings at each position using the probability matrix `pseq`.

    The function supports two parameter sets (`use_gg=True/False`), which determine
    the values used for Debye screening and Wang-Frenkel potentials.

    Args:
        bonded_nbrs (jnp.ndarray): A `(m, 2)` JAX array specifying `m` bonded
            neighbor pairs.
        base_unbonded_nbrs (jnp.ndarray): A `(p, 2)` JAX array specifying `p`
            unbonded neighbor pairs.
        displacement_fn (Callable): A function that computes displacement vectors
            between two positions, following JAX-MD conventions.
        use_gg (bool, optional): If `True`, uses the Mpipi-GG force field instead
            of the standard Mpipi force field, affecting the Debye and Wang-Frenkel
            parameters. Defaults to `True`.

    Returns:
        tuple:
            - subterms_fn (Callable): A function
              `subterms_fn(R, pseq, unbonded_nbrs, debye_kappa)` that computes
              total expected energy and individual energy terms.
            - energy_fn (Callable): A function
              `energy_fn(R, pseq, unbonded_nbrs, debye_kappa)` that computes only
              the total expected energy.

    Example:
        >>> bonded_nbrs = jnp.array([[0, 1], [1, 2]])
        >>> unbonded_nbrs = jnp.array([[0, 2]])
        >>> disp_fn = lambda r1, r2: r2 - r1  # Simple Euclidean displacement
        >>> subterms_fn, energy_fn = get_energy_fn(bonded_nbrs, unbonded_nbrs, disp_fn)
        >>> R = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        >>> pseq = jnp.array([
        ...     [0.5, 0.5] + [0.0] * 18,  # 50% 'M', 50% 'G' at position 0
        ...     [0.0, 1.0] + [0.0] * 18,  # 100% 'G' at position 1
        ...     [0.3, 0.2, 0.5] + [0.0] * 17  # Mixed distribution at position 2
        ... ])
        >>> total_energy = energy_fn(R, pseq)
    """
    if use_gg:
        default_debye_kappa = utils.DEBYE_KAPPA_GG
        debye_path = utils.DEBYE_GG_PATH
        wf_path = utils.WF_GG_PATH
    else:
        default_debye_kappa = utils.DEBYE_KAPPA
        debye_path = utils.DEBYE_PATH
        wf_path = utils.WF_PATH

    eps_table, sigma_table, nu_table, mu_table, rc_table = utils._read_wf(fpath=wf_path)
    debye_rc_table = utils._read_debye(fpath=debye_path)

    # spring_k = 8.03 * 10 / 4.184 # FIXME: double check
    spring_k = 9.6 * 2


    # Flatten the parameter tables
    eps_flattened = list()
    sigma_flattened = list()
    nu_flattened = list()
    mu_flattened = list()
    rc_flattened = list()
    debye_rc_flattened = list()
    pair_charges = list()
    for i in range(utils.NUM_RESIDUES):
        for j in range(utils.NUM_RESIDUES):
            eps_flattened.append(eps_table[i][j])
            sigma_flattened.append(sigma_table[i][j])
            nu_flattened.append(nu_table[i][j])
            mu_flattened.append(mu_table[i][j])
            rc_flattened.append(rc_table[i][j])

            debye_rc_flattened.append(debye_rc_table[i][j])

            pair_charges.append([utils.charges[i], utils.charges[j]])

    eps_flattened = jnp.array(eps_flattened, dtype=jnp.float64)
    sigma_flattened = jnp.array(sigma_flattened, dtype=jnp.float64)
    nu_flattened = jnp.array(nu_flattened, dtype=jnp.float64)
    mu_flattened = jnp.array(mu_flattened, dtype=jnp.float64)
    rc_flattened = jnp.array(rc_flattened, dtype=jnp.float64)
    debye_rc_flattened = jnp.array(debye_rc_flattened, dtype=jnp.float64)
    pair_charges = jnp.array(pair_charges)


    def subterms_fn(
        R,
        pseq,
        unbonded_nbrs=base_unbonded_nbrs,
        debye_kappa=default_debye_kappa
    ):

        ub_i = unbonded_nbrs[:, 0]
        ub_j = unbonded_nbrs[:, 1]
        mask = jnp.array(ub_i < R.shape[0], dtype=jnp.int32)

        def pairwise_unbonded(i, j):
            ipos = R[i]
            jpos = R[j]

            r = space.distance(displacement_fn(ipos, jpos))

            iprobs = pseq[i]
            jprobs = pseq[j]
            all_probs = jnp.kron(iprobs, jprobs)

            wf_vals = mapped_wang_frenkel(
                r, rc_flattened, sigma_flattened,
                nu_flattened, mu_flattened,
                eps_flattened
            )
            wf_val = jnp.dot(all_probs, wf_vals)

            coul_vals = mapped_coul(
                r, pair_charges,
                utils.debye_relative_dielectric, debye_kappa,
                debye_rc_flattened
            )
            coul_val = jnp.dot(all_probs, coul_vals)

            val = wf_val + coul_val
            return val, (wf_val, coul_val)

        def pairwise_bonded(i, j):
            ipos = R[i]
            jpos = R[j]

            r = space.distance(displacement_fn(ipos, jpos))

            return utils._harmonic_spring(r, r0=utils.spring_r0, k=spring_k)

        bnd_i = bonded_nbrs[:, 0]
        bnd_j = bonded_nbrs[:, 1]
        total_bonded_val = jnp.sum(vmap(pairwise_bonded)(bnd_i, bnd_j))
        all_unbonded_vals, (wf_val, coul_val) = vmap(pairwise_unbonded)(ub_i, ub_j)

        total_unbonded_val = jnp.where(mask, all_unbonded_vals, 0.0).sum()
        total_wf_val = jnp.where(mask, wf_val, 0.0).sum()
        total_coul_val = jnp.where(mask, coul_val, 0.0).sum()

        total_energy = total_bonded_val + total_unbonded_val

        aux = (total_bonded_val, total_unbonded_val, total_wf_val, total_coul_val)
        return total_energy, aux

    def energy_fn(
        R,
        pseq,
        unbonded_nbrs=base_unbonded_nbrs,
        debye_kappa=default_debye_kappa
    ):
        total_energy, _ = subterms_fn(R, pseq, unbonded_nbrs, debye_kappa)
        return total_energy

    return subterms_fn, energy_fn
