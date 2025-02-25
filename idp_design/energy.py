import itertools
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
from jax import vmap
from jax_md import space

import idp_design.utils as utils
from idp_design.utils import (
    DEBYE_KAPPA,
    DEBYE_KAPPA_GG,
    RES_ALPHA,
    charges,
    coul,
    debye_relative_dielectric,
    harmonic_spring,
    read_debye,
    read_wf,
    spring_r0,
    wang_frenkel,
)

jax.config.update("jax_enable_x64", True)


def get_energy_fn(bonded_nbrs, base_unbonded_nbrs, displacement_fn, use_gg=True):

    if use_gg:
        debye_kappa = DEBYE_KAPPA_GG
        debye_path = utils.DEBYE_GG_PATH
        wf_path = utils.WF_GG_PATH
    else:
        debye_kappa = DEBYE_KAPPA
        debye_path = utils.DEBYE_PATH
        wf_path = utils.WF_PATH

    eps_table, sigma_table, nu_table, mu_table, rc_table = read_wf(fpath=wf_path)
    cutoff_table = read_debye(fpath=debye_path)

    # spring_k = 8.03 * 10 / 4.184 # FIXME: double check
    spring_k = 9.6 * 2

    def subterms_fn(R, seq, unbonded_nbrs=base_unbonded_nbrs):

        def pairwise_unbonded(i, j):
            ipos = R[i]
            jpos = R[j]

            r = space.distance(displacement_fn(ipos, jpos))

            ires = seq[i]
            jres = seq[j]

            wf_val = wang_frenkel(r, r_c=rc_table[ires, jres], sigma=sigma_table[ires, jres],
                                  nu=nu_table[ires, jres], mu=mu_table[ires, jres],
                                  eps=eps_table[ires, jres])
            coul_val = coul(r, qij=jnp.array([charges[ires], charges[jres]]),
                            eps=debye_relative_dielectric, k=debye_kappa,
                            r_c=cutoff_table[ires, jres])

            val = wf_val + coul_val
            # val = jnp.where(r > 35, 0.0, val)
            return val, (wf_val, coul_val)

        def pairwise_bonded(i, j):
            ipos = R[i]
            jpos = R[j]

            r = space.distance(displacement_fn(ipos, jpos))

            return harmonic_spring(r, r0=spring_r0, k=spring_k)

        total_bonded_val = jnp.sum(vmap(pairwise_bonded)(bonded_nbrs[:, 0], bonded_nbrs[:, 1]))
        all_unbonded_vals, (wf_val, coul_val) = vmap(pairwise_unbonded)(unbonded_nbrs[:, 0], unbonded_nbrs[:, 1])
        total_unbonded_val = jnp.sum(all_unbonded_vals)
        total_wf_val = jnp.sum(wf_val)
        total_coul_val = jnp.sum(coul_val)

        total_energy = total_bonded_val + total_unbonded_val
        return total_energy, (total_bonded_val, total_unbonded_val, total_wf_val, total_coul_val)
        # return total_bonded_val, total_unbonded_val, total_wf_val, total_coul_val

    def energy_fn(R, seq, unbonded_nbrs=base_unbonded_nbrs):
        total_energy, (total_bonded_val, total_unbonded_val, total_wf_val, total_coul_val) = subterms_fn(R, seq, unbonded_nbrs)
        return total_energy

    return subterms_fn, energy_fn
