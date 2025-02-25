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

class TestEnergyCalculator(unittest.TestCase):
    def test_no_warnings(self):
        displacement_fn, shift_fn = space.free()

        # Randomly initialize a position and test that there are no errors

        ## String of beads of length n
        n = 5
        R = list()
        for i in range(n):
            R.append([0.0, 0.0, spring_r0*i])
        R = jnp.array(R)

        seq = utils.get_rand_seq(n)
        seq_idx = jnp.array([RES_ALPHA.index(res) for res in seq])

        bonded_nbrs = jnp.array([(i, i+1) for i in range(n-1)])

        unbonded_nbrs = list()
        for pair in itertools.combinations(jnp.arange(n), 2):
            unbonded_nbrs.append(pair)
        unbonded_nbrs = jnp.array(unbonded_nbrs)

        subterms_fn, energy_fn = get_energy_fn(bonded_nbrs, unbonded_nbrs, displacement_fn)

        # energy_val, (total_bonded_val, total_unbonded, wf_val, coul_val) = energy_fn(R, seq_idx)
        energy_val = energy_fn(R, seq_idx)


    def lammps_test(self, data_fpath, log_fpath, traj_fpath, tol_places, use_gg=False, show_plot=False):
        displacement_fn, shift_fn = space.free()

        all_bonds, seq, atom_type_masses, num_atoms = utils.read_data_file(data_fpath)
        log_df = utils.read_log_file(log_fpath)
        traj_positions, traj_timesteps = utils.read_traj_file(traj_fpath, num_atoms)


        bonded_nbrs = jnp.array(all_bonds)
        unbonded_nbrs = list()
        for pair in itertools.combinations(jnp.arange(num_atoms), 2):
            unbonded_nbrs.append(pair)
        unbonded_nbrs = jnp.array(unbonded_nbrs)

        unbonded_nbrs_set = set([tuple(pr) for pr in onp.array(unbonded_nbrs)])
        bonded_nbrs_set = set([tuple(pr) for pr in onp.array(bonded_nbrs)])
        unbonded_nbrs = jnp.array(list(unbonded_nbrs_set - bonded_nbrs_set))
        seq_idx = jnp.array([RES_ALPHA.index(res) for res in seq])

        num_frames = len(traj_timesteps)
        all_diffs = list()
        subterms_fn, energy_fn = get_energy_fn(
            bonded_nbrs, unbonded_nbrs, displacement_fn, use_gg=use_gg
        )
        for i in range(num_frames):
            if i == 0:
                continue

            R = traj_positions[i]
            timestep = traj_timesteps[i]

            calc_energy, (total_bonded_val, total_unbonded_val, wf_val, coul_val) = subterms_fn(R, seq_idx)
            print(f"\nCalc. Energy: {calc_energy}")
            print(f"\t- Bonded: {total_bonded_val}")
            print(f"\t- Unbonded: {total_unbonded_val}")
            print(f"\t\t- WF Val: {wf_val}")
            print(f"\t\t- Coulomb Val: {coul_val}")

            timestep_df = log_df[log_df.Step == timestep]
            assert(len(timestep_df) == 1)
            timestep_row = timestep_df.iloc[0]
            ref_energy = timestep_row.PotEng
            print(f"Reference Energy: {ref_energy}")

            if "E_bond" in timestep_row:
                ref_bonded_energy = timestep_row.E_bond
                print(f"\t- Bonded: {ref_bonded_energy}")
            if "E_coul" in timestep_row and "E_vdwl" in timestep_row:
                ref_coul_energy = timestep_row.E_coul
                ref_vdwl_energy = timestep_row.E_vdwl
                ref_unbonded_energy = ref_coul_energy + ref_vdwl_energy

                print(f"\t- Unbonded: {ref_unbonded_energy}")
                print(f"\t\t- WF Val: {ref_vdwl_energy}")
                print(f"\t\t- Coulomb Val: {ref_coul_energy}")


            diff = calc_energy - ref_energy
            print(f"Diff: {diff}")
            if "E_bond" in timestep_row:
                bonded_diff = total_bonded_val - ref_bonded_energy
                print(f"\t- Bonded: {bonded_diff}")
            if "E_coul" in timestep_row and "E_vdwl" in timestep_row:
                unbonded_diff = total_unbonded_val - ref_unbonded_energy
                print(f"\t- Unbonded: {unbonded_diff}")

                wf_diff = wf_val - ref_vdwl_energy
                coul_diff = coul_val - ref_coul_energy
                print(f"\t\t- WF Val: {wf_diff}")
                print(f"\t\t- Coulomb Val: {coul_diff}")


            all_diffs.append(diff)
            self.assertAlmostEqual(calc_energy, ref_energy, places=tol_places)

        if show_plot:
            plt.hist(all_diffs)
            plt.show()

    def test_lammps_ref(self):

        lammps_tests = list()

        # Lower precision
        basedir = Path("refdata/LAMMPS/20231101_single_chain/K25")
        data_fpath = basedir / "K25.dat"
        log_fpath = basedir / "log.lammps"
        traj_fpath = basedir / "result.lammpstrj"
        lammps_tests.append((data_fpath, log_fpath, traj_fpath, 1, False))

        basedir = Path("refdata/LAMMPS/20231101_single_chain/ACTR/")
        data_fpath = basedir / "ACTR.dat"
        log_fpath = basedir / "log.lammps"
        traj_fpath = basedir / "result.lammpstrj"
        lammps_tests.append((data_fpath, log_fpath, traj_fpath, 1, False))

        # Higher precision
        basedir = Path("refdata/LAMMPS/20231102_sc_high_precision/K25")
        data_fpath = basedir / "K25.dat"
        log_fpath = basedir / "log.lammps"
        traj_fpath = basedir / "result.lammpstrj"
        lammps_tests.append((data_fpath, log_fpath, traj_fpath, 3, False))


        for data_fpath, log_fpath, traj_fpath, tol_places, use_gg in lammps_tests:
            self.lammps_test(data_fpath, log_fpath, traj_fpath, tol_places, use_gg=use_gg)


if __name__ == "__main__":
    unittest.main()
