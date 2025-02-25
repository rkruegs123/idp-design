import pytest
import jax.numpy as jnp
import itertools
from pathlib import Path
import numpy as onp
import random

from jax_md import space

from idp_design import energy, utils


def get_rand_seq(n):
    seq = ''.join([random.choice(utils.RES_ALPHA) for _ in range(n)])
    return seq

def test_no_warnings():
    displacement_fn, shift_fn = space.free()
    n = 5
    R = jnp.array([[0.0, 0.0, utils.spring_r0 * i] for i in range(n)])

    seq = get_rand_seq(n)
    seq_idx = jnp.array([utils.RES_ALPHA.index(res) for res in seq])

    bonded_nbrs = jnp.array([(i, i+1) for i in range(n-1)])
    unbonded_nbrs = jnp.array(list(itertools.combinations(jnp.arange(n), 2)))

    subterms_fn, energy_fn = energy.get_energy_fn(bonded_nbrs, unbonded_nbrs, displacement_fn)
    energy_val = energy_fn(R, seq_idx)


@pytest.mark.parametrize(
    ("data_fpath", "log_fpath", "traj_fpath", "tol_places", "use_gg"),
    [
        (
            "refdata/LAMMPS/20231101_single_chain/K25/K25.dat",
            "refdata/LAMMPS/20231101_single_chain/K25/log.lammps",
            "refdata/LAMMPS/20231101_single_chain/K25/result.lammpstrj",
            1,
            False
        ),
        (
            "refdata/LAMMPS/20231102_sc_high_precision/K25/K25.dat",
            "refdata/LAMMPS/20231102_sc_high_precision/K25/log.lammps",
            "refdata/LAMMPS/20231102_sc_high_precision/K25/result.lammpstrj",
            3,
            False
        ),
        (
            "refdata/LAMMPS/20231101_single_chain/ACTR/ACTR.dat",
            "refdata/LAMMPS/20231101_single_chain/ACTR/log.lammps",
            "refdata/LAMMPS/20231101_single_chain/ACTR/result.lammpstrj",
            1,
            False
        ),
    ]
)
def test_lammps(data_fpath, log_fpath, traj_fpath, tol_places, use_gg):
    displacement_fn, shift_fn = space.free()


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
    seq_idx = jnp.array([utils.RES_ALPHA.index(res) for res in seq])

    num_frames = len(traj_timesteps)
    all_diffs = list()
    subterms_fn, energy_fn = energy.get_energy_fn(
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

        onp.testing.assert_allclose(calc_energy, ref_energy, atol=10**(-tol_places))
