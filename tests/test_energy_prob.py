import pytest
import jax.numpy as jnp
import itertools
from pathlib import Path
import numpy as onp
from tqdm import tqdm

from jax import jit
from jax_md import space

from idp_design import energy_prob, utils


def random_pseq(n):
    p_seq = onp.empty((n, utils.NUM_RESIDUES), dtype=onp.float64)
    for i in range(n):
        p_seq[i] = onp.random.random_sample(utils.NUM_RESIDUES)
        p_seq[i] /= onp.sum(p_seq[i])
    return p_seq


def brute_force(pseq, R, bonded_nbrs, unbonded_nbrs, displacement_fn):
    n = pseq.shape[0]

    subterms_fn, energy_fn = energy_prob.get_energy_fn(
        bonded_nbrs, unbonded_nbrs, displacement_fn
    )
    subterms_fn = jit(subterms_fn)

    residue_combinations = list(
        itertools.product(*[range(utils.NUM_RESIDUES) for i in range(n)])
    )
    n_res_combinations = len(residue_combinations)

    brute_sm = 0.0
    for n_rc in tqdm(range(n_res_combinations), desc="Brute force"):
        res_combo = residue_combinations[n_rc]
        seq = ""
        pr_seq = 1.0
        for i in range(n):
            residue = utils.RES_ALPHA[res_combo[i]]
            seq += residue

            pr_seq *= pseq[i, res_combo[i]]

        pseq_oh = jnp.array(utils.seq_to_one_hot(seq))
        val, _ = subterms_fn(R, pseq_oh)
        brute_sm += val*pr_seq

    calc_energy, aux = subterms_fn(R, jnp.array(pseq))
    (total_bonded_val, total_unbonded_val, wf_val, coul_val) = aux

    return brute_sm, calc_energy


@pytest.mark.parametrize(
    ("data_fpath", "log_fpath", "traj_fpath", "tol_places", "use_gg"),
    [
        (
            "refdata/LAMMPS/20231102_sc_high_precision/K25/K25.dat",
            "refdata/LAMMPS/20231102_sc_high_precision/K25/log.lammps",
            "refdata/LAMMPS/20231102_sc_high_precision/K25/result.lammpstrj",
            3,
            False
        ),
        (
            "refdata/MPIPI-GG/P53/P53.dat",
            "refdata/MPIPI-GG/P53/log.lammps",
            "refdata/MPIPI-GG/P53/result.lammpstrj",
            3,
            True
        ),
    ]
)
def test_lammps(data_fpath, log_fpath, traj_fpath, tol_places, use_gg):
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
    pseq = jnp.array(utils.seq_to_one_hot(seq))

    subterms_fn, energy_fn = energy_prob.get_energy_fn(bonded_nbrs, unbonded_nbrs, displacement_fn, use_gg)
    subterms_fn = jit(subterms_fn)

    num_frames = len(traj_timesteps)
    all_diffs = list()

    for i in range(num_frames):
        if i == 0:
            continue

        R = traj_positions[i]
        timestep = traj_timesteps[i]

        calc_energy, (total_bonded_val, total_unbonded_val, wf_val, coul_val) = subterms_fn(R, pseq)
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



def test_fuzzy_seq():
    n = 4
    R = list()
    for i in range(n):
        R.append([0.0, 0.0, utils.spring_r0*i])
    R = jnp.array(R)
    pseq = random_pseq(n)
    displacement_fn, _ = space.free()

    bonded_nbrs = jnp.array([(i, i+1) for i in range(n-1)])
    unbonded_nbrs = list()
    for pair in itertools.combinations(jnp.arange(n), 2):
        unbonded_nbrs.append(pair)
    unbonded_nbrs = jnp.array(unbonded_nbrs)

    unbonded_nbrs_set = set([tuple(pr) for pr in onp.array(unbonded_nbrs)])
    bonded_nbrs_set = set([tuple(pr) for pr in onp.array(bonded_nbrs)])
    unbonded_nbrs = jnp.array(list(unbonded_nbrs_set - bonded_nbrs_set))

    brute_val, calc_val = brute_force(
        pseq, R, bonded_nbrs, unbonded_nbrs, displacement_fn
    )

    print(f"Calc: {calc_val}")
    print(f"Brute Force: {brute_val}")

    onp.testing.assert_allclose(brute_val, calc_val)
