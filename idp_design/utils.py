import io
import pdb
import random

import jax
import jax.numpy as jnp
import numpy as onp
import pandas as pd
from jax import tree_util, vmap
from tqdm import tqdm

kb = 0.0019872041

RES_ALPHA = "MGKTRADEYVLQWFSHNPCI"
NUM_RESIDUES = len(RES_ALPHA)
assert(NUM_RESIDUES == 20)
N20 = jnp.arange(NUM_RESIDUES)

RES_CODE = {
    "A": "Ala",
    "R": "Arg",
    "D": "Asp",
    "N": "Asn",
    "C": "Cys",
    "E": "Glu",
    "Q": "Gln",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "L": "Leu",
    "K": "Lys",
    "M": "Met",
    "F": "Phe",
    "P": "Pro",
    "S": "Ser",
    "T": "Thr",
    "W": "Trp",
    "Y": "Tyr",
    "V": "Val"
}
RES_CODE_REV = {v: k for k, v in RES_CODE.items()}

non_polar_residues = set(["G", "A", "V", "C", "P", "L", "I", "M", "W", "F"])
non_polar_mapper = jnp.array([int(res in non_polar_residues) for res in RES_ALPHA])

polar_residues = set(["S", "T", "Y", "N", "Q"])
polar_mapper = jnp.array([int(res in polar_residues) for res in RES_ALPHA])

pos_charge_residues = set(["K", "R", "H"])
pos_charge_mapper = jnp.array([int(res in pos_charge_residues) for res in RES_ALPHA])

neg_charge_residues = set(["D", "E"])
neg_charge_mapper = jnp.array([int(res in neg_charge_residues) for res in RES_ALPHA])

all_types = [("non polar", non_polar_residues), ("polar", polar_residues),
             ("+ charge", pos_charge_residues), ("- charge", neg_charge_residues)]


def nuc_to_type_distribution(nuc):
    non_polar_ratio = jnp.dot(nuc, non_polar_mapper)
    polar_ratio = jnp.dot(nuc, polar_mapper)
    pos_charge_ratio = jnp.dot(nuc, pos_charge_mapper)
    neg_charge_ratio = jnp.dot(nuc, neg_charge_mapper)
    return jnp.array([non_polar_ratio, polar_ratio, pos_charge_ratio, neg_charge_ratio])
pseq_to_type_distribution = vmap(nuc_to_type_distribution)


masses = jnp.array([
    131.199997,
    57.049999,
    128.199997,
    101.099998,
    156.199997,
    71.080002,
    115.099998,
    129.100006,
    163.199997,
    99.070000,
    113.199997,
    128.100006,
    186.199997,
    147.199997,
    87.080002,
    137.100006,
    114.099998,
    97.120003,
    103.099998,
    113.199997
])

def get_seq_mass(seq):
    return onp.array([masses[RES_ALPHA.index(res)] for res in seq])

def get_pseq_mass(pseq, res_masses=masses):
    return vmap(jnp.dot, (0, None))(pseq, res_masses)


def recenter(R, box_size):
    body_avg_pos = jnp.mean(R, axis=0)
    box_center = jnp.array([box_size / 2, box_size / 2, box_size / 2])
    disp = body_avg_pos - box_center
    center_adjusted = R - disp
    return center_adjusted

def seq_to_one_hot(seq):
    all_vecs = list()
    for res in seq:
        res_idx = RES_ALPHA.index(res)
        res_vec = onp.zeros(20)
        res_vec[res_idx] = 1.0
        all_vecs.append(res_vec)
    return onp.array(all_vecs)


def get_rand_seq(n):
    seq = ''.join([random.choice(RES_ALPHA) for _ in range(n)])
    return seq


def random_pseq(n):
    p_seq = onp.empty((n, NUM_RESIDUES), dtype=onp.float64)
    for i in range(n):
        p_seq[i] = onp.random.random_sample(NUM_RESIDUES)
        p_seq[i] /= onp.sum(p_seq[i])
    return p_seq

def get_charge_constrained_pseq(n, pos_charge_ratio, neg_charge_ratio, unconstrained_logit=10.0,
                                constrained_pos_charge_residues=pos_charge_residues,
                                constrained_neg_charge_residues=neg_charge_residues):
    assert(pos_charge_ratio > 0.0 and pos_charge_ratio <= 1.0)
    assert(neg_charge_ratio > 0.0 and neg_charge_ratio <= 1.0)
    assert(pos_charge_ratio + neg_charge_ratio < 1.0) # to avoid division by 0

    pos_charged_res_idxs = [RES_ALPHA.index(res) for res in constrained_pos_charge_residues]
    neg_charged_res_idxs = [RES_ALPHA.index(res) for res in constrained_neg_charge_residues]

    unconstrained_residues = []
    for res in RES_ALPHA:
        if (res not in constrained_pos_charge_residues) and (res not in constrained_neg_charge_residues):
            unconstrained_residues.append(res)
    unconstrainted_ratio = 1 - (pos_charge_ratio + neg_charge_ratio)
    n_neg_charge_res = len(constrained_neg_charge_residues)
    n_pos_charge_res = len(constrained_pos_charge_residues)

    neg_charged_value = neg_charge_ratio / n_neg_charge_res * unconstrained_logit \
        * len(unconstrained_residues) / unconstrainted_ratio
    pos_charged_value = pos_charge_ratio / n_pos_charge_res * n_neg_charge_res * neg_charged_value / neg_charge_ratio

    nuc = onp.zeros(len(RES_ALPHA))

    unconstrained_res_idxs = [RES_ALPHA.index(res) for res in unconstrained_residues]
    nuc[unconstrained_res_idxs] = unconstrained_logit
    nuc[pos_charged_res_idxs] = pos_charged_value
    nuc[neg_charged_res_idxs] = neg_charged_value
    nuc_normalized = nuc / nuc.sum()

    pseq = onp.vstack([nuc_normalized]*n)
    logits = onp.vstack([nuc]*n)

    return pseq, logits


# note that this is an alternative to softmaxxing
def normalize_logits(logits):
    return logits / logits.sum(axis=1)[:, jnp.newaxis]


def read_data_file(fpath):

    with open(fpath) as f:
        lines = [line for line in f.readlines() if line.strip()]

    # Read the number of atoms
    num_atoms_line = lines[1].strip()
    assert('atoms' in num_atoms_line)
    num_atoms = int(num_atoms_line.split()[0])

    # Read the number of bonds
    num_bonds_line = lines[2].strip()
    assert('bonds' in num_bonds_line)
    num_bonds = int(num_bonds_line.split()[0])

    # Read the number of atom types
    num_atom_types_line = lines[3].strip()
    assert('atom types' in num_atom_types_line)
    num_atom_types = int(num_atom_types_line.split()[0])

    # Read the number of bond types
    num_bond_types_line = lines[4].strip()
    assert('bond types' in num_bond_types_line)

    # Read the atom type masses
    assert(lines[8].strip() == "Masses")

    atom_type_masses = dict()
    all_mass_lines = lines[9:9+num_atom_types]
    for mass_line in all_mass_lines:
        mass_line = mass_line.strip()
        line_elts = mass_line.split()
        assert(len(line_elts) == 2)

        atom_type_idx = int(line_elts[0])
        atom_type_mass = float(line_elts[1])

        atom_type_masses[atom_type_idx] = atom_type_mass

    # Read the atoms
    ## Note: for now, we just read the sequence
    assert(lines[9+num_atom_types].strip() == "Atoms")
    atom_lines_start = 9+num_atom_types+1
    all_atom_lines = lines[atom_lines_start:atom_lines_start+num_atoms]

    seq = ""
    for atom_line in all_atom_lines:
        line_elts = atom_line.strip().split()
        assert(len(line_elts) == 7)
        atom_type = int(line_elts[2])
        assert(atom_type <= 20)
        seq += RES_ALPHA[atom_type-1]

    # Read the bonds
    assert(lines[atom_lines_start+num_atoms].strip() == "Bonds")
    bond_lines_start = atom_lines_start+num_atoms+1
    all_bond_lines = lines[bond_lines_start:bond_lines_start+num_bonds]

    all_bonds = list()
    for bond_line in all_bond_lines:
        line_elts = bond_line.strip().split()
        assert(len(line_elts) == 4)
        res1 = int(line_elts[2]) - 1
        res2 = int(line_elts[3]) - 1
        all_bonds.append((res1, res2))

    return all_bonds, seq, atom_type_masses, num_atoms

def read_log_file(fpath):

    with open(fpath) as f:
        lines = [line for line in f.readlines() if line.strip()]

    start_idx = -1
    end_idx = -1
    for idx, line in enumerate(lines):
        if line.strip()[:4] == "Step":
            assert(start_idx == -1)
            start_idx = idx
        elif line.strip()[:4] == "Loop":
            assert(end_idx == -1 and start_idx != -1)
            end_idx = idx

    assert(start_idx != -1 and end_idx != -1)
    df_lines = lines[start_idx:end_idx]
    log_df = pd.read_csv(io.StringIO('\n'.join(df_lines)), sep=r'\s+')

    return log_df

def read_traj_file(fpath, n_atoms):
    with open(fpath) as f:
        lines = [line for line in f.readlines() if line.strip()]

    n_lines = len(lines)
    n_lines_per_frame = 9 + n_atoms
    assert(n_lines % n_lines_per_frame == 0)
    n_frames = n_lines // n_lines_per_frame

    all_pos = list()
    all_timesteps = list()
    for i in range(n_frames):
        frame_start = n_lines_per_frame*i

        frame_timestep = int(lines[frame_start+1].strip())
        all_timesteps.append(frame_timestep)

        frame_pos_lines = lines[frame_start+9:frame_start+n_lines_per_frame]
        frame_df = pd.read_csv(
            io.StringIO('\n'.join(frame_pos_lines)), sep=r'\s+',
            header=None, names=["id", "mol", "type", "q", "xu", "yu", "zu"]
        )

        xs = frame_df.xu.to_numpy()
        ys = frame_df.yu.to_numpy()
        zs = frame_df.zu.to_numpy()
        frame_pos = onp.stack([xs, ys, zs]).T
        all_pos.append(frame_pos)

    all_pos = jnp.array(all_pos)
    return all_pos, all_timesteps


WF_GG_PATH = "params/wf_gg.txt"
WF_PATH = "params/wf.txt"
def read_wf(fpath=WF_GG_PATH):
    wf_df = pd.read_csv(
        fpath, sep=r'\s+', header=None,
        names=["res1", "res2", "eps", "sigma", "nu", "mu", "rc"]
    )

    eps_table = onp.zeros((NUM_RESIDUES, NUM_RESIDUES))
    sigma_table = onp.zeros((NUM_RESIDUES, NUM_RESIDUES))
    nu_table = onp.zeros((NUM_RESIDUES, NUM_RESIDUES))
    mu_table = onp.zeros((NUM_RESIDUES, NUM_RESIDUES))
    rc_table = onp.zeros((NUM_RESIDUES, NUM_RESIDUES))

    for i in range(NUM_RESIDUES):
        for j in range(i, NUM_RESIDUES):
            df_row = wf_df[(wf_df['res1'] == i+1) & (wf_df['res2'] == j+1)]
            assert(len(df_row) == 1)
            df_row = df_row.iloc[0]

            eps_table[i, j] = df_row.eps
            eps_table[j, i] = df_row.eps

            sigma_table[i, j] = df_row.sigma
            sigma_table[j, i] = df_row.sigma

            nu_table[i, j] = df_row.nu
            nu_table[j, i] = df_row.nu

            mu_table[i, j] = df_row.mu
            mu_table[j, i] = df_row.mu

            rc_table[i, j] = df_row.rc
            rc_table[j, i] = df_row.rc

    eps_table = jnp.array(eps_table)
    sigma_table = jnp.array(sigma_table)
    nu_table = jnp.array(nu_table)
    mu_table = jnp.array(mu_table)
    rc_table = jnp.array(rc_table)

    return eps_table, sigma_table, nu_table, mu_table, rc_table

DEBYE_GG_PATH = "params/debye_gg.txt"
DEBYE_PATH = "params/debye.txt"
def read_debye(default_cutoff=0.0, fpath=DEBYE_GG_PATH):
    debye_df = pd.read_csv(
        fpath, sep=r'\s+', header=None,
        names=["res1", "res2", "cutoff"]
    )

    cutoff_table = onp.zeros((NUM_RESIDUES, NUM_RESIDUES))

    for i in range(NUM_RESIDUES):
        for j in range(i, NUM_RESIDUES):
            df_row = debye_df[(debye_df['res1'] == i+1) & (debye_df['res2'] == j+1)]
            assert(len(df_row) <= 1)
            if len(df_row) == 1:
                cutoff = df_row.iloc[0].cutoff
            else:
                cutoff = default_cutoff

            cutoff_table[i, j] = cutoff
            cutoff_table[j, i] = cutoff

    return jnp.array(cutoff_table)

charges = onp.zeros(NUM_RESIDUES)
charges[onp.array([2, 4])] = 0.75
charges[15] = 0.375
charges[onp.array([6, 7])] = -0.75
charges = jnp.array(charges)

DEBYE_KAPPA_GG = 0.126
DEBYE_KAPPA = 0.131

debye_kappa = 0.126 # Mpipi-GG
# debye_kappa = 0.131 # Mpipi
debye_relative_dielectric = 80.0

spring_r0 = 3.81

def wang_frenkel(r, r_c, sigma, nu, mu, eps):
    # r_min = r_c*((1+2*nu) / (1 + 2*nu*(r_c/sigma)**(2*nu)))**(1/(2*nu))

    alpha = 2*nu * (r_c/sigma)**(2*mu)
    alpha *= ((1+2*nu) / (2*nu * ((r_c/sigma)**(2*mu)-1)))**(2*nu+1)

    val = eps*alpha * ((sigma/r)**(2*mu)-1) * ((r_c/r)**(2*mu)-1)**(2*nu)

    return jnp.where(r < r_c, val, 0.0)


def coul(r, qij, eps, k, r_c):
    qi = qij[0]
    qj = qij[1]

    C = 5.513725184e-22 # (1/(4*pi*eps0)  in units of kcal* (Armstrong)/(electron^^2)

    val = (C*qi*qj) / (eps*r) * jnp.exp(-k*r)
    val *= 6.022e23
    return jnp.where(r < r_c, val, 0.0)

def harmonic_spring(r, r0, k):
    return 1/2 * k * (r-r0)**2


def tree_stack(trees):
    return tree_util.tree_map(lambda *v: jnp.stack(v), *trees)



default_color = "c0c0c0"
proline_color = "5b5b5b"
neg_charge_color = "b21b0c"

# positively charged
histidine_color = "8cbed6"
lysine_color = "bf00ff"
arginine_color = "444eff"

# aromatic
tryptophan_color = "52c71f"
tyrosine_color = "95ce7c"
phenylalanine_color = "cde1c5"

# define the color mapper
COLOR_MAPPER = dict()
for res in RES_ALPHA:
    if res in neg_charge_residues:
        COLOR_MAPPER[res] = neg_charge_color
    elif res == "P":
        COLOR_MAPPER[res] = proline_color
    elif res == "H":
        COLOR_MAPPER[res] = histidine_color
    elif res == "K":
        COLOR_MAPPER[res] = lysine_color
    elif res == "R":
        COLOR_MAPPER[res] = arginine_color
    elif res == "W":
        COLOR_MAPPER[res] = tryptophan_color
    elif res == "Y":
        COLOR_MAPPER[res] = tyrosine_color
    elif res == "F":
        COLOR_MAPPER[res] = phenylalanine_color
    else:
        COLOR_MAPPER[res] = default_color


def dump_pos(traj, filename, box_size, seq=None):
    n_states = len(traj)
    if seq is None:
        particle_type_str = 'def R "sphere 3.81 75a2f7" \n'
    else:
        particle_type_str = ''
        for res in RES_ALPHA:
            particle_type_str += f'def {res} "sphere 3.81 {COLOR_MAPPER[res]}"'
            particle_type_str += " \n"

    for pos_idx in tqdm(range(n_states)):
        pos = traj[pos_idx]
        pos -= jnp.array([box_size/2, box_size/2, box_size/2])
        with open(filename, 'a') as outfile:
            outfile.write('boxMatrix '+ str(box_size) + ' 0 0 0 ' + str(box_size) + ' 0 0 0 ' + str(box_size) + ' \n')
            # outfile.write(f'def R "sphere 3.81 {color}" \n')
            outfile.write(particle_type_str)
            for p_idx, position in enumerate(pos):
                if seq is None:
                    particle_type = "R"
                else:
                    particle_type = seq[p_idx]
                    assert(particle_type in RES_ALPHA)
                entry = f"{particle_type} {position[0]} {position[1]} {position[2]}\n"
                outfile.write(entry)

            outfile.write('eof \n')


def compute_weights(ref_energies, new_energies, beta):
    diffs = new_energies - ref_energies # element-wise subtraction
    boltzs = jnp.exp(-beta * diffs)
    denom = jnp.sum(boltzs)
    weights = boltzs / denom

    n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))

    return weights, n_eff


def get_argmax_seq(pseq, scale=False):
    max_residues = jnp.argmax(pseq, axis=1)
    argmax_seq = ''.join([RES_ALPHA[res_idx] for res_idx in max_residues])

    if scale:
        argmax_seq = []
        for r_idx in range(len(argmax_seq)):
            if pseq[r_idx, max_residues[r_idx]] < 0.5:
                argmax_seq.append(argmax_seq[r_idx].lower())
            else:
                argmax_seq.append(argmax_seq[r_idx])
        argmax_seq = ''.join(argmax_seq)

    return argmax_seq

def aa_seq_to_type_seq(aa_seq):
    type_seq = ""
    for res in aa_seq:
        if res in non_polar_residues:
            type_seq += "N"
        elif res in polar_residues:
            type_seq += "P"
        elif res in pos_charge_residues:
            type_seq += "+"
        elif res in neg_charge_residues:
            type_seq += "-"
        else:
            raise RuntimeError(f"Invalid residuee: {res}")
    return type_seq



def sample_discrete_seqs(pseq, nsamples, key):
    n = pseq.shape[0]

    def sample_indices(sample_key):
        # Generate a random number for each row
        uniform_samples = jax.random.uniform(
            sample_key, shape=(pseq.shape[0],),
            minval=0, maxval=1)

        # Compute the cumulative sum of probabilities for each row
        cumulative_probabilities = jnp.cumsum(pseq, axis=1)

        # Determine index where the cumulative probability first exceeds the random sample
        sampled_indices = jnp.sum(cumulative_probabilities < uniform_samples[:, None], axis=1)

        return sampled_indices

    sample_keys = jax.random.split(key, nsamples)
    seqs = list()
    freqs = onp.zeros((n, len(RES_ALPHA)))
    for i in range(nsamples):
        sample_key = sample_keys[i]
        sample = sample_indices(sample_key)
        freqs[onp.arange(n), sample] += 1
        seq = ''.join([RES_ALPHA[idx] for idx in sample])
        seqs.append(seq)

    # Can be used for testing -- this should approximate pseq
    freqs_norm = freqs / freqs.sum(axis=1, keepdims=True)

    return seqs, freqs_norm



def get_kappa(salt_conc, use_gg=True):
    """Computes the inverse Debye screening length from a salt concentration in mM."""
    if use_gg:
        default_kappa = DEBYE_KAPPA_GG
    else:
        default_kappa = DEBYE_KAPPA

    return default_kappa * (onp.sqrt(salt_conc / 1000) / onp.sqrt(150 / 1000))


if __name__ == "__main__":

    pseq = get_charge_constrained_pseq(10, 0.45, 0.35)

    pdb.set_trace()

    data_fpath = "refdata/LAMMPS/20231101_single_chain/ACTR/ACTR.dat"
    all_bonds, seq, atom_type_masses, num_atoms = read_data_file(data_fpath)

    log_fpath = "refdata/LAMMPS/20231101_single_chain/ACTR/log.lammps"
    log_df = read_log_file(log_fpath)

    traj_fpath = "refdata/LAMMPS/20231101_single_chain/ACTR/result.lammpstrj"
    traj_positions, traj_timesteps = read_traj_file(traj_fpath, num_atoms)

    pdb.set_trace()
