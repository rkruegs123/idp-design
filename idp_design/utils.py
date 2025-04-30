import io
import random
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as onp
import pandas as pd
import pkg_resources
from jax import tree_util, vmap
from tqdm import tqdm

PARAMS_BASEDIR = Path(pkg_resources.resource_filename("idp_design", "params"))

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
    """Computes the distribution of residue types in a probabilistic sequence.

    Given a probability distribution over amino acids (`nuc`), this function
    calculates the expected proportion of residues belonging to four
    physicochemical categories:
    - **Non-polar**: {G, A, V, C, P, L, I, M, W, F}
    - **Polar**: {S, T, Y, N, Q}
    - **Positively charged**: {K, R, H}
    - **Negatively charged**: {D, E}

    The function computes these proportions using dot products between the
    probability distribution `nuc` and precomputed binary mappers for each
    residue type.

    Args:
        nuc (jnp.ndarray): A `(20,)` JAX array representing a probability
            distribution over amino acids, following the `RES_ALPHA` ordering.

    Returns:
        jnp.ndarray: A `(4,)` JAX array containing the expected fraction of
        residues in each category: `[non-polar, polar, positively charged,
        negatively charged]`.

    Example:
        >>> nuc = jnp.array([0.05] * 20)  # Uniform distribution over all amino acids
        >>> nuc_to_type_distribution(nuc)
        Array([0.50, 0.25, 0.15, 0.10], dtype=float32)  # Example proportions
    """
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


def get_pseq_mass(pseq, res_masses=masses):
    """Computes the expected molecular mass of a probabilistic sequence.

    Given a probability matrix `pseq` of shape `(n, 20)`, where each row represents
    a probability distribution over the 20 amino acids (ordered by `RES_ALPHA`),
    this function calculates the expected residue mass at each position by
    computing the dot product between `pseq` and `res_masses`.

    Args:
        pseq (jnp.ndarray): A `(n, 20)` JAX array where each row represents a
            probability distribution over amino acids.
        res_masses (jnp.ndarray, optional): A `(20,)` JAX array containing the
            molecular masses of amino acids in the `RES_ALPHA` order. Defaults
            to `masses`.

    Returns:
        jnp.ndarray: A `(n,)` JAX array where each element is the expected molecular
        mass at the corresponding position in `pseq`.

    Example:
        >>> pseq = jnp.array([
        ...     [0.5, 0.5] + [0.0] * 18,  # 50% 'M' (131.2) and 50% 'G' (57.0)
        ...     [0.0, 1.0] + [0.0] * 18   # 100% 'G' (57.0)
        ... ])
        >>> get_pseq_mass(pseq)
        Array([94.125,  57.05], dtype=float32)
    """
    return vmap(jnp.dot, (0, None))(pseq, res_masses)


def recenter(R, box_size):
    """Recenters a set of 3D positions within a cubic simulation box.

    This function shifts the input coordinates `R` such that their average position
    aligns with the center of a cubic box of given `box_size`. The displacement is
    computed using the unweighted mean of all positions in `R`.

    Args:
        R (jnp.ndarray): A `(n, 3)` JAX array representing `n` positions in 3D space.
        box_size (float): The size of the cubic box along each axis.

    Returns:
        jnp.ndarray: A `(n, 3)` JAX array of recentered positions, where the
        unweighted average position is aligned with the box center.

    Example:
        >>> R = jnp.array([
        ...     [1.0, 2.0, 3.0],
        ...     [4.0, 5.0, 6.0]
        ... ])
        >>> recenter(R, box_size=10.0)
        Array([[-3.5, -3.5, -3.5],
               [6.5, 6.5, 6.5]], dtype=float32)
    """
    body_avg_pos = jnp.mean(R, axis=0)
    box_center = jnp.array([box_size / 2, box_size / 2, box_size / 2])
    disp = body_avg_pos - box_center
    center_adjusted = R - disp
    return center_adjusted


def seq_to_one_hot(seq):
    """Converts a discrete amino acid sequence into a one-hot encoded representation.

    Each amino acid in the input sequence is mapped to a one-hot vector of length 20,
    corresponding to its index in `RES_ALPHA`.

    The output is an `(n, 20)` NumPy array where `n` is the sequence length, and each
    row represents a one-hot encoded probability distribution.

    Args:
        seq (list of str): A list of amino acid characters representing the sequence.

    Returns:
        numpy.ndarray: A `(n, 20)` array where each row is one-hot encoded
        of the corresponding amino acid in `seq`, following the `RES_ALPHA` ordering.

    Example:
        >>> seq = ["M", "G", "K"]
        >>> seq_to_one_hot(seq)
        array([[1., 0., 0., ..., 0.],  # 'M' -> one-hot
               [0., 1., 0., ..., 0.],  # 'G' -> one-hot
               [0., 0., 1., ..., 0.]]) # 'K' -> one-hot
    """
    all_vecs = list()
    for res in seq:
        res_idx = RES_ALPHA.index(res)
        res_vec = onp.zeros(20)
        res_vec[res_idx] = 1.0
        all_vecs.append(res_vec)
    return onp.array(all_vecs)




def read_data_file(fpath):
    """Reads metadata from LAMMPS data file."""
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
    """Reads metadata from LAMMPS log file."""
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
    """Reads metadata from LAMMPS trajectory file."""
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


WF_GG_PATH = PARAMS_BASEDIR / "wf_gg.txt"
WF_PATH = PARAMS_BASEDIR / "wf.txt"
def _read_wf(fpath=WF_GG_PATH):
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

DEBYE_GG_PATH = PARAMS_BASEDIR / "debye_gg.txt"
DEBYE_PATH = PARAMS_BASEDIR / "debye.txt"
def _read_debye(default_cutoff=0.0, fpath=DEBYE_GG_PATH):
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

def _wang_frenkel(r, r_c, sigma, nu, mu, eps):
    # r_min = r_c*((1+2*nu) / (1 + 2*nu*(r_c/sigma)**(2*nu)))**(1/(2*nu))

    alpha = 2*nu * (r_c/sigma)**(2*mu)
    alpha *= ((1+2*nu) / (2*nu * ((r_c/sigma)**(2*mu)-1)))**(2*nu+1)

    val = eps*alpha * ((sigma/r)**(2*mu)-1) * ((r_c/r)**(2*mu)-1)**(2*nu)

    return jnp.where(r < r_c, val, 0.0)


def _coul(r, qij, eps, k, r_c):
    qi = qij[0]
    qj = qij[1]

    C = 5.513725184e-22 # (1/(4*pi*eps0)  in units of kcal* (Armstrong)/(electron^^2)

    val = (C*qi*qj) / (eps*r) * jnp.exp(-k*r)
    val *= 6.022e23
    return jnp.where(r < r_c, val, 0.0)

def _harmonic_spring(r, r0, k):
    return 1/2 * k * (r-r0)**2


def tree_stack(trees):
    """Stacks multiple PyTree structures element-wise along a new axis.

    This function takes a list of PyTree structures (e.g., nested dictionaries,
    lists, or tuples of JAX arrays) and stacks corresponding elements across
    all trees using `jnp.stack()`. The result maintains the same PyTree structure,
    but each leaf node is now a stacked JAX array.

    Args:
        trees (list of PyTree): A list of PyTree structures, where each leaf
            node contains a JAX array of the same shape.

    Returns:
        PyTree: A PyTree structure with the same shape as the input, where each
        leaf node is a JAX array stacked along a new axis.

    Example:
        >>> from jax import tree_util
        >>> import jax.numpy as jnp
        >>> tree1 = {"a": jnp.array([1, 2]), "b": jnp.array([3, 4])}
        >>> tree2 = {"a": jnp.array([5, 6]), "b": jnp.array([7, 8])}
        >>> tree_stack([tree1, tree2])
        {'a': Array([[1, 2],
                     [5, 6]], dtype=int32),
         'b': Array([[3, 4],
                     [7, 8]], dtype=int32)}
    """
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
    """Exports a trajectory to a `.pos` file for visualization in INJAVIS.

    This function writes a trajectory to a `.pos` file, a format compatible
    with the INJAVIS visualization software. The trajectory consists of
    `n_states` frames, each defining particle positions inside a cubic box.
    Optionally, amino acid sequence information (`seq`) can be included,
    assigning distinct colors to different residue types.

    The generated `.pos` file can be viewed using:
        java -Xmx4096m -jar injavis.jar <filename>

    Args:
        traj (list of jnp.ndarray): A list of `(n, 3)` JAX arrays, where each
            array represents a frame of `n` particle positions in 3D space.
        filename (str): The output filename for the `.pos` file.
        box_size (float): The size of the cubic simulation box.
        seq (str, optional): A sequence of amino acids (matching `RES_ALPHA`),
            used to assign particle types for color mapping. If `None`, a
            default representation is used.

    Returns:
        None: The function writes the trajectory data to `filename`.

    Example:
        >>> traj = [jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])]
        >>> dump_pos(traj, "output.pos", box_size=10.0, seq="MG")
    """
    n_states = len(traj)
    if seq is None:
        particle_type_str = 'def R "sphere 3.81 75a2f7" \n'
    else:
        particle_type_str = ''
        for res in RES_ALPHA:
            particle_type_str += f'def {res} "sphere 3.81 {COLOR_MAPPER[res]}"'
            particle_type_str += " \n"

    box_def = f"boxMatrix {box_size} 0 0 0 {box_size} 0 0 0 {box_size}\n"
    for pos_idx in tqdm(range(n_states)):
        pos = traj[pos_idx]
        pos -= jnp.array([box_size/2, box_size/2, box_size/2])
        with open(filename, 'a') as outfile:
            outfile.write(box_def)
            outfile.write(particle_type_str)
            for p_idx, position in enumerate(pos):
                if seq is None:
                    particle_type = "R"
                else:
                    particle_type = seq[p_idx]
                    assert particle_type in RES_ALPHA
                entry = f"{particle_type} {position[0]} {position[1]} {position[2]}\n"
                outfile.write(entry)
            outfile.write('eof \n')


def compute_weights(ref_energies, new_energies, beta):
    """Compute DiffTRE weights given calculated and reference energies."""
    diffs = new_energies - ref_energies # element-wise subtraction
    boltzs = jnp.exp(-beta * diffs)
    denom = jnp.sum(boltzs)
    weights = boltzs / denom

    n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))

    return weights, n_eff


def get_argmax_seq(pseq, scale=False):
    """Converts a probabilistic sequence into a deterministic sequence using argmax.

    Given a probability matrix `pseq` of shape `(n, 20)`, where each row represents
    a probability distribution over the 20 amino acids (ordered by `RES_ALPHA`),
    this function selects the most probable amino acid at each position using `argmax`.

    If `scale=True`, residues with probabilities below 0.5 are converted to lowercase
    to indicate uncertainty in the selection.

    Args:
        pseq (jnp.ndarray): A `(n, 20)` JAX array where each row represents a
            probability distribution over amino acids.
        scale (bool, optional): If `True`, amino acids with probabilities below 0.5
            are returned in lowercase. Defaults to `False`.

    Returns:
        str: A deterministic amino acid sequence derived from `pseq`, with lowercase
        letters marking uncertain assignments if `scale=True`.

    Example:
        >>> pseq = jnp.array([
        ...     [0.9, 0.1] + [0.0] * 18,  # 'M' (high confidence)
        ...     [0.4, 0.6] + [0.0] * 18   # 'G' (low confidence, if scaled)
        ... ])
        >>> get_argmax_seq(pseq)
        'MG'
    """
    max_residues = jnp.argmax(pseq, axis=1)
    argmax_seq = ''.join([RES_ALPHA[res_idx] for res_idx in max_residues])

    if scale:
        argmax_seq = []
        for r_idx in range(len(max_residues)):
            if pseq[r_idx, max_residues[r_idx]] < 0.5:
                argmax_seq.append(RES_ALPHA[max_residues[r_idx]].lower())
            else:
                argmax_seq.append(RES_ALPHA[max_residues[r_idx]])
        argmax_seq = ''.join(argmax_seq)

    return argmax_seq



def sample_discrete_seqs(pseq, nsamples, key):
    """Samples discrete sequences from a probabilistic sequence representation.

    Given a probability matrix `pseq` of shape `(n, 20)`, where each row represents
    a probability distribution over the 20 amino acids (ordered by `RES_ALPHA`),
    this function generates `nsamples` discrete sequences. Sampling is performed
    using categorical distributions derived from `pseq`.

    Args:
        pseq (jnp.ndarray): A `(n, 20)` JAX array where each row represents a
            probability distribution over amino acids.
        nsamples (int): The number of discrete sequences to sample.
        key (jax.random.PRNGKey): A JAX random key for reproducibility.

    Returns:
        tuple:
            - list of str: A list of `nsamples` sampled amino acid sequences.
            - numpy.ndarray: A `(n, 20)` matrix of normalized empirical amino
              acid frequencies from the sampled sequences.

    Example:
        >>> pseq = jnp.array([
        ...     [0.7, 0.2, 0.1] + [0.0] * 17,  # Mostly 'M'
        ...     [0.1, 0.1, 0.8] + [0.0] * 17   # Mostly 'K'
        ... ])
        >>> key = jax.random.PRNGKey(42) # exact behavior depends on JAX version
        >>> seqs, freqs = sample_discrete_seqs(pseq, nsamples=1000, key=key)
        >>> seqs[:3]  # Example sampled sequences
        ['MK', 'MK', 'MM']
        >>> freqs  # Should approximate pseq
        array([[0.7, 0.2, 0.1, ..., 0.0],
               [0.1, 0.1, 0.8, ..., 0.0]])
    """
    n = pseq.shape[0]

    def sample_indices(sample_key):
        # Generate a random number for each row
        uniform_samples = jax.random.uniform(
            sample_key, shape=(pseq.shape[0],), minval=0, maxval=1
        )

        # Compute the cumulative sum of probabilities for each row
        cumulative_probabilities = jnp.cumsum(pseq, axis=1)

        # Determine index where the cumulative probability first exceeds the sample
        sampled_indices = jnp.sum(
            cumulative_probabilities < uniform_samples[:, None], axis=1
        )

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


def get_rand_seq(n: int) -> str:
    """Generates a random amino acid sequence of length `n`.

    The sequence is generated by randomly selecting amino acids from `RES_ALPHA`,
    which defines the valid amino acid alphabet.

    :param n: The length of the random sequence.
    :type n: int

    :returns: A randomly generated amino acid sequence of length `n`.
    :rtype: str

    """
    seq = ''.join([random.choice(RES_ALPHA) for _ in range(n)])
    return seq




def get_charge_constrained_pseq(
    n: int,
    pos_charge_ratio: float,
    neg_charge_ratio: float,
    unconstrained_logit: float = 10.0,
    constrained_pos_charge_residues: set = pos_charge_residues,
    constrained_neg_charge_residues: set = neg_charge_residues
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generates a probabilistic sequence (pseq) constrained by charge composition.

    This function constructs a sequence where at least a given fraction of residues
    are **positively** and **negatively** charged, ensuring that the remaining residues
    are distributed among the unconstrained amino acids. The returned probabilistic
    sequence (`pseq`) maintains these charge constraints.

    The function normalizes the probabilities so that the specified fraction of
    positively and negatively charged residues is maintained while the unconstrained
    residues are distributed according to `unconstrained_logit`.

    :param n: The sequence length.
    :type n: int
    :param pos_charge_ratio: Minimum fraction of pos. charged residues in the sequence.
    :type pos_charge_ratio: float
    :param neg_charge_ratio: Minimum fraction of neg. charged residues in the sequence.
    :type neg_charge_ratio: float
    :param unconstrained_logit: Weighting factor for unconstrained residues.
    :type unconstrained_logit: float
    :param constrained_pos_charge_residues: Set of pos. charged residues.
    :type constrained_pos_charge_residues: set
    :param constrained_neg_charge_residues: Set of neg. charged residues.
    :type constrained_neg_charge_residues: set

    :returns:
        - **pseq** (*jnp.ndarray*): A `(n, 20)` probabilistic sequence constrained by
                                    charge ratios.
        - **logits** (*jnp.ndarray*): The raw logits used to generate `pseq`.

    :raises AssertionError: If `pos_charge_ratio` or `neg_charge_ratio` are not
                            within `(0,1]` or if their sum exceeds `1.0`.
    """
    assert(pos_charge_ratio > 0.0 and pos_charge_ratio <= 1.0)
    assert(neg_charge_ratio > 0.0 and neg_charge_ratio <= 1.0)
    assert(pos_charge_ratio + neg_charge_ratio < 1.0) # to avoid division by 0

    pos_charged_res_idxs = [
        RES_ALPHA.index(res) for res in constrained_pos_charge_residues
    ]
    neg_charged_res_idxs = [
        RES_ALPHA.index(res) for res in constrained_neg_charge_residues
    ]

    unconstrained_residues = list()
    for res in RES_ALPHA:
        cond = (res not in constrained_pos_charge_residues) \
            and (res not in constrained_neg_charge_residues)
        if cond:
            unconstrained_residues.append(res)
    unconstrainted_ratio = 1 - (pos_charge_ratio + neg_charge_ratio)
    n_neg_charge_res = len(constrained_neg_charge_residues)
    n_pos_charge_res = len(constrained_pos_charge_residues)

    neg_charged_value = neg_charge_ratio / n_neg_charge_res * unconstrained_logit \
        * len(unconstrained_residues) / unconstrainted_ratio
    pos_charged_value = pos_charge_ratio / n_pos_charge_res * n_neg_charge_res \
        * neg_charged_value / neg_charge_ratio

    nuc = onp.zeros(len(RES_ALPHA))

    unconstrained_res_idxs = [RES_ALPHA.index(res) for res in unconstrained_residues]
    nuc[unconstrained_res_idxs] = unconstrained_logit
    nuc[pos_charged_res_idxs] = pos_charged_value
    nuc[neg_charged_res_idxs] = neg_charged_value
    nuc_normalized = nuc / nuc.sum()

    pseq = onp.vstack([nuc_normalized]*n)
    logits = onp.vstack([nuc]*n)

    return pseq, logits
