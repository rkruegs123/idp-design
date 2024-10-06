import pdb
import numpy as onp
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from pathlib import Path
from copy import deepcopy
import pprint
import time
from sparrow import Protein

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import vmap, jit, value_and_grad, lax, random
from jax_md import space, simulate
import optax

from idp_design.energy_prob import get_energy_fn
import idp_design.utils as utils
import idp_design.checkpoint as checkpoint
import idp_design.observable as observable

jax.config.update("jax_enable_x64", True)


checkpoint_every = 50
scan = checkpoint.get_scan(checkpoint_every)

def get_alba_rg(pseq, nsamples, key):
    alba_seqs, _ = utils.sample_discrete_seqs(pseq, nsamples, key)
    alba_rgs = list()
    for seq in tqdm(alba_seqs):
        P = Protein(seq)
        rg = P.predictor.radius_of_gyration()
        alba_rgs.append(rg)

    return onp.mean(alba_rgs), alba_rgs


def run(args):

    minimize_rg = args['minimize_rg']
    maximize_rg = args['maximize_rg']
    assert(not (minimize_rg and maximize_rg))

    key = random.PRNGKey(args['key'])

    n_sims = args['n_sims']
    n_eq_steps = args['n_eq_steps']
    n_sample_steps = args['n_sample_steps']
    sample_every = args['sample_every']
    assert(n_sample_steps % sample_every == 0)
    num_points_per_batch = n_sample_steps // sample_every
    n_ref_states = num_points_per_batch * n_sims

    run_name = args['run_name']
    kT = args['kt']
    beta = 1 / kT
    dt = args['dt']
    gamma = args['gamma']
    out_box_size = args['out_box_size']
    seq_length = args['seq_length']
    target_rg = args['target_rg']

    n_iters = args['n_iters']
    lr = args['lr']
    min_neff_factor = args['min_neff_factor']
    min_n_eff = int(n_ref_states * min_neff_factor)
    max_approx_iters = args['max_approx_iters']

    use_gumbel = args['use_gumbel']
    gumbel_end = args['gumbel_end']
    gumbel_start = args['gumbel_start']
    gumbel_temps = onp.linspace(gumbel_start, gumbel_end, n_iters)
    # gumbel_temps = onp.linspace(0.1, 3.0, 100)**3


    def normalize(logits, temp, norm_key):
        if use_gumbel:
            gumbel_weights = jax.random.gumbel(norm_key, logits.shape)
            # pseq = jax.nn.softmax((logits + gumbel_weights) / temp)
            pseq = jax.nn.softmax(logits / temp)
        else:
            pseq = jax.nn.softmax(logits)

        return pseq

    output_basedir = Path(args['output_basedir'])
    run_dir = output_basedir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)

    ref_traj_dir = run_dir / "ref_traj"
    ref_traj_dir.mkdir(parents=False, exist_ok=False)

    pseq_dir = run_dir / "pseq"
    pseq_dir.mkdir(parents=False, exist_ok=False)

    logits_dir = run_dir / "logits"
    logits_dir.mkdir(parents=False, exist_ok=False)

    img_dir = run_dir / "img"
    img_dir.mkdir(parents=False, exist_ok=False)

    log_dir = run_dir / "log"
    log_dir.mkdir(parents=False, exist_ok=False)

    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)


    bonded_nbrs = jnp.array([(i, i+1) for i in range(seq_length-1)])
    unbonded_nbrs = list()
    for pair in itertools.combinations(jnp.arange(seq_length), 2):
        unbonded_nbrs.append(pair)
    unbonded_nbrs = jnp.array(unbonded_nbrs)

    unbonded_nbrs_set = set([tuple(pr) for pr in onp.array(unbonded_nbrs)])
    bonded_nbrs_set = set([tuple(pr) for pr in onp.array(bonded_nbrs)])
    unbonded_nbrs = jnp.array(list(unbonded_nbrs_set - bonded_nbrs_set))


    displacement_fn, shift_fn = space.free()

    subterms_fn, energy_fn = get_energy_fn(bonded_nbrs, unbonded_nbrs, displacement_fn)
    energy_fn = jit(energy_fn)
    mapped_energy_fn = vmap(energy_fn, (0, None)) # To evaluate a set of states for a given pseq


    @jit
    def eq_fn(eq_key, R, pseq, mass):
        init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)
        init_state = init_fn(eq_key, R, pseq=pseq, mass=mass)
        def fori_step_fn(t, state):
            return step_fn(state, pseq=pseq)
        fori_step_fn = jit(fori_step_fn)

        eq_state = lax.fori_loop(0, n_eq_steps, fori_step_fn, init_state)
        return eq_state.position

    @jit
    def sample_fn(sample_key, R_eq, pseq, mass):
        init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)
        init_state = init_fn(sample_key, R_eq, pseq=pseq, mass=mass)

        def fori_step_fn(t, state):
            return step_fn(state, pseq=pseq)
        fori_step_fn = jit(fori_step_fn)

        @jit
        def scan_fn(state, step):
            state = lax.fori_loop(0, sample_every, fori_step_fn, state)
            return state, state.position

        _, traj = lax.scan(scan_fn, init_state, jnp.arange(num_points_per_batch))
        return traj

    @jit
    def batch_sim(ref_key, R, pseq, mass):

        ref_key, eq_key = random.split(ref_key)
        eq_keys = random.split(eq_key, n_sims)
        eq_states = vmap(eq_fn, (0, None, None, None))(eq_keys, R, pseq, mass)

        sample_keys = random.split(ref_key, n_sims)
        sample_trajs = vmap(sample_fn, (0, 0, None, None))(sample_keys, eq_states, pseq, mass)

        sample_traj = sample_trajs.reshape(-1, seq_length, 3)
        return sample_traj


    res_masses = utils.masses
    def get_ref_states(params, i, R, iter_key, temp):
        curr_logits = params['logits']
        iter_key, norm_key = random.split(iter_key)
        curr_pseq = normalize(curr_logits, temp, norm_key)

        iter_dir = ref_traj_dir / f"iter{i}"
        iter_dir.mkdir(parents=False, exist_ok=False)

        curr_mass = utils.get_pseq_mass(curr_pseq, res_masses=res_masses)

        iter_key, batch_key = random.split(iter_key)
        start = time.time()
        sample_traj = batch_sim(batch_key, R, curr_pseq, curr_mass)
        end = time.time()
        print(f"- Batched simulation took {end - start} seconds")
        sample_traj = utils.tree_stack(sample_traj)

        utils.dump_pos(sample_traj, iter_dir / "traj.pos", box_size=out_box_size)

        sample_rgs = vmap(observable.rg, (0, None, None))(sample_traj, curr_mass, displacement_fn)
        mean_rg = onp.mean(sample_rgs)

        sample_energies = mapped_energy_fn(sample_traj, curr_pseq)


        plt.plot(sample_rgs)
        plt.savefig(iter_dir / "rg_traj.png")
        plt.clf()

        sns.histplot(sample_rgs)
        plt.savefig(iter_dir / "rg_hist.png")
        plt.clf()

        num_rgs = sample_rgs.shape[0]
        running_avg_rgs = onp.cumsum(sample_rgs) / onp.arange(1, num_rgs+1)
        plt.plot(running_avg_rgs)
        plt.savefig(iter_dir / "running_avg.png")
        plt.clf()


        # Get ALBATROSS avg.
        iter_key, alba_key = random.split(iter_key)
        nsamples = 1000
        alba_mean_rg, alba_rgs = get_alba_rg(curr_pseq, nsamples, alba_key)

        running_avg_alba = onp.cumsum(alba_rgs) / onp.arange(1, nsamples+1)
        plt.plot(running_avg_alba, label="ALBATROSS samples")
        plt.axhline(y=mean_rg, linestyle="--", label="From pseq")
        plt.legend()
        plt.savefig(iter_dir / "alba_running_avg.png")
        plt.clf()

        sns.kdeplot(alba_rgs, label=f"ALBATROSS sample dist")
        plt.axvline(x=mean_rg, linestyle="--", label="From pseq")
        plt.legend()
        plt.savefig(iter_dir / "alba_kde.png")
        plt.clf()



        summary_str = ""
        summary_str += f"Mean Rg: {mean_rg}\n"
        summary_str += f"ALBATROSS Rg: {alba_mean_rg}\n"
        with open(iter_dir / "summary.txt", "w+") as f:
            f.write(summary_str)

        return sample_traj, sample_energies, jnp.array(sample_rgs), mean_rg, alba_mean_rg

    def loss_fn(params, ref_states, ref_energies, ref_rgs, temp, loss_key):
        logits = params['logits']
        # pseq = jax.nn.softmax(logits)
        loss_key, norm_key = random.split(loss_key)
        pseq = normalize(logits, temp, norm_key)

        energy_scan_fn = lambda state, ts: (None, energy_fn(ts, pseq=pseq))
        _, new_energies = scan(energy_scan_fn, None, ref_states)
        # new_energies = mapped_energy_fn(ref_states, pseq)

        weights, n_eff = utils.compute_weights(ref_energies, new_energies, beta)
        weighted_rgs = weights * ref_rgs # element-wise multiplication
        expected_rg = jnp.sum(weighted_rgs)


        if maximize_rg:
            loss = -jnp.sqrt((expected_rg)**2)
        elif minimize_rg:
            loss = jnp.sqrt((expected_rg)**2)
        else:
            mse = (expected_rg - target_rg)**2
            rmse = jnp.sqrt(mse)
            loss = rmse

        return loss, (n_eff, expected_rg, pseq)
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    grad_fn = jit(grad_fn)

    init_logits = onp.full((seq_length, 20), 100.0)
    # init_logits = onp.full((seq_length, 20), 1000.0)
    init_logits = jnp.array(init_logits, dtype=jnp.float64)

    # Setup the optimization
    params = {"logits": init_logits}
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    R_init = list()
    init_spring_r0 = utils.spring_r0
    for i in range(seq_length):
        R_init.append([out_box_size/2, out_box_size/2, out_box_size/2+init_spring_r0*i])
    R_init = jnp.array(R_init)

    all_ref_iters = list()
    all_ref_rgs = list()
    all_ref_alba_rgs = list()

    print(f"Generating initial reference states and energies...")
    key, iter_key = random.split(key)
    ref_states, ref_energies, ref_rgs, mean_rg, alba_mean_rg = get_ref_states(params, 0, R_init, iter_key, temp=gumbel_temps[0])
    all_ref_iters.append(0)
    all_ref_rgs.append(mean_rg)
    all_ref_alba_rgs.append(alba_mean_rg)

    loss_path = log_dir / "loss.txt"
    temp_path = log_dir / "temp.txt"
    neff_path = log_dir / "neff.txt"
    entropy_path = log_dir / "entropy.txt"
    rg_path = log_dir / "rg.txt"
    alba_rg_path = log_dir / "alba_rg.txt"
    ref_alba_rg_path = log_dir / "ref_alba_rg.txt"
    ref_rg_path = log_dir / "ref_rg.txt"
    ref_iter_path = log_dir / "ref_iter.txt"
    with open(ref_alba_rg_path, "a") as f:
        f.write(f"{alba_mean_rg}\n")
    with open(ref_rg_path, "a") as f:
        f.write(f"{mean_rg}\n")
    with open(ref_iter_path, "a") as f:
        f.write(f"{0}\n")
    grads_path = log_dir / "grads.txt"
    argmax_seq_path = log_dir / "argmax_seq.txt"
    argmax_seq_scaled_path = log_dir / "argmax_seq_scaled.txt"
    plot_every = 10
    all_rgs = list()
    num_resample_iters = 0
    for i in tqdm(range(n_iters), desc="Iteration"):
        key, loss_key = random.split(key)
        (loss, aux), grads = grad_fn(params, ref_states, ref_energies, ref_rgs, gumbel_temps[i], loss_key)
        n_eff = aux[0]
        num_resample_iters += 1

        if n_eff < min_n_eff or num_resample_iters > max_approx_iters:
            print(f"N_eff was {n_eff}... resampling reference states...")
            num_resample_iters = 0

            key, iter_key = random.split(key)
            ref_states, ref_energies, ref_rgs, mean_rg, alba_mean_rg = get_ref_states(
                params, i, utils.recenter(ref_states[-1], out_box_size), iter_key, gumbel_temps[i])
            all_ref_iters.append(i)
            all_ref_rgs.append(mean_rg)
            all_ref_alba_rgs.append(alba_mean_rg)
            with open(ref_alba_rg_path, "a") as f:
                f.write(f"{alba_mean_rg}\n")
            with open(ref_rg_path, "a") as f:
                f.write(f"{mean_rg}\n")
            with open(ref_iter_path, "a") as f:
                f.write(f"{i}\n")

            plt.scatter(all_ref_iters, all_ref_rgs, color="red", label="Pseq")
            plt.plot(all_ref_iters, all_ref_rgs, linestyle="--", color="red")
            plt.scatter(all_ref_iters, all_ref_alba_rgs, color="green", label="ALBATROSS")
            plt.plot(all_ref_iters, all_ref_alba_rgs, linestyle="--", color="green")
            plt.legend()
            plt.savefig(img_dir / f"convergence_i{i}.png")
            plt.clf()

            (loss, aux), grads = grad_fn(params, ref_states, ref_energies, ref_rgs, gumbel_temps[i], loss_key)
        (n_eff, expected_rg, pseq) = aux

        with open(loss_path, "a") as f:
            f.write(f"{loss}\n")
        with open(entropy_path, "a") as f:
            pos_entropies = jnp.mean(jsp.special.entr(pseq), axis=1)
            avg_pos_entropy = onp.mean(pos_entropies)
            f.write(f"{avg_pos_entropy}\n")
        with open(temp_path, "a") as f:
            f.write(f"{gumbel_temps[i]}\n")
        with open(neff_path, "a") as f:
            f.write(f"{n_eff}\n")
        with open(grads_path, "a") as f:
            logits_grads = grads['logits']
            f.write(f"{pprint.pformat(logits_grads)}\n")
        with open(rg_path, "a") as f:
            f.write(f"{expected_rg}\n")
        key, alba_key = random.split(key)
        curr_alba_rg, _ = get_alba_rg(pseq, 1000, alba_key)
        with open(alba_rg_path, "a") as f:
            f.write(f"{curr_alba_rg}\n")

        all_rgs.append(expected_rg)

        pseq_fpath = pseq_dir / f"pseq_i{i}.npy"
        jnp.save(pseq_fpath, pseq, allow_pickle=False)

        logits_fpath = logits_dir / f"logits_i{i}.npy"
        jnp.save(logits_fpath, params['logits'], allow_pickle=False)

        max_residues = jnp.argmax(pseq, axis=1)
        argmax_seq = ''.join([utils.RES_ALPHA[res_idx] for res_idx in max_residues])
        with open(argmax_seq_path, "a") as f:
            f.write(f"{argmax_seq}\n")

        argmax_seq_scaled = ''.join([argmax_seq[r_idx].lower() if pseq[r_idx, max_residues[r_idx]] < 0.5 else argmax_seq[r_idx] for r_idx in range(len(argmax_seq))])
        with open(argmax_seq_scaled_path, "a") as f:
            f.write(f"{argmax_seq_scaled}\n")

        if i % plot_every == 0 and i:
            entropies = jnp.mean(jsp.special.entr(pseq), axis=1)
            plt.bar(jnp.arange(pseq.shape[0]), entropies)
            plt.savefig(img_dir / f"entropies_i{i}.png")
            plt.clf()

            plt.plot(all_rgs)
            plt.xlabel("Iteration")
            plt.ylabel("Rg")
            plt.axhline(y=target_rg, linestyle='--', label="Target Rg", color='red')
            plt.legend()
            plt.savefig(img_dir / f"rg_i{i}.png")
            plt.clf()


        print(f"Loss: {loss}")

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    pseq_fpath = pseq_dir / f"pseq_final.npy"
    jnp.save(pseq_fpath, pseq, allow_pickle=False)


    obj_dir = run_dir / "obj"
    obj_dir.mkdir(parents=False, exist_ok=False)
    onp.save(obj_dir / "ref_iters.npy", onp.array(all_ref_iters), allow_pickle=False)
    onp.save(obj_dir / "ref_rgs.npy", onp.array(all_ref_rgs), allow_pickle=False)
    onp.save(obj_dir / "ref_alba_rgs.npy", onp.array(all_ref_alba_rgs), allow_pickle=False)

def get_parser():

    parser = argparse.ArgumentParser(description="DiffTRE for IDP design")

    parser.add_argument('--output-basedir', type=str, help='Output base directory',
                        default="output/"
    )
    parser.add_argument('--run-name', type=str, help='Run name')

    parser.add_argument('--n-iters', type=int, default=100,
                        help="Number of iterations of gradient descent")
    parser.add_argument('--lr', type=float, default=0.1,
                        help="Learning rate")
    parser.add_argument('--min-neff-factor', type=float, default=0.95,
                        help="Factor for determining min Neff")
    parser.add_argument('--max-approx-iters', type=int, default=3,
                        help="Maximum number of iterations before resampling")
    parser.add_argument('--seq-length', type=int, default=100,
                        help="Sequence length")
    parser.add_argument('--target-rg', type=float, default=0.0,
                        help="Target radius of gyration in Angstroms")

    parser.add_argument('--use-gumbel', action='store_true',
                        help="If true, will use gumbel softmax with an annealing temperature")
    parser.add_argument('--gumbel-start', type=float, default=1.0,
                        help="Starting temperature for gumbel softmax")
    parser.add_argument('--gumbel-end', type=float, default=0.01,
                        help="End temperature for gumbel softmax")

    # Simulation arguments
    parser.add_argument('--key', type=int, default=0)
    parser.add_argument('--out-box-size', type=float, default=200.0,
                        help="Length of the box for injavis visualization")
    parser.add_argument('--n-sims', type=int, default=5,
                        help="Number of independent simulations")
    parser.add_argument('--n-eq-steps', type=int, default=10000,
                        help="Number of equilibration steps")
    parser.add_argument('--n-sample-steps', type=int, default=200000,
                        help="Number of steps from which to sample states")
    parser.add_argument('--sample-every', type=int, default=1000,
                        help="Frequency of sampling reference states.")
    parser.add_argument('--kt', type=float, default=300*utils.kb,
                        help="Temperature")
    parser.add_argument('--dt', type=float, default=0.2,
                        help="Time step")
    parser.add_argument('--gamma', type=float, default=0.001,
                        help="Friction coefficient for Langevin integrator")

    parser.add_argument('--maximize-rg', action='store_true',
                        help="If true, just try to maximize Rg")
    parser.add_argument('--minimize-rg', action='store_true',
                        help="If true, just try to minimize Rg")


    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
