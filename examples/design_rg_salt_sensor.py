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

prefix = jnp.array(utils.seq_to_one_hot("M" + "H"*6))
prefix_length = prefix.shape[0]

def run(args):
    key = random.PRNGKey(args['key'])

    n_sims = args['n_sims']
    n_eq_steps = args['n_eq_steps']
    n_sample_steps = args['n_sample_steps']
    sample_every = args['sample_every']
    assert(n_sample_steps % sample_every == 0)
    num_points_per_batch = n_sample_steps // sample_every
    n_ref_states = num_points_per_batch * n_sims

    mode = args["mode"]

    run_name = args['run_name']
    kT = args['kt']
    beta = 1 / kT

    salt_hi = args['salt_hi']
    kappa_hi = utils.get_kappa(salt_hi, use_gg=True)
    salt_lo = args['salt_lo']
    kappa_lo = utils.get_kappa(salt_lo, use_gg=True)

    dt = args['dt']
    gamma = args['gamma']
    out_box_size = args['out_box_size']
    seq_length = args['seq_length']

    n_iters = args['n_iters']
    lr = args['lr']
    min_neff_factor = args['min_neff_factor']
    min_n_eff = int(n_ref_states * 2 * min_neff_factor) # Multiple by 2 because two temperatures
    max_approx_iters = args['max_approx_iters']

    gumbel_end = args['gumbel_end']
    gumbel_start = args['gumbel_start']
    gumbel_temps = onp.linspace(gumbel_start, gumbel_end, n_iters)

    def normalize_and_constrain(logits, temp):
        pseq = jax.nn.softmax(logits / temp)
        pseq = pseq.at[:prefix_length].set(prefix)

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
    mapped_energy_fn = vmap(lambda R, pseq, kappa: energy_fn(R, pseq, debye_kappa=kappa), (0, None, None)) # To evaluate a set of states for a given pseq and kappa


    @jit
    def eq_fn(eq_key, R, pseq, mass, kappa):
        init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)
        init_state = init_fn(eq_key, R, pseq=pseq, debye_kappa=kappa, mass=mass)
        def fori_step_fn(t, state):
            return step_fn(state, pseq=pseq, debye_kappa=kappa)
        fori_step_fn = jit(fori_step_fn)

        eq_state = lax.fori_loop(0, n_eq_steps, fori_step_fn, init_state)
        return eq_state.position

    @jit
    def sample_fn(sample_key, R_eq, pseq, mass, kappa):
        init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)
        init_state = init_fn(sample_key, R_eq, pseq=pseq, debye_kappa=kappa, mass=mass)

        def fori_step_fn(t, state):
            return step_fn(state, pseq=pseq, debye_kappa=kappa)
        fori_step_fn = jit(fori_step_fn)

        @jit
        def scan_fn(state, step):
            state = lax.fori_loop(0, sample_every, fori_step_fn, state)
            return state, state.position

        _, traj = lax.scan(scan_fn, init_state, jnp.arange(num_points_per_batch))
        return traj

    @jit
    def batch_sim(ref_key, R, pseq, mass, kappa):

        ref_key, eq_key = random.split(ref_key)
        eq_keys = random.split(eq_key, n_sims)
        eq_states = vmap(eq_fn, (0, None, None, None, None))(eq_keys, R, pseq, mass, kappa)

        sample_keys = random.split(ref_key, n_sims)
        sample_trajs = vmap(sample_fn, (0, 0, None, None, None))(sample_keys, eq_states, pseq, mass, kappa)

        sample_traj = sample_trajs.reshape(-1, seq_length, 3)
        return sample_traj


    def get_ref_states(params, i, R_hi, R_lo, iter_key, temp):
        curr_logits = params['logits']
        curr_pseq = normalize_and_constrain(curr_logits, temp)

        iter_dir = ref_traj_dir / f"iter{i}"
        iter_dir.mkdir(parents=False, exist_ok=False)

        curr_mass = utils.get_pseq_mass(curr_pseq)

        sim_info = dict()

        for salt_name, kappa, R in [("hi", kappa_hi, R_hi), ("lo", kappa_lo, R_lo)]:

            salt_dir = iter_dir / f"{salt_name}"
            salt_dir.mkdir(parents=False, exist_ok=False)

            iter_key, batch_key = random.split(iter_key)
            start = time.time()
            sample_traj = batch_sim(batch_key, R, curr_pseq, curr_mass, kappa)
            end = time.time()
            print(f"- Batched simulation took {end - start} seconds")
            sample_traj = utils.tree_stack(sample_traj)

            # utils.dump_pos(sample_traj, salt_dir / "traj.pos", box_size=out_box_size)

            sample_rgs = vmap(observable.rg, (0, None, None))(sample_traj, curr_mass, displacement_fn)
            mean_rg = onp.mean(sample_rgs)

            sample_energies = mapped_energy_fn(sample_traj, curr_pseq, kappa)

            plt.plot(sample_rgs)
            plt.savefig(salt_dir / "rg_traj.png")
            plt.clf()

            sns.histplot(sample_rgs)
            plt.savefig(salt_dir / "rg_hist.png")
            plt.clf()

            num_rgs = sample_rgs.shape[0]
            running_avg_rgs = onp.cumsum(sample_rgs) / onp.arange(1, num_rgs+1)
            plt.plot(running_avg_rgs)
            plt.savefig(salt_dir / "running_avg.png")
            plt.clf()

            summary_str = ""
            summary_str += f"Mean Rg: {mean_rg}\n"
            with open(salt_dir / "summary.txt", "w+") as f:
                f.write(summary_str)

            sim_info[salt_name] = {
                "ref_traj": sample_traj,
                "ref_energies": sample_energies,
                "ref_rgs": jnp.array(sample_rgs),
                "mean_rg": mean_rg,
            }

        summary_str = ""
        mean_rg_hi = sim_info["hi"]["mean_rg"]
        mean_rg_lo = sim_info["lo"]["mean_rg"]
        summary_str = ""
        summary_str += f"Mean Rg, Hi: {mean_rg_hi}\n"
        summary_str += f"Mean Rg, Lo: {mean_rg_lo}\n"
        if mode == "expander":
            diff = mean_rg_hi - mean_rg_lo
        elif mode == "contractor":
            diff = mean_rg_lo - mean_rg_hi
        elif mode == "neutral":
            diff = jnp.sqrt((mean_rg_lo - mean_rg_hi)**2)
        else:
            raise RuntimeError(f"Invalid mode: {mode}")
        summary_str += f"Difference: {diff}\n"
        with open(iter_dir / "summary.txt", "w+") as f:
            f.write(summary_str)

        return sim_info

    # def loss_fn(params, ref_states, ref_energies, ref_rgs, temp, loss_key):
    def loss_fn(params, sim_info, temp, loss_key):
        logits = params['logits']
        pseq = normalize_and_constrain(logits, temp)

        # Hi temperature
        ref_states = sim_info["hi"]["ref_traj"]
        ref_energies = sim_info["hi"]["ref_energies"]
        ref_rgs = sim_info["hi"]["ref_rgs"]

        energy_scan_fn = lambda state, ts: (None, energy_fn(ts, pseq=pseq, debye_kappa=kappa_hi))
        _, new_energies = scan(energy_scan_fn, None, ref_states)

        weights_hi, n_eff_hi = utils.compute_weights(ref_energies, new_energies, beta)
        weighted_rgs_hi = weights_hi * ref_rgs # element-wise multiplication
        expected_rg_hi = jnp.sum(weighted_rgs_hi)


        # Lo temperature
        ref_states = sim_info["lo"]["ref_traj"]
        ref_energies = sim_info["lo"]["ref_energies"]
        ref_rgs = sim_info["lo"]["ref_rgs"]

        energy_scan_fn = lambda state, ts: (None, energy_fn(ts, pseq=pseq, debye_kappa=kappa_lo))
        _, new_energies = scan(energy_scan_fn, None, ref_states)

        weights_lo, n_eff_lo = utils.compute_weights(ref_energies, new_energies, beta)
        weighted_rgs_lo = weights_lo * ref_rgs # element-wise multiplication
        expected_rg_lo = jnp.sum(weighted_rgs_lo)

        if mode == "expander":
            diff = expected_rg_hi - expected_rg_lo
            loss = -diff
        elif mode == "contractor":
            diff = expected_rg_lo - expected_rg_hi
            loss = -diff
        elif mode == "neutral":
            diff = jnp.sqrt((expected_rg_lo - expected_rg_hi)**2)
            loss = diff
        else:
            raise RuntimeError(f"Invalid mode: {mode}")
        # diff = expected_rg_lo - expected_rg_hi

        return loss, (n_eff_hi, expected_rg_hi, n_eff_lo, expected_rg_lo, pseq)
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    grad_fn = jit(grad_fn)

    init_logits = onp.full((seq_length, 20), 100.0)
    init_logits = jnp.array(init_logits, dtype=jnp.float64)

    # Setup the optimization
    params = {"logits": init_logits}
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    R_init = list()
    for i in range(seq_length):
        R_init.append([out_box_size/2, out_box_size/2, out_box_size/2+utils.spring_r0*i])
    R_init = jnp.array(R_init)

    all_ref_iters = list()
    all_ref_rgs_hi = list()
    all_ref_rgs_lo = list()
    all_ref_diffs = list()

    print(f"Generating initial reference states and energies...")
    key, iter_key = random.split(key)
    sim_info = get_ref_states(params, 0, R_init, R_init, iter_key, temp=gumbel_temps[0])
    all_ref_iters.append(0)
    expected_rg_hi = sim_info["hi"]["mean_rg"]
    expected_rg_lo = sim_info["lo"]["mean_rg"]
    all_ref_rgs_hi.append(expected_rg_hi)
    all_ref_rgs_lo.append(expected_rg_lo)

    if mode == "expander":
        diff = expected_rg_hi - expected_rg_lo
    elif mode == "contractor":
        diff = expected_rg_lo - expected_rg_hi
    elif mode == "neutral":
        diff = jnp.sqrt((expected_rg_lo - expected_rg_hi)**2)
    else:
        raise RuntimeError(f"Invalid mode: {mode}")

    all_ref_diffs.append(diff)

    loss_path = log_dir / "loss.txt"
    temp_path = log_dir / "temp.txt"
    neff_hi_path = log_dir / "neff_hi.txt"
    neff_lo_path = log_dir / "neff_lo.txt"
    entropy_path = log_dir / "entropy.txt"
    rg_hi_path = log_dir / "rg_hi.txt"
    ref_rg_hi_path = log_dir / "ref_rg_hi.txt"
    rg_lo_path = log_dir / "rg_lo.txt"
    ref_rg_lo_path = log_dir / "ref_rg_lo.txt"
    ref_diff_path = log_dir / "ref_diff.txt"
    ref_iter_path = log_dir / "ref_iter.txt"
    with open(ref_rg_hi_path, "a") as f:
        mean_rg = sim_info["hi"]["mean_rg"]
        f.write(f"{mean_rg}\n")
    with open(ref_rg_lo_path, "a") as f:
        mean_rg = sim_info["lo"]["mean_rg"]
        f.write(f"{mean_rg}\n")
    with open(ref_diff_path, "a") as f:
        mean_rg_lo = sim_info["lo"]["mean_rg"]
        mean_rg_hi = sim_info["hi"]["mean_rg"]

        if mode == "expander":
            diff = mean_rg_hi - mean_rg_lo
        elif mode == "contractor":
            diff = mean_rg_lo - mean_rg_hi
        elif mode == "neutral":
            diff = jnp.sqrt((mean_rg_lo - mean_rg_hi)**2)
        else:
            raise RuntimeError(f"Invalid mode: {mode}")

        f.write(f"{diff}\n")
    with open(ref_iter_path, "a") as f:
        f.write(f"{0}\n")
    grads_path = log_dir / "grads.txt"
    argmax_seq_path = log_dir / "argmax_seq.txt"
    argmax_seq_scaled_path = log_dir / "argmax_seq_scaled.txt"
    plot_every = 10
    all_rgs_hi = list()
    all_rgs_lo = list()
    all_diffs = list()
    num_resample_iters = 0
    for i in tqdm(range(n_iters), desc="Iteration"):
        key, loss_key = random.split(key)
        (loss, aux), grads = grad_fn(params, sim_info, gumbel_temps[i], loss_key)
        n_eff_hi = aux[0]
        n_eff_lo = aux[2]
        n_eff = n_eff_hi + n_eff_lo
        num_resample_iters += 1

        if n_eff < min_n_eff or num_resample_iters > max_approx_iters:
            print(f"N_eff was {n_eff}... resampling reference states...")
            num_resample_iters = 0

            key, iter_key = random.split(key)
            sim_info = get_ref_states(
                params, i, utils.recenter(sim_info["hi"]["ref_traj"][-1], out_box_size), utils.recenter(sim_info["lo"]["ref_traj"][-1], out_box_size), iter_key, gumbel_temps[i])
            all_ref_iters.append(i)
            expected_rg_hi = sim_info["hi"]["mean_rg"]
            expected_rg_lo = sim_info["lo"]["mean_rg"]
            all_ref_rgs_hi.append(expected_rg_hi)
            all_ref_rgs_lo.append(expected_rg_lo)
            if mode == "expander":
                diff = expected_rg_hi - expected_rg_lo
            elif mode == "contractor":
                diff = expected_rg_lo - expected_rg_hi
            elif mode == "neutral":
                diff = jnp.sqrt((expected_rg_lo - expected_rg_hi)**2)
            else:
                raise RuntimeError(f"Invalid mode: {mode}")
            all_ref_diffs.append(diff)
            with open(ref_rg_hi_path, "a") as f:
                mean_rg = sim_info["hi"]["mean_rg"]
                f.write(f"{mean_rg}\n")
            with open(ref_rg_lo_path, "a") as f:
                mean_rg = sim_info["lo"]["mean_rg"]
                f.write(f"{mean_rg}\n")
            with open(ref_diff_path, "a") as f:
                f.write(f"{diff}\n")
            with open(ref_iter_path, "a") as f:
                f.write(f"{i}\n")

            plt.scatter(all_ref_iters, all_ref_rgs_hi, color="red", label="Pseq")
            plt.plot(all_ref_iters, all_ref_rgs_hi, linestyle="--", color="red")
            plt.legend()
            plt.savefig(img_dir / f"ref_rg_hi_i{i}.png")
            plt.clf()

            plt.scatter(all_ref_iters, all_ref_rgs_lo, color="red", label="Pseq")
            plt.plot(all_ref_iters, all_ref_rgs_lo, linestyle="--", color="red")
            plt.legend()
            plt.savefig(img_dir / f"ref_rg_lo_i{i}.png")
            plt.clf()

            plt.scatter(all_ref_iters, all_ref_diffs, color="red", label="Pseq")
            plt.plot(all_ref_iters, all_ref_diffs, linestyle="--", color="red")
            plt.legend()
            plt.savefig(img_dir / f"ref_diffs_i{i}.png")
            plt.clf()

            (loss, aux), grads = grad_fn(params, sim_info, gumbel_temps[i], loss_key)
        (n_eff_hi, expected_rg_hi, n_eff_lo, expected_rg_lo, pseq) = aux

        with open(loss_path, "a") as f:
            f.write(f"{loss}\n")
        with open(entropy_path, "a") as f:
            pos_entropies = jnp.mean(jsp.special.entr(pseq), axis=1)
            avg_pos_entropy = onp.mean(pos_entropies)
            f.write(f"{avg_pos_entropy}\n")
        with open(temp_path, "a") as f:
            f.write(f"{gumbel_temps[i]}\n")
        with open(neff_hi_path, "a") as f:
            f.write(f"{n_eff_hi}\n")
        with open(neff_lo_path, "a") as f:
            f.write(f"{n_eff_lo}\n")
        with open(grads_path, "a") as f:
            logits_grads = grads['logits']
            f.write(f"{pprint.pformat(logits_grads)}\n")
        with open(rg_hi_path, "a") as f:
            f.write(f"{expected_rg_hi}\n")
        with open(rg_lo_path, "a") as f:
            f.write(f"{expected_rg_lo}\n")

        all_rgs_hi.append(expected_rg_hi)
        all_rgs_lo.append(expected_rg_lo)
        if mode == "expander":
            diff = expected_rg_hi - expected_rg_lo
        elif mode == "contractor":
            diff = expected_rg_lo - expected_rg_hi
        elif mode == "neutral":
            diff = jnp.sqrt((expected_rg_lo - expected_rg_hi)**2)
        else:
            raise RuntimeError(f"Invalid mode: {mode}")
        all_diffs.append(diff)

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

            plt.plot(all_diffs)
            plt.xlabel("Iteration")
            plt.ylabel("Rg Difference")
            plt.savefig(img_dir / f"diff_i{i}.png")
            plt.clf()


        print(f"Loss: {loss}")

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    pseq_fpath = pseq_dir / f"pseq_final.npy"
    jnp.save(pseq_fpath, pseq, allow_pickle=False)


    obj_dir = run_dir / "obj"
    obj_dir.mkdir(parents=False, exist_ok=False)
    onp.save(obj_dir / "ref_iters.npy", onp.array(all_ref_iters), allow_pickle=False)
    onp.save(obj_dir / "ref_rgs_lo.npy", onp.array(all_ref_rgs_lo), allow_pickle=False)
    onp.save(obj_dir / "ref_rgs_hi.npy", onp.array(all_ref_rgs_hi), allow_pickle=False)

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
    parser.add_argument('--min-neff-factor', type=float, default=0.90,
                        help="Factor for determining min Neff")
    parser.add_argument('--max-approx-iters', type=int, default=5,
                        help="Maximum number of iterations before resampling")
    parser.add_argument('--seq-length', type=int, default=100,
                        help="Sequence length")

    parser.add_argument('--gumbel-start', type=float, default=1.0,
                        help="Starting temperature for gumbel softmax")
    parser.add_argument('--gumbel-end', type=float, default=0.01,
                        help="End temperature for gumbel softmax")

    parser.add_argument('--mode', type=str, default="expander",
                        choices=["expander", "contractor", "neutral"],
                        help="Optimization target")

    # Simulation arguments
    parser.add_argument('--key', type=int, default=0)
    parser.add_argument('--out-box-size', type=float, default=200.0,
                        help="Length of the box for injavis visualization")
    parser.add_argument('--n-sims', type=int, default=15,
                        help="Number of independent simulations")
    parser.add_argument('--n-eq-steps', type=int, default=10000,
                        help="Number of equilibration steps")
    parser.add_argument('--n-sample-steps', type=int, default=500000,
                        help="Number of steps from which to sample states")
    parser.add_argument('--sample-every', type=int, default=1000,
                        help="Frequency of sampling reference states.")
    parser.add_argument('--salt-hi', type=float, default=450,
                        help="High salt concentration in mM")
    parser.add_argument('--salt-lo', type=float, default=150,
                        help="Low salt concentration in mM")
    parser.add_argument('--dt', type=float, default=0.2,
                        help="Time step")
    parser.add_argument('--kt', type=float, default=300*utils.kb,
                        help="Temperature")
    parser.add_argument('--gamma', type=float, default=0.001,
                        help="Friction coefficient for Langevin integrator")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
