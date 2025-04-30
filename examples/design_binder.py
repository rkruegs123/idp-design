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

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy as jsp
from jax import vmap, jit, value_and_grad, lax, random, pmap
from jax_md import space, simulate
import optax

from idp_design.energy_prob import get_energy_fn
import idp_design.utils as utils
import idp_design.checkpoint as checkpoint
import idp_design.observable as observable



checkpoint_every = 50
scan = checkpoint.get_scan(checkpoint_every)


def run(args):
    key = random.PRNGKey(args['key'])

    n_devices = args['n_devices']
    assert(n_devices == jax.device_count(backend='gpu'))
    n_sims_per_device = args['n_sims_per_device']
    n_sims = n_devices * n_sims_per_device
    n_eq_steps = args['n_eq_steps']
    n_sample_steps = args['n_sample_steps']
    sample_every = args['sample_every']
    assert(n_sample_steps % sample_every == 0)
    num_points_per_batch = n_sample_steps // sample_every
    n_ref_states = num_points_per_batch * n_sims

    max_dist = args['max_dist']
    spring_k = args['spring_k']

    def bias_fn(dist):
        return jnp.where(dist >= max_dist, spring_k*(dist - max_dist)**2, 0.0)

    run_name = args['run_name']
    kT = args['kt']
    beta = 1 / kT
    dt = args['dt']
    gamma = args['gamma']
    out_box_size = args['out_box_size']
    binder_length = args['binder_length']
    substrate = args['substrate']
    len_substrate = len(substrate)
    substrate_oh = jnp.array(utils.seq_to_one_hot(substrate))
    n_total = binder_length + len_substrate

    n_iters = args['n_iters']
    lr = args['lr']
    min_neff_factor = args['min_neff_factor']
    min_n_eff = int(n_ref_states * min_neff_factor)
    max_approx_iters = args['max_approx_iters']


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


    substrate_pseq = jnp.array(utils.seq_to_one_hot(substrate))
    displacement_fn, shift_fn = space.free()

    # Setup for optimization
    bonded_nbrs = [(i, i+1) for i in range(len_substrate-1)] \
        + [(len_substrate+i, len_substrate+i+1) for i in range(binder_length-1)]
    bonded_nbrs = jnp.array(bonded_nbrs)
    unbonded_nbrs = list()
    for pair in itertools.combinations(jnp.arange(n_total), 2):
        unbonded_nbrs.append(pair)
    unbonded_nbrs = jnp.array(unbonded_nbrs)

    unbonded_nbrs_set = set([tuple(pr) for pr in onp.array(unbonded_nbrs)])
    bonded_nbrs_set = set([tuple(pr) for pr in onp.array(bonded_nbrs)])
    unbonded_nbrs = jnp.array(list(unbonded_nbrs_set - bonded_nbrs_set))

    _, base_energy_fn = get_energy_fn(bonded_nbrs, unbonded_nbrs, displacement_fn)
    base_energy_fn = jit(base_energy_fn)
    mapped_base_energy_fn = vmap(base_energy_fn, (0, None)) # To evaluate a set of states for a given pseq

    def interstrand_distance(R, pseq):

        substrate_pseq = pseq[:len_substrate]
        substrate_mass = utils.get_pseq_mass(substrate_pseq)
        substrate_com = observable.com(R[:len_substrate], substrate_mass)

        binder_pseq = pseq[len_substrate:]
        binder_mass = utils.get_pseq_mass(binder_pseq)
        binder_com = observable.com(R[len_substrate:], binder_mass)

        return space.distance(displacement_fn(substrate_com, binder_com))


    def biased_energy_fn(R, pseq):
        base_energy = base_energy_fn(R, pseq=pseq)

        dist = interstrand_distance(R, pseq)
        bias_energy = bias_fn(dist)

        return base_energy + bias_energy

    gumbel_end = args['gumbel_end']
    gumbel_start = args['gumbel_start']
    gumbel_temps = onp.linspace(gumbel_start, gumbel_end, n_iters)

    def normalize(logits, temp, norm_key):
        gumbel_weights = jax.random.gumbel(norm_key, logits.shape)
        # pseq = jax.nn.softmax((logits + gumbel_weights) / temp)
        pseq = jax.nn.softmax(logits / temp)

        return pseq

    def binder_logits_to_pseq(logits, temp, norm_key):
        binder_pseq = normalize(logits, temp, norm_key)
        # binder_pseq = jax.nn.softmax(logits)
        full_pseq = jnp.concatenate([substrate_oh, binder_pseq])
        return full_pseq

    @jit
    def eq_fn(eq_key, R, pseq, mass):
        init_fn, step_fn = simulate.nvt_langevin(biased_energy_fn, shift_fn, dt, kT, gamma)
        init_state = init_fn(eq_key, R, pseq=pseq, mass=mass)
        def fori_step_fn(t, state):
            return step_fn(state, pseq=pseq)
        fori_step_fn = jit(fori_step_fn)

        eq_state = lax.fori_loop(0, n_eq_steps, fori_step_fn, init_state)
        return eq_state.position

    @jit
    def sample_fn(sample_key, R_eq, pseq, mass):
        init_fn, step_fn = simulate.nvt_langevin(biased_energy_fn, shift_fn, dt, kT, gamma)
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
        eq_keys = random.split(eq_key, n_sims_per_device)
        eq_states = vmap(eq_fn, (0, None, None, None))(eq_keys, R, pseq, mass)

        sample_keys = random.split(ref_key, n_sims_per_device)
        sample_trajs = vmap(sample_fn, (0, 0, None, None))(sample_keys, eq_states, pseq, mass)

        sample_traj = sample_trajs.reshape(-1, n_total, 3)
        return sample_traj



    def get_ref_states(params, i, R, iter_key, temp):

        curr_logits = params['logits']
        iter_key, norm_key = random.split(iter_key)
        pseq = binder_logits_to_pseq(curr_logits, temp, norm_key)

        iter_dir = ref_traj_dir / f"iter{i}"
        iter_dir.mkdir(parents=False, exist_ok=False)

        utils.dump_pos(utils.tree_stack([R]), iter_dir / "init_state.pos", box_size=out_box_size)

        mass = utils.get_pseq_mass(pseq)

        iter_key, batch_key = random.split(iter_key)
        device_keys = random.split(batch_key, n_devices)
        start = time.time()
        sample_trajs = pmap(batch_sim, in_axes=(0, None, None, None))(device_keys, R, pseq, mass)
        # sample_traj = batch_sim(batch_key, R, pseq, mass)
        end = time.time()
        print(f"- Batched simulation took {end - start} seconds")
        sample_traj = sample_trajs.reshape(-1, n_total, 3)
        # sample_traj = utils.tree_stack(sample_traj)

        sample_energies = mapped_base_energy_fn(sample_traj, pseq)

        # utils.dump_pos(sample_traj, iter_dir / "traj.pos", box_size=out_box_size)

        sample_dists = vmap(interstrand_distance, (0, None))(sample_traj, pseq) # interstrand distances
        sample_biases = vmap(bias_fn)(sample_dists)
        sample_weights = jnp.exp(-beta * sample_biases)

        plt.plot(sample_dists)
        for i in range(n_sims):
            plt.axvline(x=num_points_per_batch*i, linestyle="--", color="red")
        plt.savefig(iter_dir / f"biased_dists_traj.png")
        plt.close()

        plt.hist(sample_dists, 25, histtype='bar', facecolor='blue')
        plt.savefig(iter_dir / f"biased_dists_hist.png")
        plt.clf()

        num_dists = sample_dists.shape[0]
        running_avg_dists = onp.cumsum(sample_dists) / onp.arange(1, num_dists+1)
        plt.plot(running_avg_dists)
        plt.savefig(iter_dir / "biased_running_avg_dist.png")
        plt.clf()

        biased_mean = sample_dists.mean()
        with open(iter_dir / "summary.txt", "w+") as f:
            f.write(f"Biased mean: {biased_mean}\n")
        inv_weights = 1 / sample_weights
        unbiased_mean = jnp.dot(sample_dists, inv_weights) / inv_weights.sum()



        with open(iter_dir / "summary.txt", "a") as f:
            f.write(f"Unbiased mean (interstrand distance): {unbiased_mean}\n")

        return sample_traj, sample_energies, sample_dists, unbiased_mean, sample_weights

    def loss_fn(params, ref_states, ref_energies, ref_dists, ref_ref_weights, temp, loss_key):
        logits = params['logits']
        loss_key, norm_key = random.split(loss_key)
        pseq = binder_logits_to_pseq(logits, temp, norm_key)

        energy_scan_fn = lambda state, ts: (None, base_energy_fn(ts, pseq=pseq))
        _, new_energies = scan(energy_scan_fn, None, ref_states)

        weights, n_eff = utils.compute_weights(ref_energies, new_energies, beta)

        inv_ref_weights = 1 / ref_weights
        scaled_inv_ref_weights = jnp.multiply(inv_ref_weights, weights)

        expected_dist = jnp.dot(ref_dists, scaled_inv_ref_weights) / scaled_inv_ref_weights.sum()

        loss = expected_dist

        return loss, (n_eff, expected_dist, pseq)
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    grad_fn = jit(grad_fn)

    init_logits = onp.full((binder_length, 20), 100.0)
    init_logits = jnp.array(init_logits, dtype=jnp.float64)

    params = {"logits": init_logits}

    R_init = list()
    substrate_init_x = out_box_size/2 - 10
    for i in range(len_substrate):
        R_init.append([substrate_init_x, out_box_size/2, out_box_size/2+utils.spring_r0*i])
    binder_init_x = out_box_size/2 + 10
    for i in range(binder_length):
        R_init.append([binder_init_x, out_box_size/2, out_box_size/2+utils.spring_r0*i])
    R_init = jnp.array(R_init)


    all_ref_iters = list()
    all_ref_dists = list()

    print(f"Generating initial reference states and energies...")
    key, sample_fe_key = random.split(key)
    ref_states, ref_energies, ref_dists, mean_dist, ref_weights = get_ref_states(params, 0, R_init, sample_fe_key, temp=gumbel_temps[0])
    all_ref_iters.append(0)
    all_ref_dists.append(mean_dist)

    loss_path = log_dir / "loss.txt"
    dist_path = log_dir / "dist.txt"
    grads_path = log_dir / "grads.txt"
    argmax_seq_path = log_dir / "argmax_seq.txt"
    argmax_seq_scaled_path = log_dir / "argmax_seq_scaled.txt"
    plot_every = 10
    all_dists = list()
    num_resample_iters = 0

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    for i in tqdm(range(n_iters), desc="Iteration"):
        key, loss_key = random.split(key)
        (loss, aux), grads = grad_fn(params, ref_states, ref_energies, ref_dists, ref_weights, gumbel_temps[i], loss_key)
        n_eff = aux[0]
        num_resample_iters += 1

        if n_eff < min_n_eff or num_resample_iters > max_approx_iters:
            print(f"Resampling reference states...")
            num_resample_iters = 0

            key, ref_states_key = random.split(key)

            ref_states, ref_energies, ref_dists, mean_dist, ref_weights = get_ref_states(params, i, utils.recenter(ref_states[-1], out_box_size), ref_states_key, temp=gumbel_temps[i])
            all_ref_iters.append(i)
            all_ref_dists.append(mean_dist)

            (loss, aux), grads = grad_fn(params, ref_states, ref_energies, ref_dists, ref_weights, gumbel_temps[i], loss_key)

        n_eff = aux[0]
        # expected_dist, pseq = aux[1], aux[2]
        expected_dist, pseq = aux[1:]

        with open(loss_path, "a") as f:
            f.write(f"{loss}\n")
        with open(dist_path, "a") as f:
            f.write(f"{expected_dist}\n")
        with open(grads_path, "a") as f:
            f.write(f"{pprint.pformat(grads)}\n")

        all_dists.append(expected_dist)

        max_residues = jnp.argmax(pseq, axis=1)
        argmax_seq = ''.join([utils.RES_ALPHA[res_idx] for res_idx in max_residues])
        with open(argmax_seq_path, "a") as f:
            f.write(f"{argmax_seq}\n")

        argmax_seq_scaled = ''.join([argmax_seq[r_idx].lower() if pseq[r_idx, max_residues[r_idx]] < 0.5 else argmax_seq[r_idx] for r_idx in range(len(argmax_seq))])
        with open(argmax_seq_scaled_path, "a") as f:
            f.write(f"{argmax_seq_scaled}\n")

        if i % plot_every == 0 and i:
            pseq_fpath = pseq_dir / f"pseq_i{i}.npy"
            jnp.save(pseq_fpath, pseq, allow_pickle=False)

            entropies = jnp.mean(jsp.special.entr(pseq), axis=1)
            plt.bar(jnp.arange(pseq.shape[0]), entropies)
            plt.savefig(img_dir / f"entropies_i{i}.png")
            plt.clf()

            type_dist = utils.pseq_to_type_distribution(pseq)
            type_entropies = jnp.mean(jsp.special.entr(type_dist), axis=1)
            plt.bar(jnp.arange(type_entropies.shape[0]), type_entropies)
            plt.savefig(img_dir / f"type_entropies_i{i}.png")
            plt.clf()

            plt.plot(all_dists)
            plt.xlabel("Iteration")
            plt.ylabel("Distance")
            plt.legend()
            plt.savefig(img_dir / f"distance_i{i}.png")
            plt.clf()

        print(f"Loss: {loss}")

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    pseq_fpath = pseq_dir / f"pseq_final.npy"
    jnp.save(pseq_fpath, pseq, allow_pickle=False)

    obj_dir = run_dir / "obj"
    obj_dir.mkdir(parents=False, exist_ok=False)
    onp.save(obj_dir / "ref_iters.npy", onp.array(all_ref_iters), allow_pickle=False)
    onp.save(obj_dir / "ref_dists.npy", onp.array(all_ref_dists), allow_pickle=False)

    return



def get_parser():

    parser = argparse.ArgumentParser(description="DiffTRE for IDP binder design")

    parser.add_argument('--output-basedir', type=str, help='Output base directory',
                        default="output/"
    )
    parser.add_argument('--run-name', type=str, required=True, help='Run name')

    parser.add_argument('--n-iters', type=int, default=100,
                        help="Number of iterations of gradient descent")
    parser.add_argument('--lr', type=float, default=0.1,
                        help="Learning rate")
    parser.add_argument('--min-neff-factor', type=float, default=0.95,
                        help="Factor for determining min Neff")
    parser.add_argument('--max-approx-iters', type=int, default=3,
                        help="Maximum number of iterations before resampling")
    parser.add_argument('--binder-length', type=int, default=30,
                        help="Binder sequence length")
    parser.add_argument('--substrate', type=str, required=True,
                        help="Sequence to bind")

    # Simulation arguments
    parser.add_argument('--key', type=int, default=0)
    parser.add_argument('--out-box-size', type=float, default=200.0,
                        help="Length of the box for injavis visualization")
    parser.add_argument('--n-sims-per-device', type=int, default=5,
                        help="Number of independent simulations")
    parser.add_argument('--n-eq-steps', type=int, default=25000,
                        help="Number of equilibration steps")
    parser.add_argument('--n-sample-steps', type=int, default=750000,
                        help="Number of steps from which to sample states")
    parser.add_argument('--sample-every', type=int, default=1000,
                        help="Frequency of sampling reference states.")
    parser.add_argument('--kt', type=float, default=300*utils.kb,
                        help="Temperature")
    parser.add_argument('--dt', type=float, default=0.2,
                        help="Time step")
    parser.add_argument('--gamma', type=float, default=0.001,
                        help="Friction coefficient for Langevin integrator")

    parser.add_argument('--max-dist', type=float, default=150.0,
                        help="Distance at which we start applying a harmonic constraint")
    parser.add_argument('--spring-k', type=float, default=10.0,
                        help="Spring constant for applying maximum distance")

    parser.add_argument('--gumbel-start', type=float, default=1.0,
                        help="Starting temperature for gumbel softmax")
    parser.add_argument('--gumbel-end', type=float, default=0.01,
                        help="End temperature for gumbel softmax")

    parser.add_argument('--n-devices', type=int, required=True)


    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
