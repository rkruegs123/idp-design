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
from jax import vmap, jit, value_and_grad, lax, random
from jax_md import space, simulate
from flax import linen as nn
import optax

from idp_design.energy_prob import get_energy_fn
import idp_design.utils as utils
import idp_design.checkpoint as checkpoint
import idp_design.observable as observable






checkpoint_every = 50
scan = checkpoint.get_scan(checkpoint_every)


class MLP(nn.Module):
    features: int
    layers: int
    nres: int

    @nn.compact
    def __call__(self, x, training: bool):
        for _ in range(self.layers):
            x = nn.Dense(self.features)(x)
            # x = nn.WeightNorm(Dense(self.features))(x)
            x = nn.leaky_relu(x)
            # x = nn.Dropout(0.5, deterministic=not training)(x)
        x = nn.Dense(self.nres*20, use_bias=True)(x)
        x = x.reshape((self.nres, 20))
        return x


def run(args):
    key = random.PRNGKey(args['key'])

    n_sims = args['n_sims']
    n_eq_steps = args['n_eq_steps']
    n_sample_steps = args['n_sample_steps']
    sample_every = args['sample_every']
    assert(n_sample_steps % sample_every == 0)
    num_points_per_batch = n_sample_steps // sample_every
    n_ref_states = num_points_per_batch * n_sims

    histidine_not_charged = args['histidine_not_charged']
    if histidine_not_charged:
        pos_charge_residues = set(["K", "R"])
        pos_charge_mapper = jnp.array([int(res in pos_charge_residues) for res in utils.RES_ALPHA])
    else:
        pos_charge_mapper = utils.pos_charge_mapper

    run_name = args['run_name']
    kT = args['kt']
    beta = 1 / kT
    dt = args['dt']
    gamma = args['gamma']
    out_box_size = args['out_box_size']
    seq_length = args['seq_length']
    target_rg = args['target_rg']

    ree_instead = args['ree_instead']
    target_ree = args['target_ree']

    n_iters = args['n_iters']
    lr = args['lr']
    min_neff_factor = args['min_neff_factor']
    min_n_eff = int(n_ref_states * min_neff_factor)
    max_approx_iters = args['max_approx_iters']
    optimizer_type = args['optimizer_type']


    relu_steep_slope = args['relu_steep_slope']
    relu_slope_scale = args['relu_slope_scale']
    relu_flattened_slope = relu_steep_slope / relu_slope_scale


    gumbel_end = args['gumbel_end']
    gumbel_start = args['gumbel_start']
    gumbel_temps = onp.linspace(gumbel_start, gumbel_end, n_iters)
    # gumbel_temps = onp.linspace(0.1, 3.0, 100)**3


    def normalize(logits, temp):
        pseq = jax.nn.softmax(logits / temp)

        return pseq

    init_method = "scaled"

    nn_layers = args['nn_layers']
    nn_features = args['nn_features']
    input_seed_size = 10
    key, init_key, inference_key = random.split(key, 3)
    example_input_seed = random.normal(init_key, (input_seed_size,))
    training_seed_key = random.normal(inference_key, (input_seed_size,))
    model = MLP(layers=nn_layers, features=nn_features, nres=seq_length)

    min_pos_charge_ratios = [args['min_pos_charge_ratio'] for _ in range(n_iters)]
    min_neg_charge_ratios = [args['min_neg_charge_ratio'] for _ in range(n_iters)]
    min_pos_charge_ratios = jnp.array(min_pos_charge_ratios)
    min_neg_charge_ratios = jnp.array(min_neg_charge_ratios)

    # min_pos_charge_ratio = args['min_pos_charge_ratio']

    def pos_charge_loss_multiplier_helper(pos_charge_ratio, min_pos_charge_ratio):
        pos_charge_ratio_offset = 1 - relu_steep_slope*min_pos_charge_ratio
        pos_charge_ratio_offset_small = 1 - relu_flattened_slope*min_pos_charge_ratio
        return jnp.where(
            pos_charge_ratio < min_pos_charge_ratio,
            relu_steep_slope*pos_charge_ratio + pos_charge_ratio_offset,
            relu_flattened_slope*pos_charge_ratio + pos_charge_ratio_offset_small
        )


    def pos_charge_loss_multiplier(pseq, min_pos_charge_ratio):
        all_pos_charged_ratios = vmap(jnp.dot, (None, 0))(pos_charge_mapper, pseq)
        expected_pos_charged_ratio = jnp.mean(all_pos_charged_ratios)

        return pos_charge_loss_multiplier_helper(expected_pos_charged_ratio, min_pos_charge_ratio), expected_pos_charged_ratio

    # min_neg_charge_ratio = args['min_neg_charge_ratio']


    def neg_charge_loss_multiplier_helper(neg_charge_ratio, min_neg_charge_ratio):
        neg_charge_ratio_offset = 1 - relu_steep_slope*min_neg_charge_ratio
        neg_charge_ratio_offset_small = 1 - relu_flattened_slope*min_neg_charge_ratio
        return jnp.where(
            neg_charge_ratio < min_neg_charge_ratio,
            relu_steep_slope*neg_charge_ratio + neg_charge_ratio_offset,
            relu_flattened_slope*neg_charge_ratio + neg_charge_ratio_offset_small)


    def neg_charge_loss_multiplier(pseq, min_neg_charge_ratio):
        all_neg_charged_ratios = vmap(jnp.dot, (None, 0))(utils.neg_charge_mapper, pseq)
        expected_neg_charged_ratio = jnp.mean(all_neg_charged_ratios)

        return neg_charge_loss_multiplier_helper(expected_neg_charged_ratio, min_neg_charge_ratio), expected_neg_charged_ratio




    output_basedir = Path(args['output_basedir'])
    run_dir = output_basedir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)

    ref_traj_dir = run_dir / "ref_traj"
    ref_traj_dir.mkdir(parents=False, exist_ok=False)

    pseq_dir = run_dir / "pseq"
    pseq_dir.mkdir(parents=False, exist_ok=False)


    img_dir = run_dir / "img"
    img_dir.mkdir(parents=False, exist_ok=False)

    log_dir = run_dir / "log"
    log_dir.mkdir(parents=False, exist_ok=False)

    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)

    test_pos_charge_values = onp.linspace(0.0, 1.0)
    test_vals = [pos_charge_loss_multiplier_helper(val, min_pos_charge_ratios[-1]) for val in test_pos_charge_values]
    plt.plot(test_pos_charge_values, test_vals)
    plt.axhline(y=1.0, linestyle='--', color="red", label="y=1.0")
    plt.axvline(x=min_pos_charge_ratios[-1], linestyle='--', color="green", label="min + charge ratio")
    plt.savefig(img_dir / "pos_charge_ratio_loss.png")
    plt.clf()

    test_neg_charge_values = onp.linspace(0.0, 1.0)
    test_vals = [neg_charge_loss_multiplier_helper(val, min_neg_charge_ratios[-1]) for val in test_neg_charge_values]
    plt.plot(test_neg_charge_values, test_vals)
    plt.axhline(y=1.0, linestyle='--', color="red", label="y=1.0")
    plt.axvline(x=min_neg_charge_ratios[-1], linestyle='--', color="green", label="min + charge ratio")
    plt.savefig(img_dir / "neg_charge_ratio_loss.png")
    plt.clf()


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



    def get_ref_states(params, i, R, sample_key, temp):

        iter_dir = ref_traj_dir / f"iter{i}"
        iter_dir.mkdir(parents=False, exist_ok=False)

        # Get the pseq and corresponding mass
        curr_logits = model.apply(params, training_seed_key, training=True)
        # sample_key, norm_key = random.split(sample_key)
        # curr_pseq = normalize(curr_logits, temp, sample_key)
        curr_pseq = normalize(curr_logits, temp)
        curr_mass = utils.get_pseq_mass(curr_pseq)

        # Run a batch of simulations and save the combined trajectory
        start = time.time()
        sample_traj = batch_sim(sample_key, R, curr_pseq, curr_mass)
        end = time.time()
        sample_time = end - start


        # Analyze the resulting trajectory

        ## Rg
        sample_rgs = vmap(observable.rg, (0, None, None))(sample_traj, curr_mass, displacement_fn)
        sample_energies = mapped_energy_fn(sample_traj, curr_pseq)

        sns.histplot(sample_rgs)
        plt.savefig(iter_dir / "rg_hist.png")
        plt.clf()

        num_rgs = sample_rgs.shape[0]
        running_avg_rgs = onp.cumsum(sample_rgs) / onp.arange(1, num_rgs+1)
        plt.plot(running_avg_rgs)
        plt.savefig(iter_dir / "rg_running_avg.png")
        plt.clf()

        ## Ree
        sample_rees = vmap(observable.end_to_end_dist, (0, None))(sample_traj, displacement_fn)

        sns.histplot(sample_rees)
        plt.savefig(iter_dir / "ree_hist.png")
        plt.clf()

        num_rees = sample_rees.shape[0]
        running_avg_rees = onp.cumsum(sample_rees) / onp.arange(1, num_rees+1)
        plt.plot(running_avg_rees)
        plt.savefig(iter_dir / "ree_running_avg.png")
        plt.clf()


        summary_str = ""
        summary_str += f"Mean Rg: {onp.mean(sample_rgs)}\n"
        summary_str += f"Mean Ree: {onp.mean(sample_rees)}\n"
        summary_str += f"Simulation time: {sample_time}\n"
        with open(iter_dir / "summary.txt", "w+") as f:
            f.write(summary_str)

        return sample_traj, sample_energies, jnp.array(sample_rgs), jnp.array(sample_rees)


    def loss_fn(params, ref_states, ref_energies, ref_rgs, ref_rees, temp, loss_key, i):
        logits = model.apply(params, training_seed_key, training=True)
        # loss_key, norm_key = random.split(loss_key)
        # pseq = normalize(logits, temp, norm_key)
        pseq = normalize(logits, temp)

        energy_scan_fn = lambda state, ts: (None, energy_fn(ts, pseq=pseq))
        _, new_energies = scan(energy_scan_fn, None, ref_states)

        weights, n_eff = utils.compute_weights(ref_energies, new_energies, beta)

        weighted_rgs = weights * ref_rgs # element-wise multiplication
        expected_rg = jnp.sum(weighted_rgs)

        weighted_rees = weights * ref_rees # element-wise multiplication
        expected_ree = jnp.sum(weighted_rees)

        if ree_instead:
            mse = (expected_ree - target_ree)**2
        else:
            mse = (expected_rg - target_rg)**2
        rmse = jnp.sqrt(mse)

        pos_charge_scalar, pos_charge_ratio = pos_charge_loss_multiplier(pseq, min_pos_charge_ratios[i])
        neg_charge_scalar, neg_charge_ratio = neg_charge_loss_multiplier(pseq, min_neg_charge_ratios[i])

        loss = rmse * pos_charge_scalar * neg_charge_scalar

        return loss, (n_eff, expected_rg, expected_ree, pseq, rmse,
                      pos_charge_scalar, pos_charge_ratio,
                      neg_charge_scalar, neg_charge_ratio)
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    grad_fn = jit(grad_fn)

    if init_method == "scaled":
        init_pseq, init_logits = utils.get_charge_constrained_pseq(
            seq_length, min_pos_charge_ratios[-1], min_neg_charge_ratios[-1],
            constrained_pos_charge_residues=pos_charge_residues
        )
    else:
        raise RuntimeError(f"Invalid init method: {init_method}")

    # Setup the optimization
    key, params_key = random.split(key)
    params = model.init(params_key, example_input_seed, training=False)
    # key, pretrain_norm_key = random.split(key)

    @jit
    def pretrain_loss_fn(model_params):
        logits = model.apply(model_params, training_seed_key, training=True) # FIXME: hardcoded, assumes we only have one input seed to the generative model

        # pseq = jax.nn.softmax(logits)
        pseq = normalize(logits, gumbel_temps[0])

        # return jnp.mean((logits - init_logits)**2)
        return jnp.mean((pseq - init_pseq)**2) * 100

    pretrain_grad_fn = value_and_grad(pretrain_loss_fn)
    pretrain_grad_fn = jit(pretrain_grad_fn)
    pretrain_lr = 0.0001
    pretrain_optimizer = optax.adam(learning_rate=pretrain_lr)
    opt_state = pretrain_optimizer.init(params)
    num_pretrain_iter = 250
    pretrain_loss_path = log_dir / "pretrain_loss.txt"
    pretrain_nn_params_path = log_dir / "pretrain_nn_params.txt"
    for i in range(num_pretrain_iter):
        loss, grads = pretrain_grad_fn(params)
        print(f"- Pretrain Iteration: {i}")
        print(f"- Loss: {loss}")
        with open(pretrain_loss_path, "a") as f:
            f.write(f"{loss}\n")
        with open(pretrain_nn_params_path, "a") as f:
            f.write(f"{params}\n")
        updates, opt_state = pretrain_optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)


    R_init = list()
    for i in range(seq_length):
        R_init.append([out_box_size/2, out_box_size/2, out_box_size/2+utils.spring_r0*i])
    R_init = jnp.array(R_init)

    all_ref_iters = list()
    all_ref_rgs = list()
    all_ref_rees = list()

    print(f"Generating initial reference states and energies...")
    key, ref_states_key = random.split(key)
    ref_states, ref_energies, ref_rgs, ref_rees = get_ref_states(params, 0, R_init, ref_states_key, temp=gumbel_temps[0])
    all_ref_iters.append(0)
    all_ref_rgs.append(onp.mean(ref_rgs))
    all_ref_rees.append(onp.mean(ref_rees))

    loss_path = log_dir / "loss.txt"
    rmse_path = log_dir / "rmse.txt"
    pos_charge_scalar_path = log_dir / "pos_charge_scalar.txt"
    pos_charge_ratio_path = log_dir / "pos_charge_ratio.txt"
    neg_charge_scalar_path = log_dir / "neg_charge_scalar.txt"
    neg_charge_ratio_path = log_dir / "neg_charge_ratio.txt"
    min_pos_charge_ratio_path = log_dir / "min_pos_charge_ratio.txt"
    min_neg_charge_ratio_path = log_dir / "min_neg_charge_ratio.txt"
    rg_path = log_dir / "rg.txt"
    ree_path = log_dir / "ree.txt"
    grads_path = log_dir / "grads.txt"
    argmax_seq_path = log_dir / "argmax_seq.txt"
    argmax_seq_scaled_path = log_dir / "argmax_seq_scaled.txt"
    plot_every = 10
    all_rgs = list()
    all_rees = list()
    all_pos_charge_ratios = list()
    all_neg_charge_ratios = list()
    num_resample_iters = 0

    if optimizer_type == "lamb":
        optimizer = optax.lamb(learning_rate=lr)
    elif optimizer_type == "adam":
        optimizer = optax.adam(learning_rate=lr)
    else:
        raise RuntimeError(f"Invalid optimizer type: {optimizer_type}")
    opt_state = optimizer.init(params)

    for i in tqdm(range(n_iters), desc="Iteration"):
        key, loss_key = random.split(key)
        (loss, aux), grads = grad_fn(params, ref_states, ref_energies, ref_rgs, ref_rees, gumbel_temps[i], loss_key, i)
        n_eff = aux[0]
        num_resample_iters += 1

        if n_eff < min_n_eff or num_resample_iters > max_approx_iters:
            print(f"Resampling reference states...")
            num_resample_iters = 0

            key, ref_states_key = random.split(key)
            ref_states, ref_energies, ref_rgs, ref_rees = get_ref_states(
                params, i, utils.recenter(ref_states[-1], out_box_size), ref_states_key, gumbel_temps[i])
            all_ref_iters.append(i)
            all_ref_rgs.append(onp.mean(ref_rgs))
            all_ref_rees.append(onp.mean(ref_rees))

            (loss, aux), grads = grad_fn(params, ref_states, ref_energies, ref_rgs, ref_rees, gumbel_temps[i], loss_key, i)

        n_eff = aux[0]
        expected_rg, expected_ree, pseq, rmse = aux[1], aux[2], aux[3], aux[4]
        pos_charge_scalar, pos_charge_ratio = aux[5], aux[6]
        neg_charge_scalar, neg_charge_ratio = aux[7], aux[8]

        with open(loss_path, "a") as f:
            f.write(f"{loss}\n")
        with open(min_pos_charge_ratio_path, "a") as f:
            f.write(f"{min_pos_charge_ratios[i]}\n")
        with open(min_neg_charge_ratio_path, "a") as f:
            f.write(f"{min_neg_charge_ratios[i]}\n")
        with open(rmse_path, "a") as f:
            f.write(f"{rmse}\n")
        with open(pos_charge_scalar_path, "a") as f:
            f.write(f"{pos_charge_scalar}\n")
        with open(pos_charge_ratio_path, "a") as f:
            f.write(f"{pos_charge_ratio}\n")
        with open(neg_charge_scalar_path, "a") as f:
            f.write(f"{neg_charge_scalar}\n")
        with open(neg_charge_ratio_path, "a") as f:
            f.write(f"{neg_charge_ratio}\n")
        # with open(grads_path, "a") as f:
        #     f.write(f"{pprint.pformat(grads)}\n")
        with open(rg_path, "a") as f:
            f.write(f"{expected_rg}\n")
        with open(ree_path, "a") as f:
            f.write(f"{expected_ree}\n")

        all_rgs.append(expected_rg)
        all_rees.append(expected_ree)
        all_pos_charge_ratios.append(pos_charge_ratio)
        all_neg_charge_ratios.append(neg_charge_ratio)


        argmax_seq = utils.get_argmax_seq(pseq, scale=False)
        with open(argmax_seq_path, "a") as f:
            f.write(f"{argmax_seq}\n")

        argmax_seq_scaled = utils.get_argmax_seq(pseq, scale=True)
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

            plt.plot(all_rgs)
            plt.xlabel("Iteration")
            plt.ylabel("Rg")
            plt.axhline(y=target_rg, linestyle='--', label="Target Rg", color='red')
            plt.legend()
            plt.savefig(img_dir / f"rg_i{i}.png")
            plt.clf()

            plt.plot(all_rees)
            plt.xlabel("Iteration")
            plt.ylabel("Ree")
            plt.axhline(y=target_ree, linestyle='--', label="Target Ree", color='red')
            plt.legend()
            plt.savefig(img_dir / f"ree_i{i}.png")
            plt.clf()

            plt.plot(all_pos_charge_ratios)
            plt.xlabel("Iteration")
            plt.ylabel("+ Charge Ratio")
            plt.axhline(y=min_pos_charge_ratios[-1], linestyle='--', label="Min +-Charge", color='red')
            plt.legend()
            plt.savefig(img_dir / f"pos_charge_ratio_i{i}.png")
            plt.clf()

            plt.plot(all_neg_charge_ratios)
            plt.xlabel("Iteration")
            plt.ylabel("neg-Charge Ratio")
            plt.axhline(y=min_neg_charge_ratios[-1], linestyle='--', label="Min neg-Charge", color='red')
            plt.legend()
            plt.savefig(img_dir / f"neg_charge_ratio_i{i}.png")
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
    onp.save(obj_dir / "ref_rees.npy", onp.array(all_ref_rees), allow_pickle=False)



def get_parser():

    parser = argparse.ArgumentParser(description="DiffTRE for IDP design")

    parser.add_argument('--output-basedir', type=str, help='Output base directory',
                        default="output/"
    )
    parser.add_argument('--run-name', type=str, help='Run name')

    parser.add_argument('--n-iters', type=int, default=200,
                        help="Number of iterations of gradient descent")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument('--min-neff-factor', type=float, default=0.95,
                        help="Factor for determining min Neff")
    parser.add_argument('--max-approx-iters', type=int, default=5,
                        help="Maximum number of iterations before resampling")
    parser.add_argument('--seq-length', type=int, default=100,
                        help="Sequence length")
    parser.add_argument('--target-rg', type=float, default=0.0,
                        help="Target radius of gyration in Angstroms")

    parser.add_argument('--gumbel-start', type=float, default=1.0,
                        help="Starting temperature for gumbel softmax")
    parser.add_argument('--gumbel-end', type=float, default=0.01,
                        help="End temperature for gumbel softmax")

    # Simulation arguments
    parser.add_argument('--key', type=int, default=0)
    parser.add_argument('--out-box-size', type=float, default=400.0,
                        help="Length of the box for injavis visualization")
    parser.add_argument('--n-sims', type=int, default=15,
                        help="Number of independent simulations")
    parser.add_argument('--n-eq-steps', type=int, default=10000,
                        help="Number of equilibration steps")
    parser.add_argument('--n-sample-steps', type=int, default=500000,
                        help="Number of steps from which to sample states")
    parser.add_argument('--sample-every', type=int, default=1000,
                        help="Frequency of sampling reference states.")
    parser.add_argument('--kt', type=float, default=300*utils.kb,
                        help="Temperature")
    parser.add_argument('--dt', type=float, default=0.2,
                        help="Time step")
    parser.add_argument('--gamma', type=float, default=0.001,
                        help="Friction coefficient for Langevin integrator")

    parser.add_argument('--relu-steep-slope', type=float, default=-1000.0,
                        help="Slope for steep part of ReLU")
    parser.add_argument('--relu-slope-scale', type=float, default=100000.0,
                        help="Factor for decreasing the slope after the steep part of the relu")

    parser.add_argument('--min-pos-charge-ratio', type=float, default=0.2,
                        help="Minimum ratio that is positively charged")
    parser.add_argument('--min-neg-charge-ratio', type=float, default=0.2,
                        help="Minimum ratio that is negatively charged")

    parser.add_argument('--nn-features', type=int, default=4000,
                        help="Number of features for the MLP")
    parser.add_argument('--nn-layers', type=int, default=6,
                        help="Number of layers for the MLP")

    parser.add_argument('--optimizer-type', type=str, default="lamb",
                        choices=["lamb", "adam"],
                        help='Optimizer type for NN logits')

    parser.add_argument('--ree-instead', action='store_true',
                        help="If true, will target Ree instead of Rg")
    parser.add_argument('--target-ree', type=float, default=20.0,
                        help="Target end to end distance in Angstroms")

    parser.add_argument('--histidine-not-charged', action='store_true',
                        help="If true, will not count histidine as a positively charged residue")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
