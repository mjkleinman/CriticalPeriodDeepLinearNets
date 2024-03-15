import argparse

import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from analysis_diag import MPNAnalysis
from analysis_diag import Y_default
from multichannel_net_diag import MultipathwayNet
from utils import set_seed, get_U_S_V

print_debug = True


def get_args():
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser()
    # parser.add_argument('--D', type=int, default=2, help='Depth of network')

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    return args


# For small init, we have p ** 2, instead of p ** 2 + 1 (for large init),
# since q ** 2 - p ** 2 = 0 is the constant of motion per pathway
def K_full(p, D):
    return p * np.power(p ** 2, (D - 1) / 2)


def get_init_k(n, s=0.1):
    return np.random.normal(0, s, size=(n, 2))


def k_trajectory(D, S, px0, py0, def_start, def_end, dt, T):
    # px0 is the p for first pathway (from this we can get qx)
    # py0 is the p for second pathway (from this we can get qy)

    px = np.zeros(T)
    py = np.zeros(T)

    px[0] = px0
    py[0] = py0

    Kx = np.zeros(T)
    Ky = np.zeros(T)

    Kx[0] = K_full(px0, D)
    Ky[0] = K_full(py0, D)

    for t in range(1, T):

        Omega = S - Kx[t - 1] - Ky[t - 1]
        px[t] = px[t - 1] + dt * (np.sqrt(px[t - 1] ** 2) ** (D - 1)) * Omega

        # Added by MK
        if t >= def_start and t < def_end:
            py[t] = py[t - 1]
        else:
            py[t] = py[t - 1] + dt * (np.sqrt(py[t - 1] ** 2) ** (D - 1)) * Omega

        Kx[t] = K_full(px[t], D)
        Ky[t] = K_full(py[t], D)

    return Kx, Ky


def k_trajectory_sing_def(D, S, px0, py0, def_start, def_end, dt, T):
    # px0 is the p for first pathway (from this we can get qx)
    # py0 is the p for second pathway (from this we can get qy)

    px = np.zeros(T)
    py = np.zeros(T)

    px[0] = px0
    py[0] = py0

    Kx = np.zeros(T)
    Ky = np.zeros(T)

    Kx[0] = K_full(px0, D)
    Ky[0] = K_full(py0, D)

    for t in range(1, T):

        Omega = S - Kx[t - 1] - Ky[t - 1]
        py[t] = py[t - 1]
        if t >= def_start and t < def_end:
            px[t] = px[t - 1]
        else:
            px[t] = px[t - 1] + dt * (np.sqrt(px[t - 1] ** 2) ** (D - 1)) * Omega

        # # Added by MK
        # else:
        #     py[t] = py[t - 1] + dt * (np.sqrt(py[t - 1] ** 2) ** (D - 1)) * Omega

        Kx[t] = K_full(px[t], D)
        Ky[t] = K_full(py[t], D)

    return Kx, Ky


def main(args):
    set_seed(args.seed)
    fig = plt.figure(figsize=(7, 2.7))
    gs = gridspec.GridSpec(1, 3, width_ratios=[4, 1.3, 1.3], wspace=0.2, figure=fig)
    # Create axes using the GridSpec
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    depth = args.depth  # only works for depth 2
    dt = args.dt  # learning rate
    T = args.T  # 5000
    eps = args.eps  # 0.0001
    add_noise_diag_scale = args.add_noise_diag_scale
    sing_idx_def = args.sing_def_idx
    nonlin = args.nonlin
    use_deficit = args.use_deficit

    diag_dim = 100  # args.pathway_width
    additive_noise_diag = torch.randn(diag_dim, ) * add_noise_diag_scale
    diag_values = (additive_noise_diag + torch.ones(diag_dim, )) * eps
    diag_values, _ = torch.sort(diag_values, descending=True)
    # rank = 8
    # additive_noise = torch.randn(rank, ) * add_noise_diag_scale
    # diag_values = (torch.ones(rank, ) + additive_noise) * eps
    # # diag_values = torch.abs(torch.rand(8) * std_dev)
    # diag_values, _ = torch.sort(diag_values, descending=True)

    # deficit_start
    for jj, deficit_start in enumerate(args.deficit_start_epoch):
        deficit_end = deficit_start + args.deficit_length

        U, SS, V = get_U_S_V(Y_default)

        mcn = MultipathwayNet(8, 15, depth=depth, num_pathways=2, width=args.pathway_width, bias=False,
                              nonlinearity=nonlin,
                              diagonal_init=True, U=U, V=V, eps=eps, kDiagInit=diag_values)

        mpna = MPNAnalysis(mcn)
        mpna.train_mcn(timesteps=T, lr=dt, deficit_start=deficit_start, deficit_end=deficit_end,
                       use_deficit=use_deficit, deficit_period=['gate_path', 'gate_sing'],
                       no_deficit_period=['gate_sing'], sing_idx=sing_idx_def)

        for sing_value_idx in range(5):
            S = SS[sing_value_idx] * SS.shape[0]
            for i in range(sing_value_idx, sing_value_idx + 1):  # first singular value
                z1 = np.array([K[i, i].to("cpu") for K in mpna.K_history[0]])
                z2 = np.array([K[i, i].to("cpu") for K in mpna.K_history[1]])

            eps_init_1, eps_init_2 = np.power(diag_values[sing_value_idx], 1 / depth), np.power(
                diag_values[sing_value_idx], 1 / depth)
            for di, (def_start, def_end) in enumerate([(deficit_start, deficit_end)]):
                if sing_value_idx == sing_idx_def:
                    k_traj = k_trajectory_sing_def(depth, S, eps_init_1, eps_init_2, def_start, def_end, dt, T)
                else:
                    k_traj = k_trajectory(depth, S, eps_init_1, eps_init_2, def_start, def_end, dt, T)  # no deficit

            ax.set_title('Gating and feature lesioning deficit')
            ax.axvspan(deficit_start, deficit_end, alpha=0.15, color='gray')

            ax.plot(k_traj[0], color=cm.tab10(sing_value_idx) if sing_value_idx == sing_idx_def else 'k',
                    linestyle='dashed', linewidth='1', alpha=0.7,
                    label='ODE' if sing_value_idx == 0 else None)  # use color corresponding to sing deficit
            ax.plot(k_traj[1], color=cm.tab10(sing_value_idx) if sing_value_idx == sing_idx_def else 'k', linewidth='1',
                    alpha=0.7, linestyle='dashed')
            ax.set_xlabel('Epoch')
            if jj == 0:
                ax.set_ylabel(r'Path Singular Value ($K_\alpha$)')
            indices = np.arange(len(z1))
            nskip = int(35)  # 20
            # if sing_value_idx == 0:
            #     ax.scatter(indices[::nskip], z1[::nskip], s=7, color=cm.tab10(sing_value_idx),
            #                label=r'SGD $K_{a\alpha}$')
            #     ax.scatter(indices[::nskip], z2[::nskip], s=7, color=cm.tab10(sing_value_idx), marker='^',
            #                label=r'SGD $K_{b\alpha}$')
            # else:
            #     ax.scatter(indices[::nskip], z1[::nskip], s=7, color=cm.tab10(sing_value_idx))
            #     ax.scatter(indices[::nskip], z2[::nskip], s=15 if sing_value_idx == sing_idx_def else 7,
            #                color=cm.tab10(sing_value_idx), marker='^')
            if sing_value_idx == 0:
                ax.scatter(indices[::nskip], z1[::nskip], s=13, color=cm.tab10(sing_value_idx),
                           label=r'SGD $K_{a\alpha}$', alpha=0.8, marker='x')
                ax.scatter(indices[::nskip], z2[::nskip], s=20, color=cm.tab10(sing_value_idx), marker='+',
                           label=r'SGD $K_{b\alpha}$', alpha=0.8)
            else:
                ax.scatter(indices[::nskip], z1[::nskip], s=13, color=cm.tab10(sing_value_idx), alpha=0.8,
                           marker='x')
                ax.scatter(indices[::nskip], z2[::nskip], s=30 if sing_value_idx == sing_idx_def else 20,
                           color=cm.tab10(sing_value_idx), marker='+',
                           alpha=0.8)
            ax.axvline(deficit_start, color='gray', linestyle='dashed', linewidth=1, alpha=0.5)
            ax.axvline(deficit_end, color='gray', linestyle='dashed', linewidth=1, alpha=0.5)
            leg = ax.legend(loc='upper right', frameon=False, fontsize=8, bbox_to_anchor=(1, 0.98))
            leg.legendHandles[0].set_color('black')
            leg.legendHandles[1].set_color('black')
            leg.legendHandles[2].set_color('black')

        if print_debug:
            print(z1[:100])
            print(z2[:100])
            print(k_traj[0][:100])

        mpna.plot_K([ax2, ax3], labels=['a', 'b'])

    fig.tight_layout()
    fig.savefig(
        f"plots/SING_analytical-and-sgd-depth.pdf",
        # f"development/plots-rebuttal-iclr/SING_analytical-and-sgd-depth.pdf",
        # f"development/SING_analytical-and-sgd-depth.pdf",
        bbox_inches='tight', dpi=300)
    # plt.close('all')


if __name__ == '__main__':
    opts = get_args()
    assert opts.output_dir is not None
    main(opts)
