# Code was modified starting from: https://github.com/roosephu/deep_matrix_factorization

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from model_utils import set_seed, create_matrix, init_model, get_e2e, get_train_loss, get_test_loss
# from opt import GroupRMSprop

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', default=None, type=int, help='depth of linear network')
    parser.add_argument('--rank1', default=None, type=int, help='rank of deficit task')
    parser.add_argument('--ntrain_samples', default=None, type=int, help='number of observed matrix values')
    parser.add_argument('--seed', default=None, type=int, help='random seed')
    parser.add_argument('--init_scale', default=None, type=float, help='scale for initialization')

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    return args


def get_observations(n, n_train_samples, w_gt, us=None, vs=None):
    if us is None and vs is None:
        indices = torch.multinomial(torch.ones(n * n), n_train_samples, replacement=False)
        us, vs = indices // n, indices % n
    ys_ = w_gt[us, vs]
    assert 0.8 <= ys_.pow(2).mean().sqrt() <= 1.2
    return (us, vs), ys_


def main(args):
    set_seed(args.seed)
    w_gt1 = create_matrix(args.matrix_dim, args.rank1, False)
    w_gt2 = create_matrix(args.matrix_dim, args.rank2, False)
    (us, vs), ys_ = get_observations(args.matrix_dim, args.ntrain_samples, w_gt1)
    (_, _), ys_2_ = get_observations(args.matrix_dim, args.ntrain_samples, w_gt2, us, vs)
    hidden_sizes = [[args.hidden_dim] + [args.hidden_dim]] * args.depth
    model = nn.Sequential(*[nn.Linear(f_in, f_out, bias=False) for (f_in, f_out) in hidden_sizes]).to(device)

    loss = None
    losses_critical = []
    T_critical_list = args.T_critical_list
    for T_critical in T_critical_list:
        # Initialize the model
        init_model(model, args.depth, initialization=args.initialization, init_scale=args.init_scale, seed=args.seed)
        if args.optimizer == 'grouprmsprop':
            optimizer = GroupRMSprop(model.parameters(), lr=args.sgd_lr)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.sgd_lr)
        else:
            raise NotImplementedError

        n_iters = T_critical + args.epoch_continue
        singular_values_list = []
        epoch_singular_values_list = []
        for T in range(n_iters):
            e2e = get_e2e(model)
            loss1 = get_train_loss(e2e, ys_, us, vs)
            loss2 = get_train_loss(e2e, ys_2_, us, vs)
            loss = loss2 if T > T_critical else loss1
            params_norm = 0
            for param in model.parameters():
                params_norm = params_norm + param.pow(2).sum()
            optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                test_loss = get_test_loss(e2e, w_gt2) if T > T_critical else get_test_loss(e2e, w_gt1)

            optimizer.step()
            if T % 10 == 0:
                print('Iter: {}, Train Loss: {}, Test Loss: {}, Param Norm: {}'.format(T, loss, test_loss, params_norm))
                U, singular_values, V = e2e.svd()
                singular_values_list.append(singular_values[:10].detach().cpu().numpy())
                epoch_singular_values_list.append(T)
                print('Singular Values: {}'.format(singular_values[:10].detach().cpu().numpy()))
        losses_critical.append(test_loss.item())

        plt.figure(figsize=(3.2, 2.7))
        for i in range(10):
            plt.plot(epoch_singular_values_list, np.array(singular_values_list)[:, i], color=plt.cm.summer(i / 10))
        plt.axvline(x=T_critical, linestyle='--', color='gray')
        plt.xlabel('Epoch')
        plt.ylabel('Singular Vals')
        singular_dir = os.path.join(args.output_dir, 'singular-vals')
        Path(singular_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(singular_dir, f'T_critical={T_critical}.pdf'), bbox_inches='tight')
        plt.close()

    plt.figure(figsize=(3.2, 2.7))
    plt.plot(T_critical_list, losses_critical, marker='o', markersize=3)
    plt.xlabel('Deficit Switch Epoch')
    plt.ylabel('Test Loss')
    plt.ylim(0, max(losses_critical) + 0.1)
    plt.savefig(os.path.join(args.output_dir, f'critical-period-completion.pdf'), bbox_inches='tight')

    # Saving arguments:
    to_save = {'args': args, 'losses_critical': losses_critical}
    torch.save(to_save, os.path.join(args.output_dir, 'info.pth'))


if __name__ == '__main__':
    opts = get_args()
    assert opts.output_dir is not None
    opts.output_dir = os.path.join(opts.output_dir,
                                   f'depth={opts.depth}-ntrain={opts.ntrain_samples}_rank1={opts.rank1}'
                                   f'_rank2={opts.rank2}_epochCont={opts.epoch_continue}'
                                   f'_seed={opts.seed}_initScale={opts.init_scale}_lr={opts.sgd_lr}_opt={opts.optimizer}')
    Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    # # Dump Params
    dump = os.path.join(opts.output_dir, 'params.yaml')
    with open(dump, 'w') as f:
        yaml.dump(opts, f)
    main(opts)
