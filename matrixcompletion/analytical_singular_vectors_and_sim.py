import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# from main_transfer_analytical_sim import init_model
from model_utils import set_seed, get_e2e


def init_model(model, depth, initialization='orthogonal', init_scale=0.001, hidden_sizes=[100], seed=0, u=None,
               vh=None):
    set_seed(seed)
    if initialization == 'orthogonal':
        scale = (init_scale * np.sqrt(hidden_sizes[0])) ** (1. / depth)
        # scale = init_scale * np.sqrt(hidden_sizes[0] * 2)
        matrices = []
        for param in model.parameters():
            nn.init.orthogonal_(param)
            param.data.mul_(scale)
            matrices.append(param.data.cpu().numpy())
        for a, b in zip(matrices, matrices[1:]):
            assert np.allclose(a.dot(a.T), b.T.dot(b), atol=1e-6)
        # w_init = get_e2e(model).detach().cpu().numpy()
        # u_init, s_init, vt_init = np.linalg.svd(w_init)
        # print(u.shape, vh.shape)
        # print(model)
    elif initialization == 'gaussian':
        n = hidden_sizes[0]
        assert hidden_sizes[0] == hidden_sizes[-1]
        scale = init_scale ** (1. / depth) * n ** (-0.5)
        for param in model.parameters():
            nn.init.normal_(param, std=scale)
        e2e = get_e2e(model).detach().cpu().numpy()
        e2e_fro = np.linalg.norm(e2e, 'fro')
        desired_fro = init_scale * np.sqrt(n)
        assert 0.8 <= e2e_fro / desired_fro <= 1.2
    elif initialization == 'identity':
        scale = init_scale ** (1. / depth)
        eps = 1e-5
        eps_add = torch.randn(hidden_sizes[0]) * eps
        matrices = []
        for param in model.parameters():
            nn.init.eye_(param)
            print(param.shape)
            param.data.mul_(scale).add_(torch.diag(eps_add))
            matrices.append(param.data.cpu().numpy())
        for a, b in zip(matrices, matrices[1:]):
            assert np.allclose(a.dot(a.T), b.T.dot(b), atol=1e-3)
        # w_init = get_e2e(model).detach().cpu().numpy()
        # _, s_init, _ = np.linalg.svd(w_init)
        # print(s_init)
        # print(s_init)
        # print(u.shape, vh.shape)

    elif initialization == 'analytical':
        scale = init_scale
        model[0].weight.data = torch.from_numpy(vh) * scale
        model[1].weight.data = torch.from_numpy(u) * scale


def create_matrix(n, rank, symmetric=False):
    r = rank
    U = np.random.randn(n, r).astype(np.float32)
    if symmetric:
        V = U
    else:
        V = np.random.randn(n, r).astype(np.float32)
    w_gt = U.dot(V.T) / np.sqrt(r)
    w_gt = w_gt / np.linalg.norm(w_gt, 'fro') * n
    return w_gt  # w_gt


def get_observations(n, n_train_samples, w_gt, us=None, vs=None):
    if us is None and vs is None:
        indices = np.random.choice(n * n, n_train_samples, replace=False)
        us, vs = indices // n, indices % n
    ys_ = w_gt[us, vs]
    assert 0.8 <= np.sqrt(np.mean(np.power(ys_, 2))) <= 1.2
    return (us, vs), ys_


# No longer using for loop (see compute_F_matrix)
def compute_F_forloop(N, s, depth):
    # F, skew symmetric matrix with (sigma_{r'}^2 - sigma_r^2)^{-1} in the (r', r) entry
    # TODO: Avoid for loops and just do matrix multiplication
    F = np.zeros((N, N))
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            if i != j:
                F[i, j] = (s[j] ** (2 / depth) - s[i] ** (2 / depth)) ** (-1)
    assert np.allclose(F, -F.T, atol=1e-6)
    return F
    # print('F is skew symmetric')


def compute_F_matrix(N, s, depth):
    s_reshaped = s.reshape((N, 1))
    F = (s_reshaped.T ** (2 / depth)) - (s_reshaped ** (2 / depth))
    np.fill_diagonal(F, 1)  # to handle division by zero, will set back to zero
    F = 1 / F
    np.fill_diagonal(F, 0)  # setting diagonal back to 0
    assert np.allclose(F, -F.T, atol=1e-6)
    return F


def init_W(N, scale_w=0.001):
    # Initialize W
    W = np.random.randn(N, N) * scale_w

    # Alternative initialization in task basis
    # um, sm, vhm = np.linalg.svd(M)
    # print(f'Singular values are {sm}')
    # W = scale_w * um @ np.diag(sm) @ vhm
    return W


def get_train_loss(e2e, ys_, us, vs):
    ys = e2e[us, vs]
    return 0.5 * (ys - ys_).pow(2).mean()  # Added the factor of 0.5 to correspond with analytical solution


def get_test_loss(e2e, w_gt):
    return (w_gt - e2e).view(-1).pow(2).mean()



set_seed(0)
rank1 = 8  # Rank of 1st matrix
rank2 = 2  # Rank of 2nd matrix
N = 100  # Size of matrix (ncol and nrow)
num_obs = 5000  # 1750 2000
depth = 3  # depth of network (number of layers) overparametrization
M = create_matrix(N, rank1)
M2 = create_matrix(N, rank2)
(us, vs), ys_ = get_observations(N, num_obs, M)
(_, _), ys_2_ = get_observations(N, num_obs, M2, us, vs)

I = np.eye(N)
print(f'M.shape is {M.shape}')
step = 0.25
# num_steps = 30000
continue_steps = 10000  # 60000  # 10000


def get_u_s_v_dot(W, M):
    # Compute gradient of L(W) wrt W
    grad_l = np.zeros((N, N))
    grad_l[us, vs] = -(M - W)[us, vs] * (1 / len(us))  # normalize  for average gradient over batch
    # grad_l = - (M - W)

    # SVD of W
    u, s, vh = np.linalg.svd(W)
    v = vh.T

    # Compute F
    # F = compute_F_forloop(N, s, depth)
    F = compute_F_matrix(N, s, depth)

    # Evolution of u and v: Lemma 2 of Arora et al. 2019
    A = u.T @ grad_l @ v @ np.diag(s)
    B = np.diag(s) @ v.T @ grad_l.T @ u
    C = (I - u @ u.T) @ grad_l @ v @ np.diag(s ** (1 - 2 / depth))
    udot = -u @ (F * (A + B)) - C

    D = np.diag(s) @ u.T @ grad_l @ v + v.T @ grad_l.T @ u @ np.diag(s)
    E = (I - v @ v.T) @ grad_l.T @ u.T @ np.diag(s ** (1 - 2 / depth))
    vdot = -v @ (F * D) - E

    # Evolution of sigma: Thm 3 of Arora et al. 2019
    Diag_ULV = np.diag(u.T @ grad_l @ v)
    s_dot = -depth * s ** (2 - 2 / depth) * Diag_ULV

    return s_dot, udot, vdot, s, u, v


def run_analytical(w_init=None, num_steps=20000):
    # Initialize W
    set_seed(0)
    if w_init is None:
        W = init_W(N=N, scale_w=0.001)
    else:
        W = w_init
    s_plot = np.zeros((num_steps, N))
    s = np.zeros((N,))
    epoch_switch = num_steps - continue_steps
    for t in range(num_steps):
        if t % 100 == 0:
            print(f'Epoch: {t}')
        if t < epoch_switch:
            ds, du, dv, s, u, v = get_u_s_v_dot(W, M)
        else:
            ds, du, dv, s, u, v = get_u_s_v_dot(W, M2)

        s_plot[t, :] = s
        W = u @ np.diag(s) @ v.T + step * (du @ np.diag(s) @ v.T + u @ np.diag(ds) @ v.T + u @ np.diag(s) @ dv.T)

    return s_plot, epoch_switch, num_steps


def run_nn_simulation(n_iters=20000):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    hidden_sizes = [[N] + [N]] * depth
    model = nn.Sequential(*[nn.Linear(f_in, f_out, bias=False) for (f_in, f_out) in hidden_sizes]).to(device)
    optimizer = optim.SGD(model.parameters(), lr=step)
    init_model(model, depth, initialization='identity', init_scale=0.001, seed=0)
    w_init = get_e2e(model).detach().cpu().numpy()

    singular_values_list = []
    epoch_singular_values_list = []
    n_iters_task1 = n_iters - continue_steps
    num_singular_vals = 10
    ys_2_torch = torch.from_numpy(ys_2_).to(device)
    ys_torch = torch.from_numpy(ys_).to(device)
    for T in range(n_iters):
        e2e = get_e2e(model)
        # loss = get_train_loss(e2e, ys_, us, vs) if T < n_iters_task1 else get_train_loss(e2e, ys_2_, us_2, vs_2)
        loss = get_train_loss(e2e, ys_2_torch, us, vs) if T > n_iters_task1 else get_train_loss(e2e, ys_torch, us, vs)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        if T % 10 == 0:
            # print('Iter: {}, Train Loss: {}, Test Loss: {}, Param Norm: {}'.format(T, loss, test_loss, params_norm))
            U, singular_values, V = e2e.svd()
            singular_values_list.append(singular_values[:num_singular_vals].detach().cpu().numpy())
            epoch_singular_values_list.append(T)
            print('Singular Values: {}'.format(singular_values[:num_singular_vals].detach().cpu().numpy()))

    return epoch_singular_values_list, singular_values_list, w_init


if __name__ == '__main__':
    for num_steps in [10000, 20000, 30000, 50000]:  # [200000]:  # [10000, 15000, 20000, 30000, 40000, 50000]:
        epoch_singular_values_list, singular_values_list, w_init = run_nn_simulation(n_iters=num_steps)
        s_plot, epoch_switch, _ = run_analytical(w_init, num_steps=num_steps)
        # epoch_switch = 5000
        # num_steps = 35000

        plt.figure(figsize=(3.2, 2.7))
        for i in range(10):
            plt.plot(epoch_singular_values_list, np.array(singular_values_list)[:, i], color=plt.cm.summer(i / 10))
            plt.plot(s_plot[:, i], color='black', linestyle='--', linewidth=1, alpha=0.75)

        plt.title(f'Analytical Evolution: Num obs: {num_obs}')
        plt.xlabel('Epoch')
        plt.ylabel('Singular Vals')
        plt.axvline(x=epoch_switch, linestyle='--', color='blue')
        # plt.savefig(
        #     f'plots/analytical_evolution_{num_obs}_rank1={rank1}_rank2={rank2}_step={step}_depth={depth}_numEpochs={num_steps}.pdf',
        #     bbox_inches='tight')
        plt.savefig(
            f'plots/analytical_differential_equation/simulation_evolution_{num_obs}_rank1={rank1}_rank2={rank2}_step={step}_depth={depth}_numEpochs={num_steps}_N={N}.pdf',
            bbox_inches='tight')
