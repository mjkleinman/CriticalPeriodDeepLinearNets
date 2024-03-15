import numpy as np
import torch
import torch.nn as nn


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_matrix(n, rank, symmetric=False):
    r = rank
    U = np.random.randn(n, r).astype(np.float32)
    if symmetric:
        V = U
    else:
        V = np.random.randn(n, r).astype(np.float32)
    w_gt = U.dot(V.T) / np.sqrt(r)
    w_gt = w_gt / np.linalg.norm(w_gt, 'fro') * n
    return torch.from_numpy(w_gt)  # w_gt


def init_model(model, depth, initialization='orthogonal', init_scale=0.001, hidden_sizes=[100], seed=0):
    set_seed(seed)
    if initialization == 'orthogonal':
        scale = (init_scale * np.sqrt(hidden_sizes[0])) ** (1. / depth)
        matrices = []
        for param in model.parameters():
            nn.init.orthogonal_(param)
            param.data.mul_(scale)
            matrices.append(param.data.cpu().numpy())
        for a, b in zip(matrices, matrices[1:]):
            assert np.allclose(a.dot(a.T), b.T.dot(b), atol=1e-6)
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
        for param in model.parameters():
            nn.init.eye_(param)
            param.data.mul_(scale)


def get_train_loss(e2e, ys_, us, vs):
    ys = e2e[us, vs]
    return 0.5 * (ys - ys_).pow(2).mean()  # Adding the factor of 0.5 to correspond to analytical


def get_test_loss(e2e, w_gt):
    return (w_gt - e2e).view(-1).pow(2).mean()


def get_e2e(model):
    weight = None
    for fc in model.children():
        assert isinstance(fc, nn.Linear) and fc.bias is None
        if weight is None:
            weight = fc.weight.t()
        else:
            weight = fc(weight)
    return weight  # Actually removing it .T  # MK (adding transpose): Need to add the transpose to have in correct SVD basis from init for analytical solution
