import numpy as np
import torch


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_U_S_V(Y_task):
    X = torch.eye(8)
    sigma_yx = Y_task.T.mm(X) / Y_task.shape[0]
    U, S, V = torch.svd(sigma_yx, some=False)
    print(S)
    return U, S, V
