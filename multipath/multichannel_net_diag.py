import random

import torch


def perturb_omega_singular(omega, U, V, sing_idx):
    k = U.T.mm(omega.T.mm(V))
    # set singulars to zero
    k[:, sing_idx] = 0
    k[sing_idx, :] = 0
    # k = k - k[sing_idx, sing_idx]
    omega_pert = U.mm(k).mm(V.T)
    return omega_pert.T


class MultipathwayNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim, nonlinearity=None, bias=False, num_pathways=2, depth=2, width=1000,
                 eps=1e-3, hidden=None, diagonal_init=False, U=None, V=None, kDiagInit=None):  # 0.0001

        super(MultipathwayNet, self).__init__()

        # hidden is assumed to be a list with entries corresponding to each pathway, each entry a list of the widths of that pathway by depth
        # hidden!=None will override num_pathways, depth, width
        # 'depth' above is the number of weights, not the number of hidden layers

        self.diagonal_init = diagonal_init
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.eps = eps

        if hidden is None:
            hidden = []
            for pi in range(num_pathways):
                pathway = []
                for di in range(depth - 1):
                    pathway.append(width)
                hidden.append(pathway)

        self.hidden = hidden
        self.hidden_layers = []

        # Initialize matrices to be in correct SVD basis
        # so that W_D = U D R^T, W_Di = R D R^T, ..., W_1 = R D V^T
        # where R is orthogonal, D is diagonal, here we use the same D to ensure weights are balanced
        # to match dimensions we ensure R is rank constrained to min(input_dim, output_dim)
        rank = min(self.input_dim, self.output_dim)
        R_tensor = torch.empty(width, rank)
        torch.nn.init.orthogonal_(R_tensor)
        diag_values = kDiagInit ** (1 / depth)
        diag_matrix = torch.diag(diag_values)

        R_tensor_h = torch.empty(width, width)
        torch.nn.init.orthogonal_(R_tensor_h)

        for pathway in self.hidden:
            op_list = []
            for di, depth in enumerate(pathway):
                print(depth)
                if di == 0:
                    op = torch.nn.Linear(self.input_dim, depth, bias=self.bias)
                    op.weight = torch.nn.Parameter(R_tensor_h @ diag_matrix[:, :rank] @ V.T, requires_grad=True)
                else:
                    op = torch.nn.Linear(pathway[di - 1], depth, bias=self.bias)
                    op.weight = torch.nn.Parameter(R_tensor_h @ diag_matrix @ R_tensor_h.T, requires_grad=True)

                if op.bias is not None:
                    op.bias = torch.nn.Parameter(torch.zeros_like(op.bias), requires_grad=True)
                op_list.append(op)

            op = torch.nn.Linear(pathway[-1], self.output_dim, bias=self.bias)
            op.weight = torch.nn.Parameter(U[:, :rank] @ diag_matrix[:rank, :] @ R_tensor_h.T, requires_grad=True)
            if op.bias is not None:
                op.bias = torch.nn.Parameter(torch.zeros_like(op.bias), requires_grad=True)
            op_list.append(op)

            self.hidden_layers.append(op_list)

        # MK Why do they need to do this?
        for pi, op_list in enumerate(self.hidden_layers):
            for oi, op in enumerate(op_list):
                self.register_parameter(name="Path_{}_Depth_{}_weight".format(pi, oi), param=op.weight)
                self.register_parameter(name="Path_{}_Depth_{}_bias".format(pi, oi), param=op.bias)

        if self.nonlinearity is not None:
            temp_layers = self.hidden_layers
            self.hidden_layers = []
            for op_list in temp_layers:
                new_op_list = []
                for op in op_list:
                    new_op_list.append(op)
                    new_op_list.append(self.nonlinearity)
                self.hidden_layers.append(new_op_list)

    # Quick overwriting forward to test if I can remove one singular value (in development)
    def forward(self, x, deficit=None, U=None, V=None, sing_idx=None):
        # deficit: list
        output_a, output_b = self.omega_torch()
        # Modify the first singular value of output_a
        if deficit is not None:
            if 'gate_sing' in deficit:
                output_a = perturb_omega_singular(output_a, U, V, sing_idx)
            if 'gate_path' in deficit:
                output_b = output_b.detach()
            elif 'prob_gate_path' in deficit:
                if random.random() < 0.9:
                    output_b = output_b.detach()
        output = output_a + output_b
        return output

    # MK: This is a function that returns Omega_a, Omega_b
    def omega(self):
        with torch.no_grad():
            x = torch.eye(self.input_dim).to(self.hidden_layers[0][0].weight.device)
            output = []
            for op_list in self.hidden_layers:
                xtemp = x
                for op in op_list:
                    xtemp = op(xtemp)
                output.append(xtemp.T.detach())
        return output

    def omega_torch(self):
        x = torch.eye(self.input_dim).to(self.hidden_layers[0][0].weight.device)
        output = []
        for op_list in self.hidden_layers:
            xtemp = x
            for op in op_list:
                xtemp = op(xtemp)
            output.append(xtemp)
        return output
