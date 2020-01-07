import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from memory_profiler import profile

class MeshConvPoint(nn.Module):

    def __init__(self, in_channels, out_channels, bias=True, symm_oper=None):
        super(MeshConvPoint, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                              bias=bias)
        self.symm_oper = symm_oper
        self.k = 1

    def __call__(self, point_f, mesh):
        return self.forward(point_f, mesh)

    def forward(self, x, mesh):
        # x = x.squeeze(-1)

        # build 'neighborhood image' and apply convolution
        # G = self.create_GeMM(x, mesh)

        x = self.conv(x.unsqueeze(3))
        return x


    def create_GeMM(self, x, mesh):
        """ gathers the edge features (x) with from the 1-ring indices (Gi)
        applys symmetric functions to handle order invariance
        returns a 'fake image' which can use 2d convolution on
        output dimensions: Batch x Channels x Vertices x 2
        """
        # TODO: faster way to do this?
        f = torch.zeros((x.shape[0], x.shape[1], x.shape[2], len(self.symm_oper) + 1), requires_grad=True,
                        device=x.device)
        f[:, :, :, 0] = x
        for i in range(mesh.shape[0]):
            for j in range(mesh[i].vs.shape[0]):
                # Do mean to account for different number of neighbors
                f[i, :, j, 1] = torch.mean(x[i, :, np.array(list(mesh[i].gemm_vs[j]))], dim=1)

        return f