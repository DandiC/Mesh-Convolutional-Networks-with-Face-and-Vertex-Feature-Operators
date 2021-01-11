import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
# from memory_profiler import profile

class MeshConvPoint(nn.Module):

    def __init__(self, in_channels, out_channels, bias=True, n_neighbors=6):
        super(MeshConvPoint, self).__init__()

        self.n_neighbors = n_neighbors

        # Set the size of the convolutional filter
        if n_neighbors <= 0:
            self.k = 1
        else:
            self.k = 1 + self.n_neighbors

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, self.k),
                              bias=bias)

    def __call__(self, point_f, mesh):
        return self.forward(point_f, mesh)

    def forward(self, x, mesh):
        x = x.squeeze(-1)

        # build 'neighborhood image' and apply convolution
        G = torch.cat([self.pad_gemm(i, x.shape[2], x.device) for i in mesh], 0)
        G = self.create_GeMM(x, G)

        return self.conv(G)


    def flatten_gemm_inds(self, Gi):
        (b, ne, nn) = Gi.shape
        ne += 1
        batch_n = torch.floor(torch.arange(b * ne, device=Gi.device).float() / ne).view(b, ne)
        add_fac = batch_n * ne
        add_fac = add_fac.view(b, ne, 1)
        add_fac = add_fac.repeat(1, 1, nn)
        # flatten Gi
        Gi = Gi.float() + add_fac[:, 1:, :]
        return Gi

    def create_GeMM(self, x, Gi):
        """ gathers the edge features (x) with from the 1-ring indices (Gi)
        applys symmetric functions to handle order invariance
        returns a 'fake image' which can use 2d convolution on
        output dimensions: Batch x Channels x Edges x 5
        """
        Gishape = Gi.shape
        # pad the first row of  every sample in batch with zeros
        padding = torch.zeros((x.shape[0], x.shape[1], 1), requires_grad=True, device=x.device)
        # padding = padding.to(x.device)
        x = torch.cat((padding, x), dim=2)
        Gi = Gi + 1 #shift

        # first flatten indices
        Gi_flat = self.flatten_gemm_inds(Gi)
        Gi_flat = Gi_flat.view(-1).long()
        #
        odim = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(odim[0] * odim[2], odim[1])

        f = torch.index_select(x, dim=0, index=Gi_flat)
        f = f.view(Gishape[0], Gishape[1], Gishape[2], -1)
        f = f.permute(0, 3, 1, 2)

        return f

    def pad_gemm(self, m, xsz, device):
        """ extracts face neighbors (3x for trimesh) -> m.gemm_faces
        which is of size #edges x 3
        add the edge_id itself to make #edges x 4
        then pad to desired size e.g., xsz x 4
        """

        padded_gemm = torch.tensor(m.gemm_vs, device=device).float()
        padded_gemm = padded_gemm.requires_grad_()
        padded_gemm = torch.cat((torch.arange(m.vs.shape[0], device=device).float().unsqueeze(1), padded_gemm), dim=1)
        # pad using F
        padded_gemm = F.pad(padded_gemm, (0, 0, 0, xsz - m.vs.shape[0]), "constant", 0)
        padded_gemm = padded_gemm.unsqueeze(0)
        return padded_gemm