import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
# from memory_profiler import profile

class MeshConvPoint(nn.Module):

    def __init__(self, in_channels, out_channels, bias=True, symm_oper=None, n_neighbors=6, neighbor_order='random'):
        super(MeshConvPoint, self).__init__()
        self.symm_oper = symm_oper
        self.n_neighbors = n_neighbors
        self.neighbor_order = neighbor_order

        if neighbor_order == 'random':
            if n_neighbors==0:
                self.symm_oper = []
            elif n_neighbors == -1:
                self.symm_oper = [1]

            self.k = 1+len(self.symm_oper)
        else:
            self.k = 4

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, self.k),
                              bias=bias)

    def __call__(self, point_f, mesh):
        return self.forward(point_f, mesh)

    def forward(self, x, mesh):
        x = x.squeeze(-1)

        # build 'neighborhood image' and apply convolution
        if self.n_neighbors==-1:
            G = self.create_GeMM_average(x, mesh)
        else:
            if self.neighbor_order == 'random':
                G = torch.cat([self.pad_gemm_random(i, x.shape[2], x.device) for i in mesh], 0)
            else:
                G = torch.cat([self.pad_gemm_ordered(i, x.shape[2], x.device) for i in mesh], 0)
            # else:
            #     raise ValueError(self.neighbor_order, 'Wrong value in neighbor_order')
            G = self.create_GeMM(x, G)

        return self.conv(G)
        # return self.conv(G.unsqueeze(3))

    def create_GeMM_average(self, x, mesh):
        """ gathers the edge features (x) with from the 1-ring indices (Gi)
        applys symmetric functions to handle order invariance
        returns a 'fake image' which can use 2d convolution on
        output dimensions: Batch x Channels x Vertices x 2
        """
        # TODO: faster way to do this?
        G = torch.cat((x.unsqueeze(3), torch.ones(x.shape[0], x.shape[1], x.shape[2], 1).to(x.device)), 3)

        for i in range(mesh.shape[0]):
            for j in range(mesh[i].vs.shape[0]):
                # Do mean to account for different number of neighbors
                G[i, :, j, 1] = torch.mean(x[i, :, np.fromiter(mesh[i].gemm_vs[j], int, len(mesh[i].gemm_vs[j]))],
                                           dim=1)

        return G

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

        # Features without symmetric functions
        # x_1 = f[:, :, :, 1]
        # x_2 = f[:, :, :, 2]
        # x_3 = f[:, :, :, 3]
        if self.neighbor_order == 'random':
            #Symmetric functions
            complete_f = torch.unsqueeze(f[:, :, :, 0], dim=3)
            if 1 in self.symm_oper:
                x_1 = f[:, :, :, 1] + f[:, :, :, 2] + f[:, :, :, 3]
                complete_f = torch.cat([complete_f, torch.unsqueeze(x_1, 3)], dim=3)
            if 2 in self.symm_oper:
                x_2 = f[:, :, :, 1] * f[:, :, :, 2] * f[:, :, :, 3]
                complete_f = torch.cat([complete_f, torch.unsqueeze(x_2, 3)], dim=3)
            if 3 in self.symm_oper:
                x_3 = f[:, :, :, 1] * f[:, :, :, 2] + f[:, :, :, 1] * f[:, :, :, 3] + f[:, :, :, 2] * f[:, :, :, 3]
                complete_f = torch.cat([complete_f, torch.unsqueeze(x_3, 3)], dim=3)
            if 4 in self.symm_oper:
                x_4 = f[:, :, :, 1] * f[:, :, :, 1] + f[:, :, :, 2] * f[:, :, :, 2] + f[:, :, :, 3] * f[:, :, :, 3]
                complete_f = torch.cat([complete_f, torch.unsqueeze(x_4, 3)], dim=3)
            if 5 in self.symm_oper:
                x_5 = torch.abs(f[:, :, :, 1] - f[:, :, :, 2]) + torch.abs(f[:, :, :, 1] - f[:, :, :, 3]) + torch.abs(
                    f[:, :, :, 2] - f[:, :, :, 3])
                complete_f = torch.cat([complete_f, torch.unsqueeze(x_5, 3)], dim=3)
            if 6 in self.symm_oper:
                x_6 = f[:, :, :, 1] * f[:, :, :, 1] * f[:, :, :, 1] + f[:, :, :, 2] * f[:, :, :, 2] * f[:, :, :, 2] \
                      + f[:, :, :, 3] * f[:, :, :, 3] * f[:, :, :, 3]
                complete_f = torch.cat([complete_f, torch.unsqueeze(x_6, 3)], dim=3)
        else:
            return f

        return complete_f

    def pad_gemm_random(self, m, xsz, device):
        """ extracts face neighbors (3x for trimesh) -> m.gemm_faces
        which is of size #edges x 3
        add the edge_id itself to make #edges x 4
        then pad to desired size e.g., xsz x 4
        """
        rand_gemm = -np.ones((m.vs.shape[0], self.n_neighbors), dtype=int)
        for i, gemm in enumerate(m.gemm_vs):
            if self.n_neighbors>len(gemm):
                rand_gemm[i,0:len(gemm)] = np.array(list(gemm))
            else:
                rand_gemm[i,:] = np.array(random.sample(gemm,self.n_neighbors))
        padded_gemm = torch.tensor(rand_gemm, device=device).float()
        padded_gemm = padded_gemm.requires_grad_()
        padded_gemm = torch.cat((torch.arange(m.vs.shape[0], device=device).float().unsqueeze(1), padded_gemm), dim=1)
        # pad using F
        padded_gemm = F.pad(padded_gemm, (0, 0, 0, xsz - m.vs.shape[0]), "constant", 0)
        padded_gemm = padded_gemm.unsqueeze(0)
        return padded_gemm

    def pad_gemm_ordered(self, m, xsz, device):
        """ extracts face neighbors (3x for trimesh) -> m.gemm_faces
        which is of size #edges x 3
        add the edge_id itself to make #edges x 4
        then pad to desired size e.g., xsz x 4
        """

        ord_gemm = -np.ones((m.vs.shape[0], 3), dtype=int)
        for i, gemm in enumerate(m.gemm_vs):
            # TODO: This code assumes that features are [x, y, z, mean_c, gaussian_c]. Make this generic in the future
            l_gemm = list(gemm)
            if self.neighbor_order == 'mean_c':
                curv = m.features[3,l_gemm]
                order = np.argsort(curv)
                ord_gemm[i, :] = [l_gemm[order[-1]], l_gemm[order[order.size // 2]], l_gemm[order[0]]]
            elif self.neighbor_order == 'gaussian_c':
                curv = m.features[4, l_gemm]
                order = np.argsort(curv)
                ord_gemm[i, :] = [l_gemm[order[-1]], l_gemm[order[order.size // 2]], l_gemm[order[0]]]
            elif self.neighbor_order == 'closest_d':
                dist = np.linalg.norm(m.vs[l_gemm]-m.vs[i],axis=1)
                order = np.argsort(dist)
                ord_gemm[i, :] = [l_gemm[order[0]], l_gemm[order[1]], l_gemm[order[2]]]
            elif self.neighbor_order == 'farthest_d':
                dist = np.linalg.norm(m.vs[l_gemm] - m.vs[i], axis=1)
                order = np.argsort(dist)
                ord_gemm[i, :] = [l_gemm[order[-1]], l_gemm[order[-2]], l_gemm[order[-3]]]
            elif self.neighbor_order == 'median_d':
                dist = np.linalg.norm(m.vs[l_gemm] - m.vs[i], axis=1)
                order = np.argsort(dist)
                ord_gemm[i, :] = [l_gemm[order[-1]], l_gemm[order[order.size // 2]], l_gemm[order[0]]]

        padded_gemm = torch.tensor(ord_gemm, device=device).float()
        padded_gemm = padded_gemm.requires_grad_()
        padded_gemm = torch.cat((torch.arange(m.vs.shape[0], device=device).float().unsqueeze(1), padded_gemm), dim=1)
        # pad using F
        padded_gemm = F.pad(padded_gemm, (0, 0, 0, xsz - m.vs.shape[0]), "constant", 0)
        padded_gemm = padded_gemm.unsqueeze(0)
        return padded_gemm