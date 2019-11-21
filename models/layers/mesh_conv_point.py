import torch
import torch.nn as nn
import torch.nn.functional as F

class MeshConvPoint(nn.Module):

    def __init__(self, in_channels, out_channels, bias=True, symm_oper=None):
        super(MeshConvPoint, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, len(symm_oper)+1), bias=bias)
        self.symm_oper = symm_oper
        self.k = len(symm_oper)+1

    def __call__(self, point_f, mesh):
        return self.forward(point_f, mesh)

    def forward(self, x, mesh):
        x = x.squeeze(-1)
        G = torch.cat([self.pad_gemm(i, x.shape[2], x.device) for i in mesh], 0)
        # build 'neighborhood image' and apply convolution
        G = self.create_GeMM(x, G)
        x = self.conv(G)
        return x

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
        output dimensions: Batch x Channels x Vertices x 2
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

        return complete_f

    def pad_gemm(self, m, xsz, device):
        """ extracts face neighbors (3x for trimesh) -> m.gemm_faces
        which is of size #edges x 3
        add the edge_id itself to make #edges x 4
        then pad to desired size e.g., xsz x 4
        """
        padded_gemm = torch.tensor(m.gemm_faces, device=device).float()
        padded_gemm = padded_gemm.requires_grad_()
        padded_gemm = torch.cat((torch.arange(m.faces.shape[0], device=device).float().unsqueeze(1), padded_gemm), dim=1)
        # pad using F
        padded_gemm = F.pad(padded_gemm, (0, 0, 0, xsz - m.faces.shape[0]), "constant", 0)
        padded_gemm = padded_gemm.unsqueeze(0)
        return padded_gemm
