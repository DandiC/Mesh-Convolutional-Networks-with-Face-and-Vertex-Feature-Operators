import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from models.layers.mesh_conv import MeshConv
from models.layers.mesh_conv_face import MeshConvFace
import torch.nn.functional as F
from models.layers.mesh_pool import MeshPool
from models.layers.mesh_pool_face import MeshPoolFace
from models.layers.mesh_unpool import MeshUnpool
from models.layers.mesh_unpool_face import MeshUnpoolFace
from models.layers.mesh_unpool_f import MeshUnpool_F
from models.layers.mesh_conv_point import MeshConvPoint
from models.layers.mesh_unpool_point import MeshUnpoolPoint
from models.layers.mesh_pool_point import MeshPoolPoint
from models.layers.mesh import Mesh
from models.layers.mesh_prepare import build_gemm
import numpy as np
from util.util import pad
import os
import wandb


# from memory_profiler import profile

###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance', num_groups=1):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'group':
        norm_layer = functools.partial(nn.GroupNorm, affine=True, num_groups=num_groups)
    elif norm_type == 'none':
        norm_layer = NoNorm
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_norm_args(norm_layer, nfeats_list):
    if hasattr(norm_layer, '__name__') and norm_layer.__name__ == 'NoNorm':
        norm_args = [{'fake': True} for f in nfeats_list]
    elif norm_layer.func.__name__ == 'GroupNorm':
        norm_args = [{'num_channels': f} for f in nfeats_list]
    elif norm_layer.func.__name__ == 'BatchNorm2d':
        norm_args = [{'num_features': f} for f in nfeats_list]
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_layer.func.__name__)
    return norm_args


class NoNorm(nn.Module):  # todo with abstractclass and pass
    def __init__(self, fake=True):
        self.fake = fake
        super(NoNorm, self).__init__()

    def forward(self, x):
        return x

    def __call__(self, x):
        return self.forward(x)


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type, init_gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(net, init_type, init_gain, gpu_ids, generative=False):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        if not generative:
            net.cuda(gpu_ids[0])
            net = net.cuda()
            net = torch.nn.DataParallel(net, gpu_ids)
        else:
            net.generator.cuda(gpu_ids[0])
            net.generator = net.generator.cuda()
            net.generator = torch.nn.DataParallel(net.generator, gpu_ids)

            net.discriminator.cuda(gpu_ids[0])
            net.discriminator = net.discriminator.cuda()
            net.discriminator = torch.nn.DataParallel(net.discriminator, gpu_ids)

    if init_type != 'none':
        init_weights(net, init_type, init_gain)
    return net


def define_classifier(input_nc, ncf, ninput_features, nclasses, opt, gpu_ids, arch, init_type, init_gain, feat_from,
                      device=None):
    net = None
    norm_layer = get_norm_layer(norm_type=opt.norm, num_groups=opt.num_groups)
    generative = False
    if opt.dataset_mode == 'generative':
        export_folder = os.path.join(opt.checkpoints_dir, opt.name, 'generated')
        if not os.path.exists(export_folder):
            os.makedirs(export_folder)

    if arch == 'mconvnet':
        if feat_from == 'edge':
            net = MeshConvNet(norm_layer, input_nc, ncf, nclasses, ninput_features, opt.pool_res, opt.fc_n,
                              opt.resblocks)
        elif feat_from == 'face':
            net = MeshConvNetFace(norm_layer, input_nc, ncf, nclasses, ninput_features, opt.pool_res, opt.fc_n,
                                  opt.resblocks, symm_oper=opt.symm_oper)
        elif feat_from == 'point':
            net = MeshConvNetPoint(norm_layer, input_nc, ncf, nclasses, ninput_features, opt.pool_res, opt.fc_n,
                                   opt.resblocks, symm_oper=opt.symm_oper, n_neighbors=opt.n_neighbors)
    elif arch == 'meshunet':
        down_convs = [input_nc] + ncf
        up_convs = ncf[::-1] + [nclasses]
        pool_res = [ninput_features] + opt.pool_res
        if feat_from == 'edge':
            net = MeshEncoderDecoder(pool_res,
                                     down_convs,
                                     up_convs,
                                     blocks=opt.resblocks,
                                     transfer_data=True)
        elif feat_from == 'face':
            net = MeshEncoderDecoderFace(pool_res,
                                         down_convs,
                                         up_convs,
                                         blocks=opt.resblocks,
                                         transfer_data=True,
                                         symm_oper=opt.symm_oper)
        elif feat_from == 'point':
            net = MeshEncoderDecoderPoint(pool_res,
                                          down_convs,
                                          up_convs,
                                          blocks=opt.resblocks,
                                          transfer_data=True,
                                          symm_oper=opt.symm_oper,
                                          n_neighbors=opt.n_neighbors,
                                          neighbor_order=opt.neighbor_order)
    elif arch == 'meshGAN':
        down_convs = [input_nc] + ncf
        up_convs = [1] + ncf[::-1] + [9]
        pool_res = [ninput_features] + opt.pool_res
        net = MeshGAN(pool_res, down_convs, up_convs, blocks=opt.resblocks,
                      transfer_data=False, symm_oper=opt.symm_oper)
        generative = True
    elif arch == 'meshPointGAN':
        up_convs = [3] + ncf[::-1] + [3]
        net = MeshPointGAN(opt, ncf, norm_layer, input_nc, ninput_features, export_folder=export_folder)
        generative = True
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % arch)
    return init_net(net, init_type, init_gain, gpu_ids, generative=generative)


def define_loss(opt):
    if opt.dataset_mode == 'classification':
        loss = torch.nn.CrossEntropyLoss()
    elif opt.dataset_mode == 'segmentation':
        loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    elif opt.dataset_mode == 'generative':
        if opt.arch == 'meshunet':
            loss = torch.nn.MSELoss()
        else:
            disc_loss = torch.nn.BCELoss()
            gen_loss = torch.nn.BCELoss()
            loss = [disc_loss, gen_loss]
    return loss


##############################################################################
# Classes For Classification / Segmentation Networks
##############################################################################
class MeshConvNetPoint(nn.Module):
    """Network for learning a global shape descriptor (classification)
    """

    def __init__(self, norm_layer, nf0, conv_res, nclasses, input_res, pool_res, fc_n,
                 nresblocks=3, symm_oper=None, n_neighbors=6):
        super(MeshConvNetPoint, self).__init__()
        self.k = [nf0] + conv_res
        self.res = [input_res] + pool_res
        norm_args = get_norm_args(norm_layer, self.k[1:])

        for i, ki in enumerate(self.k[:-1]):
            setattr(self, 'conv{}'.format(i),
                    MResConvPoint(ki, self.k[i + 1], nresblocks, symm_oper=symm_oper, n_neighbors=n_neighbors))
            setattr(self, 'norm{}'.format(i), norm_layer(**norm_args[i]))
            setattr(self, 'pool{}'.format(i), MeshPoolPoint(self.res[i + 1]))

        self.gp = torch.nn.AvgPool1d(self.res[-1])
        # self.gp = torch.nn.MaxPool1d(self.res[-1])
        self.fc1 = nn.Linear(self.k[-1], fc_n)
        self.fc2 = nn.Linear(fc_n, nclasses)

    def forward(self, x, mesh):

        for i in range(len(self.k) - 1):
            x = getattr(self, 'conv{}'.format(i))(x, mesh)
            x = F.relu(getattr(self, 'norm{}'.format(i))(x))
            x = getattr(self, 'pool{}'.format(i))(x, mesh)

        x = self.gp(x)
        x = x.view(-1, self.k[-1])

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MResConvPoint(nn.Module):
    def __init__(self, in_channels, out_channels, skips=1, symm_oper=None, relu=True, n_neighbors=6):
        super(MResConvPoint, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skips = skips
        self.conv0 = MeshConvPoint(self.in_channels, self.out_channels, bias=False, symm_oper=symm_oper,
                                   n_neighbors=n_neighbors)
        self.relu = relu
        for i in range(self.skips):
            setattr(self, 'bn{}'.format(i + 1), nn.BatchNorm2d(self.out_channels))
            setattr(self, 'conv{}'.format(i + 1),
                    MeshConvPoint(self.out_channels, self.out_channels, bias=False, symm_oper=symm_oper))

    def forward(self, x, mesh):
        x = self.conv0(x, mesh)
        x1 = x
        for i in range(self.skips):
            x = getattr(self, 'bn{}'.format(i + 1))(F.relu(x))
            x = getattr(self, 'conv{}'.format(i + 1))(x, mesh)
        x += x1
        if self.relu:
            x = F.relu(x)
        return x


class MResConvFace(nn.Module):
    def __init__(self, in_channels, out_channels, skips=1, symm_oper=None):
        super(MResConvFace, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skips = skips
        self.conv0 = MeshConvFace(self.in_channels, self.out_channels, bias=False, symm_oper=symm_oper)
        for i in range(self.skips):
            setattr(self, 'bn{}'.format(i + 1), nn.BatchNorm2d(self.out_channels))
            setattr(self, 'conv{}'.format(i + 1),
                    MeshConvFace(self.out_channels, self.out_channels, bias=False, symm_oper=symm_oper))

    def forward(self, x, mesh):
        x = self.conv0(x, mesh)
        x1 = x
        for i in range(self.skips):
            x = getattr(self, 'bn{}'.format(i + 1))(F.relu(x))
            x = getattr(self, 'conv{}'.format(i + 1))(x, mesh)
        x += x1
        x = F.relu(x)
        return x


class MeshConvNetFace(nn.Module):
    """Network for learning a global shape descriptor (classification)
    """

    def __init__(self, norm_layer, nf0, conv_res, nclasses, input_res, pool_res, fc_n,
                 nresblocks=3, symm_oper=None):
        super(MeshConvNetFace, self).__init__()
        self.k = [nf0] + conv_res
        self.res = [input_res] + pool_res
        norm_args = get_norm_args(norm_layer, self.k[1:])

        for i, ki in enumerate(self.k[:-1]):
            setattr(self, 'conv{}'.format(i), MResConvFace(ki, self.k[i + 1], nresblocks, symm_oper=symm_oper))
            setattr(self, 'norm{}'.format(i), norm_layer(**norm_args[i]))
            setattr(self, 'pool{}'.format(i), MeshPoolFace(self.res[i + 1]))

        self.gp = torch.nn.AvgPool1d(self.res[-1])
        # self.gp = torch.nn.MaxPool1d(self.res[-1])
        self.fc1 = nn.Linear(self.k[-1], fc_n)
        self.fc2 = nn.Linear(fc_n, nclasses)

    def forward(self, x, mesh):
        for i in range(len(self.k) - 1):
            x = getattr(self, 'conv{}'.format(i))(x, mesh)
            x = F.relu(getattr(self, 'norm{}'.format(i))(x))
            x = getattr(self, 'pool{}'.format(i))(x, mesh)

        x = self.gp(x)
        x = x.view(-1, self.k[-1])

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MResConvFace(nn.Module):
    def __init__(self, in_channels, out_channels, skips=1, symm_oper=None):
        super(MResConvFace, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skips = skips
        self.conv0 = MeshConvFace(self.in_channels, self.out_channels, bias=False, symm_oper=symm_oper)
        for i in range(self.skips):
            setattr(self, 'bn{}'.format(i + 1), nn.BatchNorm2d(self.out_channels))
            setattr(self, 'conv{}'.format(i + 1),
                    MeshConvFace(self.out_channels, self.out_channels, bias=False, symm_oper=symm_oper))

    def forward(self, x, mesh):
        x = self.conv0(x, mesh)
        x1 = x
        for i in range(self.skips):
            x = getattr(self, 'bn{}'.format(i + 1))(F.relu(x))
            x = getattr(self, 'conv{}'.format(i + 1))(x, mesh)
        x += x1
        x = F.relu(x)
        return x


class MeshConvNet(nn.Module):
    """Network for learning a global shape descriptor (classification)
    """

    def __init__(self, norm_layer, nf0, conv_res, nclasses, input_res, pool_res, fc_n,
                 nresblocks=3):
        super(MeshConvNet, self).__init__()
        self.k = [nf0] + conv_res
        self.res = [input_res] + pool_res
        norm_args = get_norm_args(norm_layer, self.k[1:])

        for i, ki in enumerate(self.k[:-1]):
            setattr(self, 'conv{}'.format(i), MResConv(ki, self.k[i + 1], nresblocks))
            setattr(self, 'norm{}'.format(i), norm_layer(**norm_args[i]))
            setattr(self, 'pool{}'.format(i), MeshPool(self.res[i + 1]))

        self.gp = torch.nn.AvgPool1d(self.res[-1])
        # self.gp = torch.nn.MaxPool1d(self.res[-1])
        self.fc1 = nn.Linear(self.k[-1], fc_n)
        self.fc2 = nn.Linear(fc_n, nclasses)

    def forward(self, x, mesh):

        for i in range(len(self.k) - 1):
            x = getattr(self, 'conv{}'.format(i))(x, mesh)
            x = F.relu(getattr(self, 'norm{}'.format(i))(x))
            x = getattr(self, 'pool{}'.format(i))(x, mesh)

        x = self.gp(x)
        x = x.view(-1, self.k[-1])

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MResConv(nn.Module):
    def __init__(self, in_channels, out_channels, skips=1):
        super(MResConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skips = skips
        self.conv0 = MeshConv(self.in_channels, self.out_channels, bias=False)
        for i in range(self.skips):
            setattr(self, 'bn{}'.format(i + 1), nn.BatchNorm2d(self.out_channels))
            setattr(self, 'conv{}'.format(i + 1),
                    MeshConv(self.out_channels, self.out_channels, bias=False))

    def forward(self, x, mesh):
        x = self.conv0(x, mesh)
        x1 = x
        for i in range(self.skips):
            x = getattr(self, 'bn{}'.format(i + 1))(F.relu(x))
            x = getattr(self, 'conv{}'.format(i + 1))(x, mesh)
        x += x1
        x = F.relu(x)
        return x


class MeshEncoderDecoder(nn.Module):
    """Network for fully-convolutional tasks (segmentation)
    """

    def __init__(self, pools, down_convs, up_convs, blocks=0, transfer_data=True):
        super(MeshEncoderDecoder, self).__init__()
        self.transfer_data = transfer_data
        self.encoder = MeshEncoder(pools, down_convs, blocks=blocks)
        unrolls = pools[:-1].copy()
        unrolls.reverse()
        self.decoder = MeshDecoder(unrolls, up_convs, blocks=blocks, transfer_data=transfer_data)

    def forward(self, x, meshes):
        fe, before_pool = self.encoder((x, meshes))
        fe = self.decoder((fe, meshes), before_pool)
        # meshes[0].export(file=meshes[0].filename)
        return fe

    def __call__(self, x, meshes):
        return self.forward(x, meshes)


class MeshEncoderDecoderFace(nn.Module):
    """Network for fully-convolutional tasks (segmentation) using face-based features
    """

    def __init__(self, pools, down_convs, up_convs, blocks=0, transfer_data=True, symm_oper=None):
        super(MeshEncoderDecoderFace, self).__init__()
        self.transfer_data = transfer_data
        self.encoder = MeshEncoderFace(pools, down_convs, blocks=blocks, symm_oper=symm_oper)
        unrolls = pools[:-1].copy()
        unrolls.reverse()
        self.decoder = MeshDecoderFace(unrolls, up_convs, blocks=blocks, transfer_data=transfer_data,
                                       symm_oper=symm_oper)

    def forward(self, x, meshes):
        fe, before_pool = self.encoder((x, meshes))
        fe = self.decoder((fe, meshes), before_pool)
        # meshes[0].export(file=meshes[0].filename)
        return fe

    def __call__(self, x, meshes):
        return self.forward(x, meshes)


class MeshEncoderDecoderPoint(nn.Module):
    """Network for fully-convolutional tasks (segmentation) using face-based features
    """

    def __init__(self, pools, down_convs, up_convs, blocks=0, transfer_data=True, symm_oper=None, n_neighbors=3,
                 neighbor_order='closest_d'):
        super(MeshEncoderDecoderPoint, self).__init__()
        self.transfer_data = transfer_data

        self.encoder = MeshEncoderPointSeg(pools,
                                           down_convs,
                                           blocks=blocks,
                                           symm_oper=symm_oper,
                                           n_neighbors=n_neighbors,
                                           neighbor_order=neighbor_order)
        unrolls = pools[:-1].copy()
        unrolls.reverse()
        self.decoder = MeshDecoderPointSeg(unrolls,
                                           up_convs,
                                           blocks=blocks,
                                           transfer_data=transfer_data,
                                           symm_oper=symm_oper,
                                           n_neighbors=n_neighbors,
                                           neighbor_order=neighbor_order)

    def forward(self, x, meshes):
        fe, before_pool = self.encoder((x, meshes))
        fe = self.decoder((fe, meshes), before_pool)
        return fe

    def __call__(self, x, meshes):
        return self.forward(x, meshes)


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=0, pool=0):
        super(DownConv, self).__init__()
        self.bn = []
        self.pool = None
        self.conv1 = MeshConv(in_channels, out_channels)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(MeshConv(out_channels, out_channels))
            self.conv2 = nn.ModuleList(self.conv2)
        for _ in range(blocks + 1):
            self.bn.append(nn.InstanceNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        if pool:
            self.pool = MeshPool(pool)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        fe, meshes = x
        x1 = self.conv1(fe, meshes)
        if self.bn:
            x1 = self.bn[0](x1)
        x1 = F.relu(x1)
        x2 = x1
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1, meshes)
            if self.bn:
                x2 = self.bn[idx + 1](x2)
            x2 = x2 + x1
            x2 = F.relu(x2)
            x1 = x2
        x2 = x2.squeeze(3)
        before_pool = x2
        if self.pool:
            x2 = self.pool(x2, meshes)
        return x2, before_pool


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=0, unroll=0, residual=True,
                 batch_norm=True, transfer_data=True):
        super(UpConv, self).__init__()
        self.residual = residual
        self.bn = []
        self.unroll = None
        self.transfer_data = transfer_data
        self.up_conv = MeshConv(in_channels, out_channels)
        if transfer_data:
            self.conv1 = MeshConv(2 * out_channels, out_channels)
        else:
            self.conv1 = MeshConv(out_channels, out_channels)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(MeshConv(out_channels, out_channels))
            self.conv2 = nn.ModuleList(self.conv2)
        if batch_norm:
            for _ in range(blocks + 1):
                self.bn.append(nn.InstanceNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        if unroll:
            self.unroll = MeshUnpool(unroll)

    def __call__(self, x, from_down=None):
        return self.forward(x, from_down)

    def forward(self, x, from_down):
        from_up, meshes = x
        x1 = self.up_conv(from_up, meshes).squeeze(3)
        if self.unroll:
            x1 = self.unroll(x1, meshes)
        if self.transfer_data:
            x1 = torch.cat((x1, from_down), 1)
        x1 = self.conv1(x1, meshes)
        if self.bn:
            x1 = self.bn[0](x1)
        x1 = F.relu(x1)
        x2 = x1
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1, meshes)
            if self.bn:
                x2 = self.bn[idx + 1](x2)
            if self.residual:
                x2 = x2 + x1
            x2 = F.relu(x2)
            x1 = x2
        x2 = x2.squeeze(3)
        return x2


class MeshEncoder(nn.Module):
    def __init__(self, pools, convs, fcs=None, blocks=0, global_pool=None):
        super(MeshEncoder, self).__init__()
        self.fcs = None
        self.convs = []
        for i in range(len(convs) - 1):
            if i + 1 < len(pools):
                pool = pools[i + 1]
            else:
                pool = 0
            self.convs.append(DownConv(convs[i], convs[i + 1], blocks=blocks, pool=pool))
        self.global_pool = None
        if fcs is not None:
            self.fcs = []
            self.fcs_bn = []
            last_length = convs[-1]
            if global_pool is not None:
                if global_pool == 'max':
                    self.global_pool = nn.MaxPool1d(pools[-1])
                elif global_pool == 'avg':
                    self.global_pool = nn.AvgPool1d(pools[-1])
                else:
                    assert False, 'global_pool %s is not defined' % global_pool
            else:
                last_length *= pools[-1]
            if fcs[0] == last_length:
                fcs = fcs[1:]
            for length in fcs:
                self.fcs.append(nn.Linear(last_length, length))
                self.fcs_bn.append(nn.InstanceNorm1d(length))
                last_length = length
            self.fcs = nn.ModuleList(self.fcs)
            self.fcs_bn = nn.ModuleList(self.fcs_bn)
        self.convs = nn.ModuleList(self.convs)
        reset_params(self)

    def forward(self, x):
        fe, meshes = x
        encoder_outs = []
        for conv in self.convs:
            fe, before_pool = conv((fe, meshes))
            encoder_outs.append(before_pool)
        if self.fcs is not None:
            if self.global_pool is not None:
                fe = self.global_pool(fe)
            fe = fe.contiguous().view(fe.size()[0], -1)
            for i in range(len(self.fcs)):
                fe = self.fcs[i](fe)
                if self.fcs_bn:
                    x = fe.unsqueeze(1)
                    fe = self.fcs_bn[i](x).squeeze(1)
                if i < len(self.fcs) - 1:
                    fe = F.relu(fe)
        return fe, encoder_outs

    def __call__(self, x):
        return self.forward(x)


class MeshDecoder(nn.Module):
    def __init__(self, unrolls, convs, blocks=0, batch_norm=True, transfer_data=True):
        super(MeshDecoder, self).__init__()
        self.up_convs = []
        for i in range(len(convs) - 2):
            if i < len(unrolls):
                unroll = unrolls[i]
            else:
                unroll = 0
            self.up_convs.append(UpConv(convs[i], convs[i + 1], blocks=blocks, unroll=unroll,
                                        batch_norm=batch_norm, transfer_data=transfer_data))
        self.final_conv = UpConv(convs[-2], convs[-1], blocks=blocks, unroll=False,
                                 batch_norm=batch_norm, transfer_data=False)
        self.up_convs = nn.ModuleList(self.up_convs)
        reset_params(self)

    def forward(self, x, encoder_outs=None):
        fe, meshes = x
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i + 2)]
            fe = up_conv((fe, meshes), before_pool)
        fe = self.final_conv((fe, meshes))
        return fe

    def __call__(self, x, encoder_outs=None):
        return self.forward(x, encoder_outs)


class MeshEncoderFace(nn.Module):
    def __init__(self, pools, convs, fcs=None, blocks=0, global_pool=None, symm_oper=None):
        super(MeshEncoderFace, self).__init__()
        self.fcs = None
        self.convs = []
        for i in range(len(convs) - 1):
            if i + 1 < len(pools):
                pool = pools[i + 1]
            else:
                pool = 0
            self.convs.append(DownConvFace(convs[i], convs[i + 1], blocks=blocks, pool=pool, symm_oper=symm_oper))
        self.global_pool = None
        if fcs is not None:
            self.fcs = []
            self.fcs_bn = []
            last_length = convs[-1]
            if global_pool is not None:
                if global_pool == 'max':
                    self.global_pool = nn.MaxPool1d(pools[-1])
                elif global_pool == 'avg':
                    self.global_pool = nn.AvgPool1d(pools[-1])
                else:
                    assert False, 'global_pool %s is not defined' % global_pool
            else:
                last_length *= pools[-1]
            if fcs[0] == last_length:
                fcs = fcs[1:]
            for length in fcs:
                self.fcs.append(nn.Linear(last_length, length))
                self.fcs_bn.append(nn.InstanceNorm1d(length))
                last_length = length
            self.fcs = nn.ModuleList(self.fcs)
            self.fcs_bn = nn.ModuleList(self.fcs_bn)
        self.convs = nn.ModuleList(self.convs)
        reset_params(self)

    def forward(self, x):
        fe, meshes = x
        encoder_outs = []
        # meshes[0].export(file='pool_0.obj')
        i = 1
        for conv in self.convs:
            fe, before_pool = conv((fe, meshes))
            # meshes[0].export(file='pool_'+str(i)+'.obj')
            i += 1
            encoder_outs.append(before_pool)
        if self.fcs is not None:
            if self.global_pool is not None:
                fe = self.global_pool(fe)
            fe = fe.contiguous().view(fe.size()[0], -1)
            for i in range(len(self.fcs)):
                fe = self.fcs[i](fe)
                if self.fcs_bn:
                    x = fe.unsqueeze(1)
                    fe = self.fcs_bn[i](x).squeeze(1)
                if i < len(self.fcs) - 1:
                    fe = F.relu(fe)
        return fe, encoder_outs

    def __call__(self, x):
        return self.forward(x)


class MeshDecoderFace(nn.Module):
    def __init__(self, unrolls, convs, blocks=0, batch_norm=True, transfer_data=True, symm_oper=None):
        super(MeshDecoderFace, self).__init__()
        self.up_convs = []
        for i in range(len(convs) - 2):
            if i < len(unrolls):
                unroll = unrolls[i]
            else:
                unroll = 0
            self.up_convs.append(UpConvFace(convs[i], convs[i + 1], blocks=blocks, unroll=unroll,
                                            batch_norm=batch_norm, transfer_data=transfer_data, symm_oper=symm_oper))
        self.final_conv = UpConvFace(convs[-2], convs[-1], blocks=blocks, unroll=False,
                                     batch_norm=batch_norm, transfer_data=False, symm_oper=symm_oper)
        self.up_convs = nn.ModuleList(self.up_convs)
        reset_params(self)

    def forward(self, x, encoder_outs=None):
        fe, meshes = x
        # meshes[0].export(file='unpool_' + str(0) + '.obj')
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i + 2)]
            fe = up_conv((fe, meshes), before_pool)
            # meshes[0].export(file='unpool_' + str(i+1) + '.obj')
        fe = self.final_conv((fe, meshes))
        return fe

    def __call__(self, x, encoder_outs=None):
        return self.forward(x, encoder_outs)


class MeshEncoderPointSeg(nn.Module):
    def __init__(self, pools, convs, fcs=None, blocks=0, global_pool=None, symm_oper=None, n_neighbors=3,
                 neighbor_order='closest_d'):
        super(MeshEncoderPointSeg, self).__init__()
        self.fcs = None
        self.convs = []
        for i in range(len(convs) - 1):
            if i + 1 < len(pools):
                pool = pools[i + 1]
            else:
                pool = 0
            self.convs.append(DownConvPoint(convs[i], convs[i + 1], blocks=blocks, pool=pool, symm_oper=symm_oper,
                                            n_neighbors=n_neighbors, neighbor_order=neighbor_order))
        self.global_pool = None
        if fcs is not None:
            self.fcs = []
            self.fcs_bn = []
            last_length = convs[-1]
            if global_pool is not None:
                if global_pool == 'max':
                    self.global_pool = nn.MaxPool1d(pools[-1])
                elif global_pool == 'avg':
                    self.global_pool = nn.AvgPool1d(pools[-1])
                else:
                    assert False, 'global_pool %s is not defined' % global_pool
            else:
                last_length *= pools[-1]
            if fcs[0] == last_length:
                fcs = fcs[1:]
            for length in fcs:
                self.fcs.append(nn.Linear(last_length, length))
                self.fcs_bn.append(nn.InstanceNorm1d(length))
                last_length = length
            self.fcs = nn.ModuleList(self.fcs)
            self.fcs_bn = nn.ModuleList(self.fcs_bn)
        self.convs = nn.ModuleList(self.convs)
        reset_params(self)

    def forward(self, x):
        fe, meshes = x
        encoder_outs = []
        for conv in self.convs:
            fe, before_pool = conv((fe, meshes))
            encoder_outs.append(before_pool)
        if self.fcs is not None:
            if self.global_pool is not None:
                fe = self.global_pool(fe)
            fe = fe.contiguous().view(fe.size()[0], -1)
            for i in range(len(self.fcs)):
                fe = self.fcs[i](fe)
                if self.fcs_bn:
                    x = fe.unsqueeze(1)
                    fe = self.fcs_bn[i](x).squeeze(1)
                if i < len(self.fcs) - 1:
                    fe = F.relu(fe)
        return fe, encoder_outs

    def __call__(self, x):
        return self.forward(x)


class MeshDecoderPointSeg(nn.Module):
    def __init__(self, unrolls, convs, blocks=0, batch_norm=True, transfer_data=True, symm_oper=None, n_neighbors=3,
                 neighbor_order='closest_d'):
        super(MeshDecoderPointSeg, self).__init__()
        self.up_convs = []
        for i in range(len(convs) - 2):
            if i < len(unrolls):
                unroll = unrolls[i]
            else:
                unroll = 0
            self.up_convs.append(UpConvPoint(convs[i],
                                             convs[i + 1],
                                             blocks=blocks,
                                             unroll=unroll,
                                             batch_norm=batch_norm,
                                             transfer_data=transfer_data,
                                             symm_oper=symm_oper,
                                             n_neighbors=n_neighbors,
                                             neighbor_order=neighbor_order,
                                             relu=True))
        self.final_conv = UpConvPoint(convs[-2],
                                      convs[-1],
                                      blocks=blocks,
                                      unroll=False,
                                      batch_norm=batch_norm,
                                      transfer_data=False,
                                      symm_oper=symm_oper,
                                      n_neighbors=n_neighbors,
                                      neighbor_order=neighbor_order,
                                      relu=True)
        self.up_convs = nn.ModuleList(self.up_convs)
        reset_params(self)

    def forward(self, x, encoder_outs=None):
        fe, meshes = x
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i + 2)]
            fe = up_conv((fe, meshes), before_pool)
        fe = self.final_conv((fe, meshes))
        return fe

    def __call__(self, x, encoder_outs=None):
        return self.forward(x, encoder_outs)


def reset_params(model):  # todo replace with my init
    for i, m in enumerate(model.modules()):
        weight_init(m)


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


class DownConvFace(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=0, pool=0, symm_oper=None):
        super(DownConvFace, self).__init__()
        self.bn = []
        self.pool = None
        self.conv1 = MeshConvFace(in_channels, out_channels, symm_oper=symm_oper)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(MeshConvFace(out_channels, out_channels, symm_oper=symm_oper))
            self.conv2 = nn.ModuleList(self.conv2)
        for _ in range(blocks + 1):
            self.bn.append(nn.BatchNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        if pool:
            self.pool = MeshPoolFace(pool)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        fe, meshes = x
        x1 = self.conv1(fe, meshes)
        if self.bn:
            x1 = self.bn[0](x1)
        x1 = F.leaky_relu(x1, negative_slope=0.2)
        x2 = x1
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1, meshes)
            if self.bn:
                x2 = self.bn[idx + 1](x2)
            x2 = x2 + x1
            x2 = F.leaky_relu(x2, negative_slope=0.2)
            x1 = x2
        x2 = x2.squeeze(3)
        before_pool = x2
        if self.pool:
            x2 = self.pool(x2, meshes)
        return x2, before_pool


class UpConvFace(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=0, unroll=0, residual=True,
                 batch_norm=True, transfer_data=True, symm_oper=None, relu=True):
        super(UpConvFace, self).__init__()
        self.residual = residual
        self.bn = []
        self.unroll = None
        self.transfer_data = transfer_data
        self.up_conv = MeshConvFace(in_channels, out_channels, symm_oper=symm_oper)
        self.relu = relu
        if transfer_data:
            self.conv1 = MeshConvFace(2 * out_channels, out_channels, symm_oper=symm_oper)
        else:
            self.conv1 = MeshConvFace(out_channels, out_channels, symm_oper=symm_oper)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(MeshConvFace(out_channels, out_channels, symm_oper=symm_oper))
            self.conv2 = nn.ModuleList(self.conv2)
        if batch_norm:
            for _ in range(blocks + 1):
                self.bn.append(nn.BatchNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        if unroll:
            self.unroll = MeshUnpool_F(unroll)

    def __call__(self, x, from_down=None):
        return self.forward(x, from_down)

    def forward(self, x, from_down):
        from_up, meshes = x
        x1 = self.up_conv(from_up, meshes).squeeze(3)
        if self.unroll:
            x1 = self.unroll(x1, meshes)
        if self.transfer_data:
            x1 = torch.cat((x1, from_down), 1)
        x1 = self.conv1(x1, meshes)
        if self.bn:
            x1 = self.bn[0](x1)

        if self.relu:
            x1 = F.relu(x1)
        x2 = x1
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1, meshes)
            if self.bn:
                x2 = self.bn[idx + 1](x2)
            if self.residual:
                x2 = x2 + x1
            if self.relu:
                x2 = F.relu(x2)
            x1 = x2
        x2 = x2.squeeze(3)
        return x2


# GAN NETWORK CODE

class MeshGAN(nn.Module):
    """Network for fully-convolutional tasks (segmentation)
    """

    def __init__(self, pools, down_convs, up_convs, blocks=0, transfer_data=False, symm_oper=None):
        super(MeshGAN, self).__init__()
        self.transfer_data = transfer_data
        self.discriminator = MeshDiscriminator(pools, down_convs, fcs=[1], blocks=blocks, global_pool='avg',
                                               symm_oper=symm_oper)
        unrolls = pools[::].copy()
        unrolls.reverse()
        self.generator = MeshGenerator(unrolls, up_convs, blocks=blocks, transfer_data=transfer_data,
                                       symm_oper=symm_oper)

    # def forward(self, x, meshes):
    #     fe, before_pool = self.encoder((x, meshes))
    #     fe = self.decoder((fe, meshes), before_pool)
    #     return fe
    #
    # def __call__(self, x, meshes):
    #     return self.forward(x, meshes)


class MeshDiscriminator(nn.Module):
    def __init__(self, pools, convs, fcs=None, blocks=0, global_pool=None, symm_oper=None):
        super(MeshDiscriminator, self).__init__()

        self.fcs = None
        self.convs = []
        for i in range(len(convs) - 1):
            if i + 1 < len(pools):
                pool = pools[i + 1]
            else:
                pool = 0
            self.convs.append(DownConvFace(convs[i], convs[i + 1], blocks=blocks, pool=pool, symm_oper=symm_oper))
        self.global_pool = None
        if fcs is not None:
            self.fcs = []
            self.fcs_bn = []
            last_length = convs[-1]
            if global_pool is not None:
                if global_pool == 'max':
                    self.global_pool = nn.MaxPool1d(pools[-1])
                elif global_pool == 'avg':
                    self.global_pool = nn.AvgPool1d(pools[-1])
                else:
                    assert False, 'global_pool %s is not defined' % global_pool
            else:
                last_length *= pools[-1]
            if fcs[0] == last_length:
                fcs = fcs[1:]
            for length in fcs:
                self.fcs.append(nn.Linear(last_length, length))
                self.fcs_bn.append(nn.BatchNorm1d(length))
                last_length = length
            self.fcs = nn.ModuleList(self.fcs)
            self.fcs_bn = nn.ModuleList(self.fcs_bn)
        self.last_layer = nn.Sigmoid()
        self.convs = nn.ModuleList(self.convs)
        reset_params(self)

    def forward(self, x):
        fe, meshes = x
        for conv in self.convs:
            fe, before_pool = conv((fe, meshes))
        if self.fcs is not None:
            if self.global_pool is not None:
                fe = self.global_pool(fe)
            fe = fe.contiguous().view(fe.size()[0], -1)
            for i in range(len(self.fcs)):
                fe = self.fcs[i](fe)
                if self.fcs_bn:
                    x = fe.unsqueeze(1)
                    fe = self.fcs_bn[i](x).squeeze(1)
                if i < len(self.fcs) - 1:
                    fe = F.leaky_relu(fe, negative_slope=0.2)
        fe = self.last_layer(fe)
        # assert((fe.data.numpy()==0.5).all())
        return fe

    def __call__(self, x):
        return self.forward(x)


class MeshGenerator(nn.Module):
    def __init__(self, unrolls, convs, blocks=0, batch_norm=True, transfer_data=False, symm_oper=None):
        super(MeshGenerator, self).__init__()
        self.up_convs = []
        for i in range(len(convs) - 1):
            if i < len(unrolls):
                unroll = unrolls[i]
            else:
                unroll = 0
            self.up_convs.append(UpConvFace(convs[i], convs[i + 1], blocks=blocks, unroll=unroll,
                                            batch_norm=batch_norm, transfer_data=transfer_data, symm_oper=symm_oper))
        self.final_conv = UpConvFace(convs[-1], convs[-1], blocks=blocks, unroll=False,
                                     batch_norm=batch_norm, transfer_data=False, symm_oper=symm_oper, relu=False)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.final_activation = nn.Tanh()
        reset_params(self)

    def forward(self, x, encoder_outs=None):
        fe, meshes = x
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i + 2)]
            fe = up_conv((fe, meshes), before_pool)
        fe = self.final_conv((fe, meshes))
        fe = self.final_activation(fe)
        features = fe.data.numpy()
        out_features = []
        # make meshes.faces=fe, call build_mesh(meshes), extract_features(meshes) and return extracted features and generated meshes
        for i in range(len(meshes)):
            mesh = meshes[i]
            vt_values = np.swapaxes(features[i], 0, 1)
            vt_values = np.reshape(vt_values, [vt_values.shape[0], 3, 3])
            for v in range(mesh.vs.shape[0]):
                mesh.vs[v, :] = np.mean(vt_values[np.where(v == mesh.faces)], axis=0)
            meshes[i] = Mesh(faces=mesh.faces, vertices=mesh.vs, export_folder='generated')
            out_features.append(meshes[i].extract_features())
            out_features[i] = pad(out_features[i], mesh.faces.shape[0])

        fe = torch.from_numpy(np.asarray(out_features)).float()
        return fe, meshes

    def __call__(self, x, encoder_outs=None):
        return self.forward(x, encoder_outs)


class UpConvPoint(nn.Module):

    def __init__(self, in_channels, out_channels, blocks=0, unroll=0, residual=True,
                 batch_norm=True, transfer_data=True, symm_oper=None, relu=False, n_neighbors=6,
                 neighbor_order='random'):
        super(UpConvPoint, self).__init__()
        self.residual = residual
        self.bn = []
        self.unroll = None
        self.transfer_data = transfer_data
        self.relu = relu
        self.up_conv = MeshConvPoint(in_channels, out_channels, symm_oper=symm_oper, n_neighbors=n_neighbors,
                                     neighbor_order=neighbor_order)
        if transfer_data:
            self.conv1 = MeshConvPoint(2 * out_channels, out_channels, symm_oper=symm_oper, n_neighbors=n_neighbors,
                                       neighbor_order=neighbor_order)
        else:
            self.conv1 = MeshConvPoint(out_channels, out_channels, symm_oper=symm_oper, n_neighbors=n_neighbors,
                                       neighbor_order=neighbor_order)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(MeshConvPoint(out_channels, out_channels, symm_oper=symm_oper, n_neighbors=n_neighbors,
                                            neighbor_order=neighbor_order))
            self.conv2 = nn.ModuleList(self.conv2)
        if batch_norm:
            for _ in range(blocks + 1):
                self.bn.append(nn.InstanceNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        if unroll:
            self.unroll = MeshUnpool(unroll)

    def __call__(self, x, from_down=None):
        return self.forward(x, from_down)

    def forward(self, x, from_down):
        from_up, meshes = x
        x1 = self.up_conv(from_up, meshes).squeeze(3)
        if self.unroll:
            x1 = self.unroll(x1, meshes)
        if self.transfer_data:
            x1 = torch.cat((x1, from_down), 1)
        x1 = self.conv1(x1, meshes)
        if self.bn:
            x1 = self.bn[0](x1)
        if self.relu:
            x1 = F.relu(x1)
        # else:
        #     x1 = torch.tanh(x1)
        x2 = x1
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1, meshes)
            if self.bn:
                x2 = self.bn[idx + 1](x2)
            if self.residual:
                x2 = x2 + x1
            if self.relu:
                x2 = F.relu(x2)
            # else:
            #     x2 = torch.tanh(x2)
            x1 = x2
        x2 = x2.squeeze(3)
        return x2


class DownConvPoint(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=0, pool=0, symm_oper=None, relu=True, n_neighbors=6,
                 neighbor_order='random'):
        super(DownConvPoint, self).__init__()
        self.bn = []
        self.pool = None
        self.conv1 = MeshConvPoint(in_channels, out_channels, symm_oper=symm_oper, n_neighbors=n_neighbors,
                                   neighbor_order=neighbor_order)
        self.conv2 = []
        self.relu = relu
        for _ in range(blocks):
            self.conv2.append(MeshConvPoint(out_channels, out_channels, symm_oper=symm_oper, n_neighbors=n_neighbors,
                                            neighbor_order=neighbor_order))
            self.conv2 = nn.ModuleList(self.conv2)
        for _ in range(blocks + 1):
            self.bn.append(nn.InstanceNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        if pool:
            self.pool = MeshPoolPoint(pool)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        fe, meshes = x
        x1 = self.conv1(fe, meshes)
        if self.bn:
            x1 = self.bn[0](x1)
        if self.relu:
            x1 = F.relu(x1)
        else:
            x1 = torch.tanh(x1)
        x2 = x1
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1, meshes)
            if self.bn:
                x2 = self.bn[idx + 1](x2)
            x2 = x2 + x1
            if self.relu:
                x2 = F.relu(x2)
            else:
                x2 = torch.tanh(x2)
            x1 = x2
        x2 = x2.squeeze(3)
        before_pool = x2
        if self.pool:
            # before_pool = x2
            x2 = self.pool(x2, meshes)
        return x2, before_pool


class MeshPointGAN(nn.Module):
    """GAN Network that generates points (vertices)
    """

    def __init__(self, opt, conv_res, norm_layer, nf0, input_res, export_folder='generated'):
        super(MeshPointGAN, self).__init__()
        self.discriminator = MeshPointDiscriminator(opt.pool_res, conv_res, opt.fc_n, norm_layer, nf0, input_res,
                                                    nresblocks=opt.resblocks, symm_oper=opt.symm_oper)

        up_convs = conv_res[::].copy()
        up_convs.reverse()
        # self.generator = MeshPointGenerator(unpool_res, up_convs, norm_layer, 3, input_res, nresblocks=nresblocks,
        #                                     symm_oper=symm_oper, device=device, export_folder=export_folder, dilation=dilation)
        self.generator = MeshPointGenerator2(opt.unpool_res, up_convs, opt=opt)

    # unrolls, convs, blocks = 0, batch_norm = True, transfer_data = True
    # def forward(self, x, meshes):
    #     fe, before_pool = self.encoder((x, meshes))
    #     fe = self.decoder((fe, meshes), before_pool)
    #     return fe
    #
    # def __call__(self, x, meshes):
    #     return self.forward(x, meshes)


class MeshPointDiscriminator(nn.Module):
    def __init__(self, pool_res, conv_res, fc_n, norm_layer, nf0, input_res, nresblocks=3, symm_oper=1,
                 global_pool=None):
        super(MeshPointDiscriminator, self).__init__()
        self.k = [nf0] + conv_res
        self.res = [input_res] + pool_res
        norm_args = get_norm_args(norm_layer, self.k[1:])

        for i, ki in enumerate(self.k[:-1]):
            setattr(self, 'conv{}'.format(i), MResConvPoint(ki, self.k[i + 1], nresblocks, symm_oper=symm_oper))
            setattr(self, 'norm{}'.format(i), norm_layer(**norm_args[i]))
            setattr(self, 'pool{}'.format(i), MeshPoolPoint(self.res[i + 1]))

        self.gp = torch.nn.AvgPool1d(self.res[-1])
        # self.gp = torch.nn.MaxPool1d(self.res[-1])
        self.global_pool = None
        last_length = self.k[-1]
        if global_pool is not None:
            if global_pool == 'max':
                self.global_pool = nn.MaxPool1d(pool_res[-1])
            elif global_pool == 'avg':
                self.global_pool = nn.AvgPool1d(pool_res[-1])
            else:
                assert False, 'global_pool %s is not defined' % global_pool
        else:
            last_length *= pool_res[-1]
        self.fc1 = nn.Linear(last_length, fc_n)
        # self.fc2 = nn.Sequential(nn.Linear(fc_n, 1), nn.Sigmoid())
        self.fc2 = nn.Linear(fc_n, 1)

    def forward(self, input):
        x, mesh = input
        for i in range(len(self.k) - 1):
            x = getattr(self, 'conv{}'.format(i))(x, mesh)
            x = F.relu(getattr(self, 'norm{}'.format(i))(x))
            x = getattr(self, 'pool{}'.format(i))(x, mesh)

        if self.global_pool is not None:
            x = self.gp(x)
        x = x.contiguous().view(x.size()[0], -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MeshPointGenerator(nn.Module):
    def __init__(self, pool_res, conv_res, norm_layer, nf0, input_res, nresblocks=3, symm_oper=1, device=None,
                 export_folder='generated', dilation=True):
        super(MeshPointGenerator, self).__init__()
        self.k = [nf0] + conv_res
        self.res = pool_res
        self.device = device
        self.export_folder = export_folder
        self.dilation = dilation
        norm_args = get_norm_args(norm_layer, self.k[1:])

        for i, ki in enumerate(self.k[:-1]):
            setattr(self, 'conv{}'.format(i),
                    MResConvPoint(ki, self.k[i + 1], nresblocks, symm_oper=symm_oper, relu=False))
            setattr(self, 'norm{}'.format(i), norm_layer(**norm_args[i]))
            setattr(self, 'unpool{}'.format(i), MeshUnpoolPoint(self.res[i]))

        if self.dilation:
            self.final_conv = MResConvPoint(self.k[-1], 1, nresblocks, symm_oper=symm_oper, relu=False)
            self.final_activation = nn.Sigmoid()
        else:
            self.final_conv = MResConvPoint(self.k[-1], 3, nresblocks, symm_oper=symm_oper, relu=False)
            self.final_activation = nn.Tanh()

    def forward(self, input):
        x, mesh = input
        for i in range(len(self.k) - 1):
            x = getattr(self, 'conv{}'.format(i))(x, mesh)
            if self.dilation:
                x = F.relu(getattr(self, 'norm{}'.format(i))(x))
            else:
                x = F.leaky_relu(getattr(self, 'norm{}'.format(i))(x), negative_slope=0.2)
            x = getattr(self, 'unpool{}'.format(i))(x, mesh)

        x = self.final_conv(x, mesh)
        x = self.final_activation(x)

        out_features = []
        gen_output = []
        for i in range(len(mesh)):
            gen_output.append(np.transpose(x.cpu().data.numpy()[i, :, :, 0]))

            # print(np.transpose(gen_output))
            if self.dilation:
                gen_vertices = mesh[i].vs * gen_output[i]
            else:
                gen_vertices = gen_output[i]
            mesh[i] = Mesh(faces=mesh[i].faces, vertices=gen_vertices, export_folder=self.export_folder)
            out_features.append(mesh[i].extract_features())
            out_features[i] = pad(out_features[i], mesh[i].faces.shape[0])

        wandb.log({'gen_output': np.asarray(gen_output)[:, :, 0]})
        fe = torch.from_numpy(np.asarray(out_features)).float().to(x.device)

        return fe, mesh


class MeshPointGenerator2(nn.Module):
    def __init__(self, unrolls, convs, blocks=0, batch_norm=True, transfer_data=False, device=None,
                 export_folder='generated', opt=None):
        super(MeshPointGenerator2, self).__init__()
        self.device = device
        self.export_folder = export_folder
        self.opt = opt
        self.up_convs = []
        convs.insert(0, 4)
        if opt.dilation:
            convs.append(1)
        else:
            convs.append(3)
        for i in range(len(convs) - 2):
            if i < len(unrolls):
                unroll = unrolls[i]
            else:
                unroll = 0
            self.up_convs.append(UpConvPoint(convs[i], convs[i + 1], blocks=blocks, unroll=unroll,
                                             batch_norm=batch_norm, transfer_data=transfer_data,
                                             symm_oper=opt.symm_oper))
        self.final_conv = UpConvPoint(convs[-2], convs[-1], blocks=blocks, unroll=False,
                                      batch_norm=batch_norm, transfer_data=False, symm_oper=opt.symm_oper)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.final_activation = nn.Tanh()

        reset_params(self)

    def forward(self, input, encoder_outs=None):
        x, meshes = input
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i + 2)]
            x = up_conv((x, meshes), before_pool)
        x = self.final_conv((x, meshes))
        x = self.final_activation(x)

        out_features = []
        gen_output = []
        for i in range(len(meshes)):
            gen_output.append(np.transpose(x.cpu().data.numpy()[i, :, :]))

            # print(np.transpose(gen_output))
            if self.opt.dilation:
                gen_vertices = meshes[i].vs * gen_output[i]
            else:
                gen_vertices = gen_output[i]
            meshes[i] = Mesh(faces=meshes[i].faces, vertices=gen_vertices, export_folder='', opt=self.opt)
            out_features.append(meshes[i].extract_features())
            out_features[i] = pad(out_features[i], meshes[i].vs.shape[0])

        wandb.log({'gen_output': np.asarray(gen_output)[:, :, 0]})
        fe = torch.from_numpy(np.asarray(out_features)).float().to(x.device)

        return fe, meshes

    def __call__(self, x, encoder_outs=None):
        return self.forward(x, encoder_outs)


class MeshVAE(nn.Module):
    """Autoencoder Network for generative learning
    """

    def __init__(self, pools, down_convs, up_convs, z_dim, blocks=0, transfer_data=False, symm_oper=1, opt=None):
        super(MeshVAE, self).__init__()
        self.transfer_data = transfer_data
        self.encoder = MeshEncoderPoint(pools, down_convs, z_dim, blocks=blocks, symm_oper=symm_oper, variational=True,
                                        opt=opt)
        unrolls = pools[:-1].copy()
        unrolls.reverse()
        self.decoder = MeshDecoderPoint(unrolls, up_convs, blocks=blocks, transfer_data=transfer_data,
                                        symm_oper=symm_oper, opt=opt)
        self.fc = nn.Linear(z_dim, z_dim)
        self.opt = opt

    def reparameterize(self, mu, logvar):
        if self.opt.is_train:
            std = logvar.mul(0.5).exp_()
            # return torch.normal(mu, std)
            esp = torch.randn(*mu.size()).to(mu.device)
            z = mu + std * esp
        else:
            z = mu
        return z

    def forward(self, x, meshes):
        fe, mu, lvar = self.encode(x, meshes)
        fe = self.decode(fe, meshes)
        return fe, mu, lvar

    def encode(self, x, meshes):
        mu, lvar, before_pool = self.encoder((x, meshes))
        if self.opt.pool_res != []:
            for i, m in enumerate(meshes):
                # TODO: Optimize current process (mesh -> pool with edges and verts -> mesh with faces -> mesh with gemm)
                m.build_faces()
                meshes[i] = Mesh(faces=m.faces, vertices=m.vs, export_folder='', opt=self.opt)
        fe = self.reparameterize(mu, lvar)
        return fe, mu, lvar

    def decode(self, z, meshes):
        fe = self.fc(z)
        fe = self.decoder((fe, meshes))
        return fe

    # Mode is one of 'encode', 'decode', 'autoencode'
    def __call__(self, x, meshes, mode='autoencode'):

        if mode == 'autoencode':
            return self.forward(x, meshes)
        elif mode == 'decode':
            return self.decode(x, meshes)
        elif mode == 'encode':
            return self.encode(x, meshes)
        else:
            raise ValueError(mode, 'Wrong value in vae mode')


class MeshAutoencoder(nn.Module):
    """Autoencoder Network for generative learning
    """

    def __init__(self, pools, down_convs, up_convs, z_dim, blocks=0, transfer_data=True, symm_oper=1, opt=None):
        super(MeshAutoencoder, self).__init__()
        self.transfer_data = transfer_data
        self.encoder = MeshEncoderPoint(pools, down_convs, z_dim, blocks=blocks, symm_oper=symm_oper, opt=opt)
        unrolls = pools[:-1].copy()
        unrolls.reverse()
        self.decoder = MeshDecoderPoint(unrolls, up_convs, blocks=blocks, transfer_data=transfer_data,
                                        symm_oper=symm_oper, opt=opt)
        self.opt = opt

    def forward(self, x, meshes):
        fe, before_pool = self.encoder((x, meshes))
        fe = self.decoder((fe, meshes), before_pool)
        return fe

    def encode(self, x, meshes):
        fe, before_pool = self.encoder((x, meshes))
        if self.opt.pool_res != []:
            for i, m in enumerate(meshes):
                # TODO: Optimize current process (mesh -> pool with edges and verts -> mesh with faces -> mesh with gemm)
                m.build_faces()
                meshes[i] = Mesh(faces=m.faces, vertices=m.vs, export_folder='', opt=self.opt)
        return fe

    def decode(self, z, meshes):
        return self.decoder((z, meshes))

    # Mode is one of 'encode', 'decode', 'autoencode'
    def __call__(self, x, meshes, mode='autoencode'):

        if mode == 'autoencode':
            return self.forward(x, meshes)
        elif mode == 'decode':
            return self.decode(x, meshes)
        elif mode == 'encode':
            return self.encode(x, meshes)
        else:
            raise ValueError(mode, 'Wrong value in vae mode')


class MeshEncoderPoint(nn.Module):
    def __init__(self, pools, convs, z_dim, fcs=None, blocks=0, global_pool=None, symm_oper=None, variational=False,
                 opt=None):
        super(MeshEncoderPoint, self).__init__()
        self.fcs = None
        self.convs = []
        self.variational = variational
        self.opt = opt
        for i in range(len(convs) - 1):
            if i + 1 < len(pools):
                pool = pools[i + 1]
            else:
                pool = 0
            self.convs.append(DownConvPoint(convs[i], convs[i + 1], blocks=blocks, pool=pool, symm_oper=symm_oper,
                                            n_neighbors=opt.n_neighbors, neighbor_order=opt.neighbor_order))
        self.global_pool = None

        last_length = convs[-1]
        if global_pool is not None:
            if global_pool == 'max':
                self.global_pool = nn.MaxPool1d(pools[-1])
            elif global_pool == 'avg':
                self.global_pool = nn.AvgPool1d(pools[-1])
            else:
                assert False, 'global_pool %s is not defined' % global_pool
        else:
            last_length *= pools[-1]

        if fcs is not None:
            self.fcs = []
            self.fcs_bn = []
            if fcs[0] == last_length:
                fcs = fcs[1:]
            for length in fcs:
                self.fcs.append(nn.Linear(last_length, length))
                self.fcs_bn.append(nn.InstanceNorm1d(length))
                last_length = length
            self.fcs = nn.ModuleList(self.fcs)
            self.fcs_bn = nn.ModuleList(self.fcs_bn)

        self.last_fc = nn.Linear(last_length, z_dim)
        self.last_bn = nn.InstanceNorm1d(z_dim)
        if self.variational:
            self.last_fc_2 = nn.Linear(last_length, z_dim)
            self.last_bn_2 = nn.InstanceNorm1d(z_dim)

        self.convs = nn.ModuleList(self.convs)
        reset_params(self)

    def forward(self, x):
        fe, meshes = x
        encoder_outs = []
        for conv in self.convs:
            fe, before_pool = conv((fe, meshes))
            encoder_outs.append(before_pool)

        if self.global_pool is not None:
            fe = self.global_pool(fe)
        fe = fe.contiguous().view(fe.size()[0], -1)

        if self.fcs is not None:
            for i in range(len(self.fcs)):
                fe = self.fcs[i](fe)
                if self.fcs_bn:
                    x = fe.unsqueeze(1)
                    fe = self.fcs_bn[i](x).squeeze(1)
                if i < len(self.fcs) - 1:
                    fe = F.relu(fe)
        if self.variational:
            fe_2 = self.last_fc_2(fe)
            x = fe_2.unsqueeze(1)
            fe_2 = self.last_bn_2(x).squeeze(1)
            fe_2 = fe_2.unsqueeze(1)

        fe = self.last_fc(fe)
        x = fe.unsqueeze(1)
        fe = self.last_bn(x).squeeze(1)
        fe = fe.unsqueeze(1)

        if self.variational:
            return fe, fe_2, encoder_outs
        else:
            return fe, encoder_outs

    def __call__(self, x):
        return self.forward(x)


class MeshDecoderPoint(nn.Module):
    def __init__(self, unrolls, convs, blocks=0, batch_norm=True, transfer_data=True, symm_oper=None, opt=None):
        super(MeshDecoderPoint, self).__init__()
        self.up_convs = []
        self.transfer_data = transfer_data
        for i in range(len(convs) - 2):
            if i < len(unrolls):
                unroll = unrolls[i]
            else:
                unroll = 0
            self.up_convs.append(UpConvPoint(convs[i], convs[i + 1], blocks=blocks, unroll=unroll,
                                             batch_norm=batch_norm, transfer_data=transfer_data, symm_oper=symm_oper,
                                             relu=True, n_neighbors=opt.n_neighbors, neighbor_order=opt.neighbor_order))
        self.final_conv = UpConvPoint(convs[-2], convs[-1], blocks=blocks, unroll=False,
                                      batch_norm=batch_norm, transfer_data=False, symm_oper=symm_oper,
                                      n_neighbors=opt.n_neighbors, neighbor_order=opt.neighbor_order)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.final_activation = nn.Tanh()
        reset_params(self)

    def forward(self, x, encoder_outs=None):
        fe, meshes = x
        fe = fe.reshape((fe.shape[0], 3, -1))
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if self.transfer_data and encoder_outs is not None:
                before_pool = encoder_outs[-(i + 2)]
            fe = up_conv((fe, meshes), before_pool)
        fe = self.final_conv((fe, meshes))
        fe = self.final_activation(fe)
        return fe

    def __call__(self, x, encoder_outs=None):
        return self.forward(x, encoder_outs)
