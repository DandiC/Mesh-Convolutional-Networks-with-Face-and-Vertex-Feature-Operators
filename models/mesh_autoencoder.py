import torch
from . import networks
from os.path import join
from util.util import seg_accuracy, print_network
import wandb
from .networks import MeshAutoencoder
import copy
import numpy as np
from models.layers.mesh import Mesh
import os

class AutoencoderModel:
    """ Class for training Model weights

    :args opt: structure containing configuration params
    e.g.,
    --dataset_mode -> generative
    --arch -> network type (meshunet)
    """
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.features = None
        self.labels = None
        self.mesh = None
        self.soft_label = None
        self.loss = None

        #
        self.nclasses = opt.nclasses

        # load/define networks
        down_convs = [3] + opt.ncf + [1]
        up_convs = [1] + opt.ncf[::-1] + [3]
        pool_res = [opt.ninput_features] + opt.pool_res
        self.net = MeshAutoencoder(pool_res, down_convs, up_convs, blocks=0, transfer_data=True, symm_oper=opt.symm_oper)

        self.net.train(self.is_train)
        self.criterion = networks.define_loss(opt).to(self.device)

        export_folder = os.path.join(opt.checkpoints_dir, opt.name, 'generated')
        if not os.path.exists(export_folder):
            os.makedirs(export_folder)

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = networks.get_scheduler(self.optimizer, opt)
            num_params = print_network(self.net)
            wandb.log({"Params": num_params})

        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch)

    def set_input(self, data):
        input_features = torch.from_numpy(data['features']).float()
        labels = torch.from_numpy(data['label']).long()
        # set inputs
        self.features = input_features.to(self.device).requires_grad_(self.is_train)
        self.labels = input_features.to(self.device)
        self.mesh = data['mesh']
        if self.opt.dataset_mode == 'segmentation' and not self.is_train:
            self.soft_label = torch.from_numpy(data['soft_label'])


    def forward(self):
        out = self.net(self.features, self.mesh)
        return out

    def backward(self, out):
        self.loss = self.criterion(out, self.labels)
        self.loss.backward()

    def optimize_parameters(self, epoch=0):
        self.optimizer.zero_grad()
        out = self.forward()
        self.backward(out)
        self.optimizer.step()
        self.gen_models = copy.deepcopy(self.mesh)
        vs_out = out.cpu().data.numpy()
        for i in range(self.gen_models.shape[0]):
            self.gen_models[i] = Mesh(faces=self.gen_models[i].faces, vertices=np.transpose(vs_out[i]), export_folder='generated',
                             opt=self.opt)

##################

    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        # PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)


    def save_network(self, which_epoch, wandb_save=False, dataset_mode=None):
        """save model to disk"""
        save_filename = '%s_net.pth' % (which_epoch)
        save_path = join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

        if wandb_save:
            wandb.save(save_path)

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def test(self):
        """tests model
        returns: number correct and total number
        """
        with torch.no_grad():
            out = self.forward()
            # compute number of correct
            pred_class = out.data.max(1)[1]
            label_class = self.labels
            self.export_segmentation(pred_class.cpu())
            correct = self.get_accuracy(pred_class, label_class)
        return correct, len(label_class)

    def get_accuracy(self, pred, labels):
        """computes accuracy for classification / segmentation """
        if self.opt.dataset_mode == 'classification':
            correct = pred.eq(labels).sum()
        elif self.opt.dataset_mode == 'segmentation':
            correct = seg_accuracy(pred, self.soft_label, self.mesh)
        return correct

    def export_segmentation(self, pred_seg):
        if self.opt.dataset_mode == 'segmentation':
            for meshi, mesh in enumerate(self.mesh):
                mesh.export_segments(pred_seg[meshi, :])
