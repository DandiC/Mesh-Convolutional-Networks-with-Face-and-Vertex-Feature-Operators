import torch
from . import networks
from os.path import join
from util.util import seg_accuracy, print_network, batch_point_to_point
import wandb
from .networks import MeshAutoencoder, MeshVAE
import copy
import numpy as np
from models.layers.mesh import Mesh
from .networks import init_net
import os
import torch.nn.functional as F
import neuralnet_pytorch as nnt


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
        self.is_train = opt.phase == 'train'
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
        down_convs = [5] + opt.ncf
        up_convs = [3] + opt.ncf[::-1] + [3]

        pool_res = [opt.ninput_features] + opt.pool_res
        if self.opt.vae:
            self.net = init_net(MeshVAE(pool_res, down_convs, up_convs, opt.ninput_features*3, blocks=0,
                                        transfer_data=opt.skip_connections,
                                        symm_oper=opt.symm_oper, opt=opt), opt.init_type, opt.init_gain, self.gpu_ids,
                                generative=False)
        else:
            self.net = init_net(
                MeshAutoencoder(pool_res, down_convs, up_convs, opt.ninput_features*3, blocks=0, transfer_data=opt.skip_connections,
                                symm_oper=opt.symm_oper, opt=opt), opt.init_type, opt.init_gain, self.gpu_ids,
                generative=False)

        self.net.train(self.is_train)
        self.criterion = networks.define_loss(opt).to(self.device)

        export_folder = os.path.join(opt.checkpoints_dir, opt.name, 'generated')
        if not os.path.exists(export_folder):
            os.makedirs(export_folder)

        self.results_folder = os.path.join(opt.checkpoints_dir, opt.name, 'results')
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

        self.sample_folder = os.path.join(opt.checkpoints_dir, opt.name, 'sample')
        if not os.path.exists(self.sample_folder):
            os.makedirs(self.sample_folder)

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = networks.get_scheduler(self.optimizer, opt)
            num_params = print_network(self.net)
            wandb.log({"Params": num_params})

        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch)

    def loss_vae(self, recon_x, x, mu, logvar):
        # BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
        # TODO: Avoid comparing dummy features when computing loss
        if self.opt.loss == 'mse':
            BCE = self.criterion(recon_x, x)
        elif self.opt.loss == 'chamfer':
            BCE = nnt.metrics.chamfer_loss(recon_x.permute(0,2,1), x.permute(0,2,1), reduce='mean')
        elif self.opt.loss == 'ptp':
            BCE = batch_point_to_point(recon_x,self.mesh, x)
        else:
            raise ValueError(self.opt.loss, 'Wrong parameter value in --loss')
        # BCE = chamfer_distance(recon_x, x)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD, BCE, KLD

    def set_input(self, data):
        input_features = torch.from_numpy(data['features']).float()
        labels = torch.from_numpy(data['coordinates']).float()
        # set inputs
        self.features = input_features.to(self.device).requires_grad_(self.is_train)
        self.labels = labels.to(self.device)
        self.mesh = data['mesh']
        if self.opt.dataset_mode == 'segmentation' and not self.is_train:
            self.soft_label = torch.from_numpy(data['soft_label'])

    def forward(self):
        out = self.net(self.features, self.mesh)
        return out

    def backward(self, out):
        if self.opt.vae:
            self.loss, bce, kld = self.loss_vae(out[0], self.labels, out[1], out[2])
        else:
            self.loss = self.criterion(out, self.labels)
        self.loss.backward()

    def optimize_parameters(self, epoch=0):
        # Compute loss and backpropagate
        self.optimizer.zero_grad()
        out = self.forward()
        self.backward(out)
        self.optimizer.step()

        # Generate models
        self.gen_models = copy.deepcopy(self.mesh)
        if self.opt.vae:
            vs_out = out[0].cpu().data.numpy()
        else:
            vs_out = out.cpu().data.numpy()
        for i in range(self.gen_models.shape[0]):
            self.gen_models[i] = Mesh(faces=self.gen_models[i].faces, vertices=np.transpose(vs_out[i]),
                                      export_folder='',
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

    def generate(self, z):
        with torch.no_grad():
            latent_mesh = Mesh(self.opt.latent_path, opt=self.opt)
            out = self.net(z, [latent_mesh], mode='decode')
            return Mesh(faces=latent_mesh.faces, vertices=np.transpose(out[0].cpu().data.numpy()), export_folder='',
                        opt=self.opt)

    def encode(self, mesh):
        with torch.no_grad():
            x = torch.tensor(mesh.features).to(self.device).float()
            if self.opt.vae:
                fe, mu, lvar = self.net(x.unsqueeze(0), [mesh], mode='encode')
            else:
                fe = self.net(x.unsqueeze(0), [mesh], mode='encode')
            return fe

    def test(self):
        """tests model
        returns: MSE of each reconstructed mesh
        """
        with torch.no_grad():
            out = self.forward()
            self.gen_models = copy.deepcopy(self.mesh)
            if self.opt.vae:
                vs_out = out[0].cpu().data.numpy()
            else:
                vs_out = out.cpu().data.numpy()
            rmse = np.sqrt(np.mean((np.reshape(vs_out,(vs_out.shape[0],-1))-np.reshape(self.labels.data.cpu().numpy(),(vs_out.shape[0],-1)))**2, axis=1))
            chamfer = []
            emd = []
            for i in range(self.gen_models.shape[0]):
                self.gen_models[i] = Mesh(faces=self.gen_models[i].faces, vertices=np.transpose(vs_out[i]),
                                          export_folder='',
                                          opt=self.opt)
                export_file = os.path.join(self.results_folder, self.mesh[i].filename)
                self.gen_models[i].export(file=export_file)
                original_samples, _ = self.mesh[i].sample(self.opt.sample_points)
                generated_samples, _ = self.gen_models[i].sample(self.opt.sample_points)
                chamfer.append(nnt.metrics.chamfer_loss(original_samples, generated_samples, reduce='mean').data.cpu().numpy())
                emd.append(nnt.metrics.emd_loss(original_samples, generated_samples, reduce='mean').data.cpu().numpy())

            return rmse, np.asarray(chamfer), np.asarray(emd)

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
