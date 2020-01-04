import torch
from . import networks
from os.path import join
from util.util import seg_accuracy, print_network
import wandb
from models.layers.mesh import Mesh
import numpy as np
from torch.autograd import Variable
import copy


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class GenerativeModel:
    """ Class for training Model weights

    :args opt: structure containing configuration params
    e.g.,
    --dataset_mode -> generative)
    --arch -> network type (GAN)
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
        self.disc_accuracy = 0
        self.latent_path = opt.latent_path
        #
        self.nclasses = opt.nclasses

        # load/define networks
        self.net = networks.define_classifier(opt.input_nc, opt.ncf, opt.ninput_features, opt.nclasses, opt,
                                              self.gpu_ids, opt.arch, opt.init_type, opt.init_gain, opt.feat_from, device=self.device)
        self.net.generator.train(self.is_train)
        self.net.discriminator.train(self.is_train)
        self.net.discriminator.apply(weights_init_normal)
        self.net.generator.apply(weights_init_normal)

        self.criterion_disc = networks.define_loss(opt)[0].to(self.device)
        self.criterion_gen = networks.define_loss(opt)[1].to(self.device)

        if self.is_train:
            self.optimizer_D = torch.optim.Adam(self.net.discriminator.parameters(), lr=opt.lr_disc,
                                                betas=(opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(self.net.generator.parameters(), lr=opt.lr_gen,
                                                betas=(opt.beta1, 0.999))
            # self.scheduler = networks.get_scheduler(self.optimizer, opt)
            print("DISCRIMINATOR:")
            disc_params = print_network(self.net.discriminator)
            print("GENERATOR:")
            gen_params = print_network(self.net.generator)
            wandb.log({"Gen Params": gen_params, "Disc Params": disc_params, "Params": disc_params+gen_params})


        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch)

    def set_input(self, data):
        input_features = torch.from_numpy(data['features']).float()
        labels = torch.from_numpy(data['label']).long()
        # set inputs
        self.features = input_features.to(self.device).requires_grad_(self.is_train)
        self.labels = labels.to(self.device)
        self.mesh = data['mesh']

        self.valid = Variable(torch.FloatTensor(self.features.shape[0], 1).fill_(1.0), requires_grad=False).to(self.device)
        self.fake = Variable(torch.FloatTensor(self.features.shape[0], 1).fill_(0.0), requires_grad=False).to(self.device)
        self.label = np.concatenate([self.valid.data.cpu().numpy(), self.fake.data.cpu().numpy()], axis=0)

    def forward(self):
        out = self.net(self.features, self.mesh)
        return out

    def backward(self, out):
        self.loss = self.criterion(out, self.labels)
        self.loss.backward()

    def optimize_parameters(self):
        for i in range(self.opt.gen_steps):
            self.trainGenerator()

        for i in range(self.opt.disc_steps):
            self.trainDiscriminator()
        del self.valid
        del self.fake
        del self.label

    def trainGenerator(self):
        self.optimizer_G.zero_grad()
        #     Fake initial data
        latent_mesh = Mesh(self.latent_path, opt=self.opt)
        self.fake_mesh = np.asarray([copy.deepcopy(latent_mesh) for i in range(self.features.shape[0])])

        if 'Point' in self.opt.arch:
            self.fake_features = torch.rand(self.opt.batch_size, 1, self.fake_mesh[0].vs_count).to(
                self.device).requires_grad_(self.is_train)
        else:
            self.fake_features = torch.rand(self.opt.batch_size, 1, self.fake_mesh[0].face_count).to(
                self.device).requires_grad_(self.is_train)
        # self.gen_features, self.gen_models = self.net.generator((self.fake_features, self.fake_mesh))
        self.gen_features, self.gen_models = self.features, self.mesh
        self.gen_features = self.gen_features.to(self.device).requires_grad_(self.is_train)
        self.g_loss = self.criterion_gen(self.net.discriminator((self.gen_features,self.gen_models)), self.valid)
        # self.g_loss.backward()
        # self.optimizer_G.step()
        del self.fake_mesh

    def trainDiscriminator(self):
        self.optimizer_D.zero_grad()
        output_disc_real = self.net.discriminator((self.features, self.mesh))
        output_disc_fake = self.net.discriminator((self.gen_features, self.gen_models))
        pred = np.concatenate([output_disc_real.data.cpu().numpy(), output_disc_fake.data.cpu().numpy()], axis=0)
        self.disc_accuracy = np.mean(np.round(pred) == self.label)
        self.mean_output_disc_real = torch.mean(output_disc_real).tolist()
        self.mean_output_disc_fake = torch.mean(output_disc_fake).tolist()
        real_loss = self.criterion_disc(output_disc_real, self.valid)
        fake_loss = self.criterion_disc(output_disc_fake, self.fake)
        self.d_loss = (real_loss+fake_loss)/2
        if self.disc_accuracy < 0.8:
            self.d_loss.backward()
            self.optimizer_D.step()

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
            if dataset_mode != 'generative':
                torch.save(self.net.module.cpu().state_dict(), save_path)
                self.net.cuda(self.gpu_ids[0])
            else:
                torch.save(self.net.generator.module.cpu().state_dict(), save_path.replace('.pth', '_gen.pth'))
                self.net.generator.cuda(self.gpu_ids[0])
                torch.save(self.net.discriminator.module.cpu().state_dict(), save_path.replace('.pth', '_disc.pth'))
                self.net.discriminator.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

        if wandb_save:
            if dataset_mode != 'generative':
                wandb.save(save_path)
            else:
                wandb.save(save_path.replace('.pth', '_disc.pth'))
                wandb.save(save_path.replace('.pth', '_gen.pth'))

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
