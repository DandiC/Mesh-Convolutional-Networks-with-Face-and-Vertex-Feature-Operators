import argparse
import os
from util import util
import torch


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # PARAMETERS FOR CLASSIFICATION
        # data params
        # self.parser.add_argument('--dataroot', default='datasets/shrec_6', help='path to meshes (should have subfolders train, test)')
        # self.parser.add_argument('--dataset_mode', choices={"classification", "segmentation", "generative"}, default='classification')
        # self.parser.add_argument('--ninput_features', type=int, default=252, help='# of input features (will include dummy features)')
        # # network params
        # self.parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        # self.parser.add_argument('--arch', type=str, default='meshPointGAN',
        #                          choices={"mconvnet", "meshunet", "meshGAN", "meshPointGAN"},
        #                          help='selects network to use')
        # self.parser.add_argument('--resblocks', type=int, default=1, help='# of res blocks')
        # self.parser.add_argument('--fc_n', type=int, default=100, help='# between fc and nclasses') #todo make generic
        # self.parser.add_argument('--ncf', nargs='+', default=[128, 256, 256, 512], type=int, help='conv filters')
        # self.parser.add_argument('--pool_res', nargs='+', default=[202,152,102,62], type=int, help='pooling res')
        # self.parser.add_argument('--norm', type=str, default='batch',help='instance normalization or batch normalization or group normalization')
        # self.parser.add_argument('--num_groups', type=int, default=16, help='# of groups for groupnorm')
        # self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        # self.parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        # self.parser.add_argument('--face_pool', type=str, default='v2', help='Version of face pool. For tracking purposes only.')
        # self.parser.add_argument('--symm_oper', nargs='+', default=[1], type=int, help='pooling res')
        # self.parser.add_argument('--vertex_features', nargs='+', type=str, default=['mean_c', 'gaussian_c'],
        #                          help='Type of vertex features to be used (for vertex convolution). Options are coord (for coordinates), norm (for normals), mean_c (for mean curvature) and gauss_c (for gaussian curvature)')
        # self.parser.add_argument('--n_neighbors', default=6, type=int, help='Number of neighbors selected for point (vertex) convolution. If set to -1, network does the average of all vertices')
        # self.parser.add_argument('--gen_steps', type=int, default=1, help='# of training steps for the generator')
        # self.parser.add_argument('--disc_steps', type=int, default=1, help='# of training steps for the discriminator')
        # self.parser.add_argument('--dilation',  default=False, action='store_true',
        #                          help='Determines if the generator outputs dilation (true) or vertex positions (false)')
        # # general params
        # self.parser.add_argument('--feat_from', type=str, default='point', help='Primitive to extract features from. One of: edge, face, point')
        # self.parser.add_argument('--num_threads', default=3, type=int, help='# threads for loading data')
        # self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        # self.parser.add_argument('--name', type=str, default='shrec_point', help='name of the experiment. It decides where to store samples and models')
        # self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes meshes in order, otherwise takes them randomly')
        # self.parser.add_argument('--seed', type=int, help='if specified, uses seed')
        # self.parser.add_argument('--clean_data',  default=False, action='store_true',
        #                          help='If true, it deletes the cache NPZ files in the dataset before training.')
        # # visualization params
        # self.parser.add_argument('--export_folder', type=str, default='', help='exports intermediate collapses to this folder')

        # PARAMETERS FOR SEGMENTATION
        # data params
        # self.parser.add_argument('--dataroot', default='datasets/coseg_aliens',
        #                          help='path to meshes (should have subfolders train, test)')
        # self.parser.add_argument('--dataset_mode', choices={"classification", "segmentation", "generative"}, default='segmentation')
        # self.parser.add_argument('--ninput_features', type=int, default=2280,
        #                          help='# of input features (will include dummy features)')
        # # network params
        # self.parser.add_argument('--batch_size', type=int, default=12, help='input batch size')
        # self.parser.add_argument('--arch', type=str, default='meshunet',  choices={"mconvnet", "meshunet", "meshGAN", "meshPointGAN"},
        #                          help='selects network to use')
        # self.parser.add_argument('--resblocks', type=int, default=3, help='# of res blocks')
        # self.parser.add_argument('--fc_n', type=int, default=100, help='# between fc and nclasses')  # todo make generic
        # self.parser.add_argument('--ncf', nargs='+', default=[32, 64, 128, 256], type=int, help='conv filters')
        # self.parser.add_argument('--pool_res', nargs='+', default=[1800, 1350, 600], type=int, help='pooling res')
        # self.parser.add_argument('--norm', type=str, default='batch',
        #                          help='instance normalization or batch normalization or group normalization')
        # self.parser.add_argument('--num_groups', type=int, default=16, help='# of groups for groupnorm')
        # self.parser.add_argument('--init_type', type=str, default='normal',
        #                          help='network initialization [normal|xavier|kaiming|orthogonal]')
        # self.parser.add_argument('--init_gain', type=float, default=0.02,
        #                          help='scaling factor for normal, xavier and orthogonal.')
        # self.parser.add_argument('--face_pool', type=str, default='v2',
        #                          help='Version of face pool. For tracking purposes only.')
        # self.parser.add_argument('--symm_oper', nargs='+', default=[1], type=int, help='pooling res')
        # self.parser.add_argument('--vertex_features', nargs='+', type=str, default=['mean_c', 'gaussian_c'],
        #                          help='Type of vertex features to be used (for vertex convolution). Options are coord (for coordinates), norm (for normals), mean_c (for mean curvature) and gauss_c (for gaussian curvature)')
        # self.parser.add_argument('--gen_steps', type=int, default=1, help='# of training steps for the generator')
        # self.parser.add_argument('--disc_steps', type=int, default=1, help='# of training steps for the discriminator')
        # # general params
        # self.parser.add_argument('--feat_from', type=str, default='edge',
        #                          help='Primitive to extract features from. One of: edge, face')
        # self.parser.add_argument('--num_threads', default=3, type=int, help='# threads for loading data')
        # self.parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        # self.parser.add_argument('--name', type=str, default='coseg',
        #                          help='name of the experiment. It decides where to store samples and models')
        # self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # self.parser.add_argument('--serial_batches', action='store_true',
        #                          help='if true, takes meshes in order, otherwise takes them randomly')
        # self.parser.add_argument('--seed', type=int, help='if specified, uses seed')
        # self.parser.add_argument('--clean_data',  default=False, action='store_true',
        #                          help='If true, it deletes the cache NPZ files in the dataset before training.')
        # # visualization params
        # self.parser.add_argument('--export_folder', type=str, default='',
        #                          help='exports intermediate collapses to this folder')
        # self.initialized = True
        #
        # # PARAMETERS FOR GENERATIVE LEARNING
        # # data params
        # self.parser.add_argument('--dataroot', default='datasets/shrec_bird',
        #                          help='path to meshes (should have subfolders train, test)')
        # self.parser.add_argument('--dataset_mode', choices={"classification", "segmentation", "generative"},
        #                          default='generative')
        # self.parser.add_argument('--ninput_features', type=int, default=500,
        #                          help='# of input features (will include dummy features)')
        # self.parser.add_argument('--latent_path', default='datasets/latent/sphere.obj', help='Path to the OBJ containing the latent for connectivity')
        # # network params
        # self.parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        # self.parser.add_argument('--arch', type=str, default='meshPointGAN',
        #                          choices={"mconvnet", "meshunet", "meshGAN", "meshPointGAN"},
        #                          help='selects network to use')
        # self.parser.add_argument('--resblocks', type=int, default=0, help='# of res blocks')
        # self.parser.add_argument('--fc_n', type=int, default=100, help='# between fc and nclasses')  # todo make generic
        # self.parser.add_argument('--ncf', nargs='+', default=[32, 64, 128, 256], type=int, help='conv filters')
        # self.parser.add_argument('--pool_res', nargs='+', default=[400,300,200,120], type=int, help='pooling res')
        # self.parser.add_argument('--unpool_res', nargs='+', default=[102,152,202,252], type=int, help='unpooling res (for MeshPointGAN)')
        # self.parser.add_argument('--norm', type=str, default='batch',
        #                          help='instance normalization or batch normalization or group normalization')
        # self.parser.add_argument('--num_groups', type=int, default=16, help='# of groups for groupnorm')
        # self.parser.add_argument('--init_type', type=str, default='normal',
        #                          help='network initialization [normal|xavier|kaiming|orthogonal]')
        # self.parser.add_argument('--init_gain', type=float, default=0.02,
        #                          help='scaling factor for normal, xavier and orthogonal.')
        # self.parser.add_argument('--face_pool', type=str, default='v2',
        #                          help='Version of face pool. For tracking purposes only.')
        # self.parser.add_argument('--symm_oper', nargs='+', default=[1], type=int, help='pooling res')
        # self.parser.add_argument('--vertex_features', nargs='+', type=str, default=['mean_c', 'gaussian_c'],
        #                          help='Type of vertex features to be used (for vertex convolution). Options are coord (for coordinates), norm (for normals), mean_c (for mean curvature) and gauss_c (for gaussian curvature)')
        # self.parser.add_argument('--gen_steps', type=int, default=1, help='# of training steps for the generator')
        # self.parser.add_argument('--disc_steps', type=int, default=1, help='# of training steps for the discriminator')
        # self.parser.add_argument('--max_disc_acc', type=float, default=0.8, help='Maximum accuracy for the discriminator')
        # self.parser.add_argument('--dilation',  default=False, action='store_true',
        #                          help='Determines if the generator outputs dilation (true) or vertex positions (false)')
        # # general params
        # self.parser.add_argument('--feat_from', type=str, default='face',
        #                          help='Primitive to extract features from. One of: edge, face')
        # self.parser.add_argument('--num_threads', default=3, type=int, help='# threads for loading data')
        # self.parser.add_argument('--gpu_ids', type=str, default='-1',
        #                          help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        # self.parser.add_argument('--name', type=str, default='coseg',
        #                          help='name of the experiment. It decides where to store samples and models')
        # self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # self.parser.add_argument('--serial_batches', action='store_true',
        #                          help='if true, takes meshes in order, otherwise takes them randomly')
        # self.parser.add_argument('--seed', type=int, help='if specified, uses seed')
        # self.parser.add_argument('--clean_data',  default=False, action='store_true',
        #                          help='If true, it deletes the cache NPZ files in the dataset before training.')
        # # visualization params
        # self.parser.add_argument('--export_folder', type=str, default='',
        #                          help='exports intermediate collapses to this folder')
        # self.initialized = True

        # PARAMETERS FOR GENERATIVE LEARNING WITH SIMPLEST CUBES
        # data params
        # self.parser.add_argument('--dataroot', default='datasets/simplest_cubes_2',
        #                          help='path to meshes (should have subfolders train, test)')
        # self.parser.add_argument('--dataset_mode', choices={"classification", "segmentation", "generative"},
        #                          default='generative')
        # self.parser.add_argument('--ninput_features', type=int, default=12,
        #                          help='# of input features (will include dummy features)')
        # self.parser.add_argument('--latent_path', default='datasets/latent/simplest_cube.obj',
        #                          help='Path to the OBJ containing the latent for connectivity')
        # # network params
        # self.parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
        # self.parser.add_argument('--arch', type=str, default='meshPointGAN',
        #                          choices={"mconvnet", "meshunet", "meshGAN", "meshPointGAN"},
        #                          help='selects network to use')
        # self.parser.add_argument('--resblocks', type=int, default=0, help='# of res blocks')
        # self.parser.add_argument('--fc_n', type=int, default=16, help='# between fc and nclasses')  # todo make generic
        # self.parser.add_argument('--ncf', nargs='+', default=[16, 32, 64], type=int, help='conv filters')
        # self.parser.add_argument('--pool_res', nargs='+', default=[12, 12, 12], type=int, help='pooling res')
        # self.parser.add_argument('--unpool_res', nargs='+', default=[12, 12, 12], type=int,
        #                          help='unpooling res (for MeshPointGAN)')
        # self.parser.add_argument('--norm', type=str, default='batch',
        #                          help='instance normalization or batch normalization or group normalization')
        # self.parser.add_argument('--num_groups', type=int, default=16, help='# of groups for groupnorm')
        # self.parser.add_argument('--init_type', type=str, default='normal',
        #                          help='network initialization [normal|xavier|kaiming|orthogonal]')
        # self.parser.add_argument('--init_gain', type=float, default=0.02,
        #                          help='scaling factor for normal, xavier and orthogonal.')
        # self.parser.add_argument('--face_pool', type=str, default='v2',
        #                          help='Version of face pool. For tracking purposes only.')
        # self.parser.add_argument('--vertex_features', nargs='+', type=str, default=['mean_c', 'gaussian_c'],
        #                          help='Type of vertex features to be used (for vertex convolution). Options are coord (for coordinates), norm (for normals), mean_c (for mean curvature) and gauss_c (for gaussian curvature)')
        #
        # self.parser.add_argument('--symm_oper', nargs='+', default=[1], type=int, help='pooling res')
        # self.parser.add_argument('--gen_steps', type=int, default=1, help='# of training steps for the generator')
        # self.parser.add_argument('--disc_steps', type=int, default=1, help='# of training steps for the discriminator')
        # self.parser.add_argument('--max_disc_acc', type=float, default=0.8,
        #                          help='Maximum accuracy for the discriminator')
        # self.parser.add_argument('--dilation',  default=False, action='store_true',
        #                          help='Determines if the generator outputs dilation (true) or vertex positions (false)')
        # # general params
        # self.parser.add_argument('--feat_from', type=str, default='point',
        #                          help='Primitive to extract features from. One of: edge, face, point')
        # self.parser.add_argument('--num_threads', default=3, type=int, help='# threads for loading data')
        # self.parser.add_argument('--gpu_ids', type=str, default='0',
        #                          help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        # self.parser.add_argument('--name', type=str, default='coseg',
        #                          help='name of the experiment. It decides where to store samples and models')
        # self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # self.parser.add_argument('--serial_batches', action='store_true',
        #                          help='if true, takes meshes in order, otherwise takes them randomly')
        # self.parser.add_argument('--seed', type=int, help='if specified, uses seed')
        # self.parser.add_argument('--clean_data',  default=False, action='store_true',
        #                          help='If true, it deletes the cache NPZ files in the dataset before training.')
        # # visualization params
        # self.parser.add_argument('--export_folder', type=str, default='',
        #                          help='exports intermediate collapses to this folder')
        # self.initialized = True

        # PARAMETERS FOR GENERATIVE LEARNING USING AUTOENCODER
        # data params
        self.parser.add_argument('--dataroot', default='datasets/simplest_cubes',
                                 help='path to meshes (should have subfolders train, test)')
        self.parser.add_argument('--dataset_mode', choices={"classification", "segmentation", "generative"}, default='generative')
        self.parser.add_argument('--ninput_features', type=int, default=8,
                                 help='# of input features (will include dummy features)')
        # network params
        self.parser.add_argument('--batch_size', type=int, default=12, help='input batch size')
        self.parser.add_argument('--arch', type=str, default='meshunet',
                                 choices={"mconvnet", "meshunet", "meshGAN", "meshPointGAN"},
                                 help='selects network to use')
        self.parser.add_argument('--resblocks', type=int, default=3, help='# of res blocks')
        self.parser.add_argument('--fc_n', type=int, default=100, help='# between fc and nclasses')  # todo make generic
        self.parser.add_argument('--ncf', nargs='+', default=[8, 16, 32], type=int, help='conv filters')
        self.parser.add_argument('--pool_res', nargs='+', default=[], type=int, help='pooling res')
        self.parser.add_argument('--norm', type=str, default='batch',
                                 help='instance normalization or batch normalization or group normalization')
        self.parser.add_argument('--num_groups', type=int, default=16, help='# of groups for groupnorm')
        self.parser.add_argument('--init_type', type=str, default='normal',
                                 help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--init_gain', type=float, default=0.02,
                                 help='scaling factor for normal, xavier and orthogonal.')
        self.parser.add_argument('--face_pool', type=str, default='v2',
                                 help='Version of face pool. For tracking purposes only.')
        self.parser.add_argument('--symm_oper', nargs='+', default=[1], type=int, help='pooling res')
        self.parser.add_argument('--vertex_features', nargs='+', type=str, default=['coord'],
                                 help='Type of vertex features to be used (for vertex convolution). Options are coord (for coordinates), norm (for normals), mean_c (for mean curvature) and gauss_c (for gaussian curvature)')
        self.parser.add_argument('--gen_steps', type=int, default=1, help='# of training steps for the generator')
        self.parser.add_argument('--disc_steps', type=int, default=1, help='# of training steps for the discriminator')
        # general params
        self.parser.add_argument('--feat_from', type=str, default='point',
                                 help='Primitive to extract features from. One of: edge, face')
        self.parser.add_argument('--num_threads', default=3, type=int, help='# threads for loading data')
        self.parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='autoencoder',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes meshes in order, otherwise takes them randomly')
        self.parser.add_argument('--seed', type=int, help='if specified, uses seed')
        self.parser.add_argument('--clean_data',  default=True, action='store_true',
                                 help='If true, it deletes the cache NPZ files in the dataset before training.')
        # visualization params
        self.parser.add_argument('--export_folder', type=str, default='',
                                 help='exports intermediate collapses to this folder')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt, unknown = self.parser.parse_known_args()
        self.opt.is_train = self.is_train   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        if self.opt.seed is not None:
            import numpy as np
            import random
            torch.manual_seed(self.opt.seed)
            np.random.seed(self.opt.seed)
            random.seed(self.opt.seed)

        if self.opt.export_folder:
            self.opt.export_folder = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.export_folder)
            util.mkdir(self.opt.export_folder)

        if self.is_train:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

            # save to the disk
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            util.mkdir(expr_dir)

            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt