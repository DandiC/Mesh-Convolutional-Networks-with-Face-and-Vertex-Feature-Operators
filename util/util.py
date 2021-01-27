from __future__ import print_function
import torch
import numpy as np
import os
import glob
import shutil

def clean_data(opt):
    dirs = glob.glob(opt.dataroot + '/*/*/cache') + glob.glob(opt.dataroot + '/*/cache')
    for dir in dirs:
        shutil.rmtree(dir)
    mean_files = glob.glob(opt.dataroot + '/*.p')
    for file in mean_files:
        os.remove(file)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

MESH_EXTENSIONS = [
    '.obj',
]


def is_mesh_file(filename):
    return any(filename.endswith(extension) for extension in MESH_EXTENSIONS)


def pad(input_arr, target_length, val=0, dim=1):
    shp = input_arr.shape
    npad = [(0, 0) for _ in range(len(shp))]
    npad[dim] = (0, target_length - shp[dim])
    return np.pad(input_arr, pad_width=npad, mode='constant', constant_values=val)


def seg_accuracy(predicted, ssegs, meshes, feature_type='edge'):
    correct = 0
    ssegs = ssegs.squeeze(-1)
    correct_mat = ssegs.gather(2, predicted.cpu().unsqueeze(dim=2))
    for mesh_id, mesh in enumerate(meshes):
        if feature_type == 'edge':
            correct_vec = correct_mat[mesh_id, :mesh.edges_count, 0]
            edge_areas = torch.from_numpy(mesh.get_edge_areas())
            correct += (correct_vec.float() * edge_areas).sum()
        elif feature_type == 'face':
            correct_vec = correct_mat[mesh_id, :mesh.face_count, 0]
            face_areas = torch.from_numpy(mesh.face_areas / np.sum(mesh.face_areas))
            correct += (correct_vec.float() * face_areas).sum()
        elif feature_type == 'point':
            correct_vec = correct_mat[mesh_id, :mesh.vs_count, 0]
            correct += (correct_vec.float()).sum() / mesh.vs_count
    return correct


def print_network(net):
    """Print the total number of parameters in the network
    Parameters:
        network
    """
    print('---------- Network initialized -------------')
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
    print('-----------------------------------------------')
    return num_params
