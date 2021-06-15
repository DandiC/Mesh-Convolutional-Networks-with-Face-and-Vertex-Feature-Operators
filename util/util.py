from __future__ import print_function
import torch
import numpy as np
import os
import wandb
import glob
import shutil
import neuralnet_pytorch as nnt
from neuralnet_pytorch.utils.tensor_utils import batch_pairwise_dist

def clean_data(opt):
    dirs = glob.glob(opt.dataroot + '/*/*/cache') + glob.glob(opt.dataroot + '/*/cache')
    for dir in dirs:
        shutil.rmtree(dir)
    mean_files = glob.glob(opt.dataroot + '/*.p')
    for file in mean_files:
        os.remove(file)
    latent_files = glob.glob('datasets/latent/cache/'+opt.latent_path.replace('datasets/latent/','').replace('.obj','*npz'))
    for file in latent_files:
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
            # correct += correct_vec.float().sum() / mesh.edges_count
        elif feature_type == 'face':
            correct_vec = correct_mat[mesh_id, :mesh.face_count, 0]
            face_areas = torch.from_numpy(mesh.face_areas / np.sum(mesh.face_areas))
            correct += (correct_vec.float() * face_areas).sum()
            # correct += correct_vec.float().sum() / mesh.face_count
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

def get_heatmap_color(value, minimum=0, maximum=1):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b


def normalize_np_array(np_array):
    min_value = np.min(np_array)
    max_value = np.max(np_array)
    return (np_array - min_value) / (max_value - min_value)


def calculate_entropy(np_array):
    entropy = 0
    np_array /= np.sum(np_array)
    for a in np_array:
        if a != 0:
            entropy -= a * np.log(a)
    entropy /= np.log(np_array.shape[0])
    return entropy

def chamfer_distance(p1, p2, debug=False):

    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[B, N, D]
    :param p2: size[B, M, D]
    :param debug: whether need to output debug info
    :return: sum of all batches of Chamfer Distance of two point sets
    '''

    assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)

    if debug:
        print(p1[0])

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)
    if debug:
        print('p1 size is {}'.format(p1.size()))
        print('p2 size is {}'.format(p2.size()))
        print(p1[0][0])

    p1 = p1.repeat(1, p2.size(2), 1, 1)
    if debug:
        print('p1 size is {}'.format(p1.size()))

    p1 = p1.transpose(1, 2)
    if debug:
        print('p1 size is {}'.format(p1.size()))
        print(p1[0][0])

    p2 = p2.repeat(1, p1.size(1), 1, 1)
    if debug:
        print('p2 size is {}'.format(p2.size()))
        print(p2[0][0])

    dist = torch.add(p1, torch.neg(p2))
    if debug:
        print('dist size is {}'.format(dist.size()))
        print(dist[0])

    dist = torch.norm(dist, 2, dim=3)
    if debug:
        print('dist size is {}'.format(dist.size()))
        print(dist)

    dist = torch.min(dist, dim=2)[0]
    if debug:
        print('dist size is {}'.format(dist.size()))
        print(dist)

    dist = torch.sum(dist)
    if debug:
        print('-------')
        print(dist)

    return dist

def sample(vertices, mesh, num_samples: int, eps: float = 1e-10):
        r""" Uniformly samples the surface of a mesh.
            Args:
                num_samples (int): number of points to sample
                eps (float): a small number to prevent division by zero
                             for small surface areas.
            Returns:
                (torch.Tensor, torch.Tensor) uniformly sampled points and
                    the face idexes which each point corresponds to.
            Example:
                points, chosen_faces = mesh.sample(10)
                points
                tensor([[ 0.0293,  0.2179,  0.2168],
                        [ 0.2003, -0.3367,  0.2187],
                        [ 0.2152, -0.0943,  0.1907],
                        [-0.1852,  0.1686, -0.0522],
                        [-0.2167,  0.3171,  0.0737],
                        [ 0.2219, -0.0289,  0.1531],
                        [ 0.2217, -0.0115,  0.1247],
                        [-0.1400,  0.0364, -0.1618],
                        [ 0.0658, -0.0310, -0.2198],
                        [ 0.1926, -0.1867, -0.2153]])
                chosen_faces
                tensor([ 953,  38,  6, 3480,  563,  393,  395, 3309, 373, 271])
        """

        # vertices = torch.Tensor(self.vs)
        faces = torch.Tensor(mesh.faces).long().to(vertices.device)

        if vertices.is_cuda:
            dist_uni = torch.distributions.Uniform(
                torch.tensor([0.0]).cuda(), torch.tensor([1.0]).cuda())
        else:
            dist_uni = torch.distributions.Uniform(
                torch.tensor([0.0]), torch.tensor([1.0]))

        # calculate area of each face
        Areas = torch.tensor(mesh.face_areas).unsqueeze(1)
        # percentage of each face w.r.t. full surface area
        Areas = Areas / (torch.sum(Areas) + eps)

        # define descrete distribution w.r.t. face area ratios caluclated
        cat_dist = torch.distributions.Categorical(Areas.view(-1))
        face_choices = cat_dist.sample([num_samples])

        # from each face sample a point
        select_faces = faces[face_choices]
        v0 = torch.index_select(vertices, 0, select_faces[:, 0])
        v1 = torch.index_select(vertices, 0, select_faces[:, 1])
        v2 = torch.index_select(vertices, 0, select_faces[:, 2])
        u = torch.sqrt(dist_uni.sample([num_samples]))
        v = dist_uni.sample([num_samples])
        points = (1 - u) * v0 + (u * (1 - v)) * v1 + u * v * v2

        return points

def batch_sample(verts, mesh, num=10000):
    dist_uni = torch.distributions.Uniform(torch.tensor([0.0]).cuda(), torch.tensor([1.0]).cuda())
    batch_size = verts.shape[0]
    # calculate area of each face
    faces = torch.Tensor(mesh[0].faces).to(verts.device).long()
    x1, x2, x3 = torch.split(torch.index_select(verts, 1, faces[:, 0]) - torch.index_select(verts, 1, faces[:, 1]), 1, dim=-1)
    y1, y2, y3 = torch.split(torch.index_select(verts, 1, faces[:, 1]) - torch.index_select(verts, 1, faces[:, 2]), 1, dim=-1)
    a = (x2 * y3 - x3 * y2) ** 2
    b = (x3 * y1 - x1 * y3) ** 2
    c = (x1 * y2 - x2 * y1) ** 2
    Areas = torch.sqrt(a + b + c) / 2
    Areas = Areas.squeeze(-1) / torch.sum(Areas, dim=1)  # percentage of each face w.r.t. full surface area

    # define descrete distribution w.r.t. face area ratios caluclated
    choices = None
    for A in Areas:

        if choices is None:
            choices = torch.multinomial(A, num, True)  # list of faces to be sampled from
        else:
            choices = torch.cat((choices, torch.multinomial(A, num, True)))

    # from each face sample a point
    select_faces = faces[choices].view(verts.shape[0], 3, num)

    face_arange = verts.shape[1] * torch.arange(0, batch_size).cuda().unsqueeze(-1).expand(batch_size, num)
    select_faces = select_faces + face_arange.unsqueeze(1)

    select_faces = select_faces.view(-1, 3)
    flat_verts = verts.view(-1, 3)

    xs = torch.index_select(flat_verts, 0, select_faces[:, 0])
    ys = torch.index_select(flat_verts, 0, select_faces[:, 1])
    zs = torch.index_select(flat_verts, 0, select_faces[:, 2])
    u = torch.sqrt(dist_uni.sample_n(batch_size * num))
    v = dist_uni.sample_n(batch_size * num)

    points = (1 - u) * xs + (u * (1 - v)) * ys + u * v * zs
    points = points.view(batch_size, num, 3)

    return points

def f1_score(generated, gt, reduce='mean'):
    assert len(generated.shape) in (2, 3) and len(gt.shape) in (2, 3), 'Unknown shape of tensors'

    if generated.dim() == 2:
        generated = generated.unsqueeze(0)

    if gt.dim() == 2:
        gt = gt.unsqueeze(0)

    batch_size = generated.shape[0]
    num_points= generated.shape[1]

    P = batch_pairwise_dist(generated, gt)
    dist_to_pred, _ = torch.min(P, 1)
    dist_to_gt, _ = torch.min(P, 2)

    f_score = 0
    for i in range(dist_to_pred.shape[0]):
        recall = float(torch.where(dist_to_pred[i] <= 1e-2)[0].shape[0]) / float(num_points)
        precision = float(torch.where(dist_to_gt[i] <= 1e-2)[0].shape[0]) / float(num_points)

        f_score += 2 * (precision * recall) / (precision + recall + 1e-8)
    f_score = f_score / (batch_size)
    return f_score

def batch_point_to_point(pred_vert, mesh, gt_points, num=1000, f1=False):


    # grab the faces still in use
    batch_size = pred_vert.shape[0]

    # sample from faces and calculate pairs

    pred_points = batch_sample(pred_vert, mesh, num=num)

    id_p, id_g = nnt.chamfer_loss(gt_points, pred_points)

    # select pairs and calculate chamfer distance

    pred_points = pred_points.view(-1, 3)
    gt_points = gt_points.contiguous().view(-1, 3)

    points_range = num * torch.arange(0, batch_size).cuda().unsqueeze(-1).expand(batch_size, num)
    id_p = (id_p.long() + points_range).view(-1)
    id_g = (id_g.long() + points_range).view(-1)

    pred_counters = torch.index_select(pred_points, 0, id_p)
    gt_counters = torch.index_select(gt_points, 0, id_g)

    dist_1 = torch.mean(torch.sum((gt_counters - pred_points) ** 2, dim=1))
    dist_2 = torch.mean(torch.sum((pred_counters - gt_points) ** 2, dim=1))

    loss = (dist_1 + dist_2) * 3000

    if f1:
        dist_to_pred = torch.sqrt(torch.sum((.57 * pred_counters - .57 * gt_points) ** 2, dim=1)).view(batch_size, -1)
        dist_to_gt = torch.sqrt(torch.sum((.57 * gt_counters - .57 * pred_points) ** 2, dim=1)).view(batch_size, -1)

        f_score = 0
        for i in range(dist_to_pred.shape[0]):
            recall = float(torch.where(dist_to_pred[i] <= 1e-2)[0].shape[0]) / float(num)
            precision = float(torch.where(dist_to_gt[i] <= 1e-2)[0].shape[0]) / float(num)

            f_score += 2 * (precision * recall) / (precision + recall + 1e-8)
        f_score = f_score / (batch_size)
        return loss, f_score
    else:
        return loss