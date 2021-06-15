from options.test_options import TestOptions
from data import DataLoader
import numpy as np
import glob
from models.layers.mesh import Mesh
import sys
import os

# Using dataloader
# if __name__ == '__main__':
#     opt = TestOptions().parse()
#     dataset = DataLoader(opt)
#     max_faces = 0
#     max_edges = 0
#     max_vs = 0
#     for i, data in enumerate(dataset):
#         for mesh in data['mesh']:
#             if mesh.vs_count > max_vs:
#                 max_vs = mesh.vs_count
#             if mesh.edges_count > max_edges:
#                 max_edges = mesh.edges_count
#             if mesh.face_count > max_faces:
#                 max_faces = mesh.face_count
#     print('Max faces: ', max_faces)
#     print('Max vertices: ', max_vs)
#     print('Max edges: ', max_edges)

# Reading files
if __name__ == '__main__':
    opt = TestOptions().parse()
    resolution = 1000
    # dataset_folder = 'datasets/ModelNet40_manifold_res_' + str(resolution) + '/'
    dataset_folder = 'datasets/human_seg'
    mesh_files = glob.glob(dataset_folder + '*/*/*.obj')
    # mesh_files = glob.glob('datasets/latent/*v.obj')
    # mesh_files = glob.glob('checkpoints/*/results/*.obj')
    min_vs = sys.float_info.max
    max_vs = -sys.float_info.max
    max_faces = 0
    max_edges = 0
    max_vs = 0
    for i, file in enumerate(mesh_files):
        if i % 100 == 0:
            print('Processing mesh {}/{}'.format(i, len(mesh_files)))
        try:
            mesh = Mesh(file, opt=opt)
        except:
            print('Removed {} due to reading error.'.format(file))
            os.remove(file)
            continue
        # if mesh.face_count != resolution:
        #     print('Removed {} ({} faces)'.format(file, mesh.face_count))
        #     os.remove(file)
        #     continue
        # mesh_min = np.min(mesh.vs)
        # mesh_max = np.max(mesh.vs)
        # if mesh_min < min_vs:
        #     min_vs = mesh_min
        # if mesh_max > max_vs:
        #     max_vs = mesh_max
        if mesh.vs_count > max_vs:
            max_vs = mesh.vs_count
        if mesh.edges_count > max_edges:
            max_edges = mesh.edges_count
        if mesh.face_count > max_faces:
            max_faces = mesh.face_count

    print('Max faces: ', max_faces)
    print('Max edges: ', max_edges)
    print('Max vertices: ', max_vs)

    file = open(os.path.join(dataset_folder, 'stats.txt'), 'w')
    file.write('Max faces: {}\n'.format(max_faces))
    file.write('Max edges: {}\n'.format(max_edges))
    file.write('Max vertices: {}\n'.format(max_vs))

    # print('Max:', max_vs)
    # print('Min:', min_vs)