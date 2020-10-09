from options.test_options import TestOptions
from data import DataLoader
import numpy as np
import glob
from models.layers.mesh import Mesh
import sys

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
    mesh_files = glob.glob('D:/Daniel/Google Drive/Universidad/PhD/MeshCNN/MeshCNN/datasets/cubes/*/*.obj')
    # mesh_files = glob.glob('datasets/latent/*v.obj')
    # mesh_files = glob.glob('checkpoints/*/results/*.obj')
    min_vs = sys.float_info.max
    max_vs = -sys.float_info.max
    max_faces = 0
    max_edges = 0
    max_vs = 0
    for file in mesh_files:
        mesh = Mesh(file, opt=opt)
        mesh_min = np.min(mesh.vs)
        mesh_max = np.max(mesh.vs)
        if mesh_min < min_vs:
            min_vs = mesh_min
        if mesh_max > max_vs:
            max_vs = mesh_max
        if mesh.vs_count > max_vs:
            max_vs = mesh.vs_count
        if mesh.edges_count > max_edges:
            max_edges = mesh.edges_count
        if mesh.face_count > max_faces:
            max_faces = mesh.face_count

    print('Max faces: ', max_faces)
    print('Max vertices: ', max_vs)
    print('Max edges: ', max_edges)

    print('Max:', max_vs)
    print('Min:', min_vs)