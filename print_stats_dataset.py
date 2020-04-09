from options.test_options import TestOptions
from data import DataLoader
import numpy as np


if __name__ == '__main__':
    opt = TestOptions().parse()
    dataset = DataLoader(opt)
    max_faces = 0
    max_edges = 0
    max_vs = 0
    for i, data in enumerate(dataset):
        for mesh in data['mesh']:
            if mesh.vs_count > max_vs:
                max_vs = mesh.vs_count
            if mesh.edges_count > max_edges:
                max_edges = mesh.edges_count
            if mesh.face_count > max_faces:
                max_faces = mesh.face_count
    print('Max faces: ', max_faces)
    print('Max vertices: ', max_vs)
    print('Max edges: ', max_edges)