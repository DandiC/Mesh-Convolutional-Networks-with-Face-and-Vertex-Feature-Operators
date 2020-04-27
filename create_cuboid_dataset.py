from models.layers.mesh import Mesh
from options.train_options import TrainOptions
import numpy as np
import copy
import os

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset_folder = 'datasets/sphere_382v/'
    latent_mesh = Mesh('datasets/latent/spheroids_382v.obj', opt=opt)

    mesh = copy.deepcopy(latent_mesh)

    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    if not os.path.exists(dataset_folder+'train/'):
        os.makedirs(dataset_folder+'train/')
    if not os.path.exists(dataset_folder+'test/'):
        os.makedirs(dataset_folder+'test/')


    print('Creating training data')
    print_id = 0
    for i in range(1000):
        rnd_ids = np.random.permutation(mesh.vs.shape[0])
        mesh.vs = latent_mesh.vs[rnd_ids]
        for v_id in range(mesh.vs.shape[0]):
            mesh.faces[latent_mesh.faces==rnd_ids[v_id]] = v_id
        rnd = np.random.uniform(low=0.1, high=2, size=3)
        mesh.vs = latent_mesh.vs[rnd_ids] * rnd
        mesh.export_raw(file=dataset_folder+'train/mesh_'+str(print_id)+'.obj')
        print_id+=1

    print('Creating testing data')
    print_id = 0
    for i in range(100):
        rnd_ids = np.random.permutation(mesh.vs.shape[0])
        mesh.vs = latent_mesh.vs[rnd_ids]
        for v_id in range(mesh.vs.shape[0]):
            mesh.faces[latent_mesh.faces == rnd_ids[v_id]] = v_id
        rnd = np.random.uniform(low=0.1, high=2, size=3)
        mesh.vs = latent_mesh.vs[rnd_ids] * rnd
        mesh.export_raw(file=dataset_folder + 'test/mesh_' + str(print_id) + '.obj')
        print_id += 1
