from models.layers.mesh import Mesh
from options.train_options import TrainOptions
import numpy as np
import copy

if __name__ == '__main__':
    opt = TrainOptions().parse()
    latent_mesh = Mesh('datasets/latent/simple_cube.obj', opt=opt)
    mesh = copy.deepcopy(latent_mesh)
    print_id = 0
    for i in range(1000):
        rnd_ids = np.random.permutation(mesh.vs.shape[0])
        mesh.vs = latent_mesh.vs[rnd_ids]
        for v_id in range(mesh.vs.shape[0]):
            mesh.faces[latent_mesh.faces==rnd_ids[v_id]] = v_id
        rnd = np.random.uniform(low=0.1, high=2, size=3)
        mesh.vs = latent_mesh.vs[rnd_ids] * rnd
        # first_v = copy.deepcopy(mesh.vs[0])
        # mesh.vs[:-1] = mesh.vs[1:]
        # mesh.vs[-1] = first_v
        # mesh.faces[mesh.faces == 0] = mesh.vs.shape[0]
        # mesh.faces -= 1
        # mesh.edges[mesh.edges == 0] = mesh.vs.shape[0]
        # mesh.edges -= 1
        # if i%2==0:
        mesh.export_raw(file='datasets/cuboids_26v/cuboids/cuboid_'+str(print_id)+'.obj')
        print_id+=1
    a=1