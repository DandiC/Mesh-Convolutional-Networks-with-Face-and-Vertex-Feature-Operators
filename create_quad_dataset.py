from models.layers.mesh import Mesh
from options.train_options import TrainOptions
import numpy as np
import copy

if __name__ == '__main__':
    opt = TrainOptions().parse()
    latent_mesh = Mesh('datasets/latent/simplest_cube.obj', opt=opt)
    mesh = copy.deepcopy(latent_mesh)
    for i in range(100000):
        rnd = np.random.uniform(low=0.1, high=2, size=3)
        mesh.vs = latent_mesh.vs*rnd
        mesh.export(file='datasets/simple_quads/quads/quad_'+str(i)+'.obj')
    a=1