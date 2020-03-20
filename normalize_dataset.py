from options.test_options import TestOptions
from data import DataLoader
import numpy as np


if __name__ == '__main__':
    opt = TestOptions().parse()
    dataset = DataLoader(opt)

    for i, data in enumerate(dataset):
        for mesh in data['mesh']:
            mesh.vs = 2.*(mesh.vs - np.min(mesh.vs))/np.ptp(mesh.vs)-1
            mesh.export(file=opt.dataroot + '/normalized/' + mesh.filename)