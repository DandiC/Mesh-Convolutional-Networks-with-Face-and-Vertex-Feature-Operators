from options.test_options import TestOptions
from data import DataLoader
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    opt = TestOptions().parse()
    dataset = DataLoader(opt)

    for i, data in enumerate(dataset):
        if i==0:
            features = data['features']
        else:
            features = np.concatenate((features, data['features']))
    title = 'Feature(s): '
    for f in opt.vertex_features:
        title += f + ' '
    _ = plt.hist(features.flatten(), bins='auto')  # arguments are passed to np.histogram
    plt.title(title)
    plt.show()
    while(1): a=1