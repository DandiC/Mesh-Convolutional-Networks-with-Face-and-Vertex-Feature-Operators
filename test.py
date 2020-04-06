from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
import glob
import shutil
import os
import wandb
import torch
from models.layers.mesh import Mesh
import copy
import matplotlib.pyplot as plt
import numpy as np

def run_test(epoch=-1):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle

    if opt.name == 'sweep':
        if wandb.run.id != None:
            opt.name = wandb.run.id
        else:
            raise ValueError(wandb.run.id, 'Wrong value value in wandb.run.name')

    if opt.clean_data:
        dirs = glob.glob(opt.dataroot+'/*/*/cache') + glob.glob(opt.dataroot+'/*/cache')
        for dir in dirs:
            shutil.rmtree(dir)
        mean_files = glob.glob(opt.dataroot+'/*.p')
        for file in mean_files:
            os.remove(file)

    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()
    if opt.dataset_mode == 'generative':
        #Interpolate in latent space
        mesh = Mesh(opt.latent_path, opt=opt)
        latent = model.encode(mesh)
        for i in range(8):
            for j in range(-4,5):
                z = copy.deepcopy(latent)
                z[0,0,i] = 0.5*j
                gen_mesh = model.generate(z)
                gen_mesh.export(file=model.sample_folder + '/gen_mesh_' + str(i) + '_' + str(j) + '.obj')

        latents_dir = opt.checkpoints_dir + '/' + opt.name + '/latents/'
        if not os.path.exists(latents_dir):
            os.makedirs(latents_dir)
        # Generate mesh for each input mesh
        for i, data in enumerate(dataset):
            model.set_input(data)
            model.test()

            for j, mesh in enumerate(data['mesh']):
                latent = model.encode(mesh).squeeze().data.cpu().numpy()
                title = mesh.filename
                _ = plt.bar(np.arange(latent.size), latent)
                plt.title(title)
                plt.savefig(latents_dir + mesh.filename.replace('.obj', '.png'))
                plt.clf()
                if i==0 and j==0:
                    latents = latent
                else:
                    latents = np.concatenate((latents, latent))

        _ = plt.hist(latents, bins='auto')
        plt.savefig(latents_dir + 'all_latents_hist.png')


    else:
        for i, data in enumerate(dataset):
            model.set_input(data)
            ncorrect, nexamples = model.test()
            writer.update_counter(ncorrect, nexamples)
        writer.print_acc(epoch, writer.acc)
        return writer.acc



if __name__ == '__main__':
    run_test()
