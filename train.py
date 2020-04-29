import time
from options.train_options import TrainOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
from test import run_test
import os
from random import randrange
# from memory_profiler import profile
import torch
import glob
import shutil


if __name__ == '__main__':

    # Parse options from arguments
    opt = TrainOptions().parse()

    # # torch.autograd.set_detect_anomaly(True)
    # import wandb
    # wandb.init(project="meshcnn")
    #
    #
    #
    # if opt.name == 'sweep':
    #     if wandb.run.id != None:
    #         opt.name = wandb.run.id
    #     else:
    #         raise ValueError(wandb.run.id, 'Wrong value value in wandb.run.name')
    #
    # wandb.config.update(opt, allow_val_change=True)
    #
    # if opt.clean_data:
    #     dirs = glob.glob(opt.dataroot+'/*/*/cache') + glob.glob(opt.dataroot+'/*/cache')
    #     for dir in dirs:
    #         shutil.rmtree(dir)
    #     mean_files = glob.glob(opt.dataroot+'/*.p')
    #     for file in mean_files:
    #         os.remove(file)
    #
    # #Load dataset ready for training (with extracted features)
    # dataset = DataLoader(opt)
    # dataset_size = len(dataset)
    # print('#training meshes = %d' % dataset_size)
    # wandb.config.update({"training_samples": dataset_size})
    #
    # model = create_model(opt)
    # writer = Writer(opt)
    # total_steps = 0
    # freq_steps = 0
    # if 'GAN' in opt.arch:
    #     wandb.watch((model.net.generator, model.net.discriminator), log="all")
    # else:
    #     wandb.watch(model.net, log="all")
    # startT = time.time()
    #
    # gen_model_id = 0
    #
    # # for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    # for epoch in range(1):
    #     epoch_start_time = time.time()
    #     iter_data_time = time.time()
    #     epoch_iter = 0
    #
    #     for i, data in enumerate(dataset):
    #         iter_start_time = time.time()
    #         if freq_steps+opt.batch_size > opt.print_freq:
    #             t_data = iter_start_time - iter_data_time
    #         total_steps += opt.batch_size
    #         freq_steps += opt.batch_size
    #         epoch_iter += opt.batch_size
    #         model.set_input(data)
    #         model.optimize_parameters(epoch=epoch)
    #
    #         if freq_steps > opt.print_freq:
    #             freq_steps = 0
    #             t = (time.time() - iter_start_time) / opt.batch_size
    #             if opt.dataset_mode == 'generative' and 'GAN' in opt.arch:
    #                 gen_loss = model.g_loss
    #                 disc_loss = model.d_loss
    #                 wandb.log({"Gen_loss": gen_loss, "Disc_loss": disc_loss, "D(x)": model.mean_output_disc_real,
    #                            "D(G(z))": model.mean_output_disc_fake, "Iters": total_steps,
    #                            "Disc Accuracy": model.disc_accuracy})
    #                 writer.print_current_lossesGAN(epoch, epoch_iter, gen_loss, disc_loss, t, t_data)
    #             else:
    #                 loss = model.loss
    #                 writer.print_current_losses(epoch, epoch_iter, loss, t, t_data)
    #                 writer.plot_loss(loss, epoch, epoch_iter, dataset_size)
    #                 wandb.log({"loss": loss, "Iters": total_steps, "lr": model.optimizer.param_groups[-1]['lr']})
    #
    #             if opt.dataset_mode == 'generative':
    #                 # Pick a random generated model and export it
    #                 gen_model_id += 1
    #                 export_filename = os.path.join(opt.checkpoints_dir, opt.name, 'generated',
    #                                                'gen_mesh_' + str(gen_model_id) + '.obj')
    #                 ridx = randrange(model.gen_models.shape[0])
    #                 model.gen_models[ridx].export(file=export_filename)
    #                 wandb.log({"generated_model": wandb.Object3D(open(export_filename))})
    #
    #         if i % opt.save_latest_freq == 0:
    #             print('saving the latest model (epoch %d, total_steps %d)' %
    #                   (epoch, total_steps))
    #             model.save_network('latest', dataset_mode=opt.dataset_mode)
    #
    #         iter_data_time = time.time()
    #     if epoch % opt.save_epoch_freq == 0:
    #         print('saving the model at the end of epoch %d, iters %d' %
    #               (epoch, total_steps))
    #         model.save_network('latest', wandb_save=False, dataset_mode=opt.dataset_mode)
    #         # model.save_network(epoch, dataset_mode=opt.dataset_mode)
    #
    #     print('End of epoch %d / %d \t Time Taken: %d sec' %
    #           (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    #     wandb.log({"Epoch": epoch})
    #
    #     if 'GAN' not in opt.arch:
    #         model.update_learning_rate()
    #
    #     if opt.dataset_mode != 'generative':
    #
    #         if opt.verbose_plot:
    #             writer.plot_model_wts(model, epoch)
    #
    #         if epoch % opt.run_test_freq == 0:
    #             acc = run_test(epoch)
    #             writer.plot_acc(acc, epoch)
    #             wandb.log({"Test Accuracy": acc})
    #
    #     writer.close()
    # wandb.log({"Training Time": time.time() - startT})
    # run_test()