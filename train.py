import time
from options.train_options import TrainOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
from test import run_test
import os
from random import randrange
from util.util import clean_data
import torch
import glob
import shutil
import numpy as np

if __name__ == '__main__':

    # Parse options from arguments.
    opt = TrainOptions().parse()

    # TODO: Remove wandb references.
    import wandb
    wandb.init(project="meshcnn")
    if opt.name == 'sweep':
        if wandb.run.id != None:
            opt.name = wandb.run.id
        else:
            raise ValueError(wandb.run.id, 'Wrong value value in wandb.run.name')

    wandb.config.update(opt, allow_val_change=True)

    if opt.clean_data:
        clean_data(opt)

    # Load dataset ready for training (with extracted features).
    dataset = DataLoader(opt)
    dataset_size = len(dataset)
    print('#training meshes = %d' % dataset_size)
    wandb.config.update({"training_samples": dataset_size})

    model = create_model(opt)
    writer = Writer(opt)
    total_steps = 0
    freq_steps = 0

    wandb.watch(model.net, log="all")

    startT = time.time()

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if freq_steps + opt.batch_size > opt.print_freq:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            freq_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters(epoch=epoch)

            if freq_steps > opt.print_freq:
                freq_steps = 0
                t = (time.time() - iter_start_time) / opt.batch_size
                loss = model.loss
                writer.print_current_losses(epoch, epoch_iter, loss, t, t_data)
                writer.plot_loss(loss, epoch, epoch_iter, dataset_size)
                wandb.log({"loss": loss, "Iters": total_steps, "lr": model.optimizer.param_groups[-1]['lr']})

            if i % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_network('latest', dataset_mode=opt.dataset_mode)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            # TODO: Change this before release.
            model.save_network('latest', wandb_save=False, dataset_mode=opt.dataset_mode)
            # model.save_network(epoch, dataset_mode=opt.dataset_mode)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        wandb.log({"Epoch": epoch})

        # TODO: Look into this, I've never used it so maybe I should delete it.
        if opt.verbose_plot:
            writer.plot_model_wts(model, epoch)

        if epoch % opt.run_test_freq == 0:
            acc = run_test(epoch)
            writer.plot_acc(acc, epoch)
            wandb.log({"Test Accuracy": acc})

        writer.close()

    wandb.log({"Training Time": time.time() - startT})
    run_test()