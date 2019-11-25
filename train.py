import time
from options.train_options import TrainOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
from test import run_test

if __name__ == '__main__':
    
    import wandb
    wandb.init(project="meshcnn")

    #Parse options from arguments
    opt = TrainOptions().parse()
    wandb.config.update(opt)

    #Load dataset ready for training (with extracted features)
    dataset = DataLoader(opt)
    dataset_size = len(dataset)
    print('#training meshes = %d' % dataset_size)
    wandb.config.update({"training_samples": dataset_size})

    model = create_model(opt)
    writer = Writer(opt)
    total_steps = 0
    if (opt.arch=='meshGAN'):
        wandb.watch((model.net.generator, model.net.discriminator), log="all")
    else:
        wandb.watch(model.net, log="all")
    startT = time.time()
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.print_freq == 0:
                t = (time.time() - iter_start_time) / opt.batch_size
                if opt.dataset_mode == 'generative':
                    gen_loss = model.g_loss
                    disc_loss = model.d_loss
                    wandb.log({"Gen_loss": gen_loss, "Disc_loss": disc_loss, "D(x)": model.mean_output_disc_real,
                               "D(G(z))": model.mean_output_disc_fake, "Iters": total_steps})
                    writer.print_current_lossesGAN(epoch, epoch_iter, gen_loss, disc_loss, t, t_data)
                else:
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
            model.save_network('latest', wandb_save=False, dataset_mode=opt.dataset_mode)
            model.save_network(epoch, dataset_mode=opt.dataset_mode)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        wandb.log({"Epoch": epoch, "generated_model": wandb.Object3D(open('generated/unknown_0.obj'))})
        if opt.dataset_mode != 'generative':
            model.update_learning_rate()
            if opt.verbose_plot:
                writer.plot_model_wts(model, epoch)

            if epoch % opt.run_test_freq == 0:
                acc = run_test(epoch)
                writer.plot_acc(acc, epoch)
                wandb.log({"Test Accuracy": acc})

    wandb.log({"Training Time": time.time()-startT})
    writer.close()
