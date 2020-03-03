from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
import glob
import shutil
import os
import wandb

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
        for i, data in enumerate(dataset):
            model.set_input(data)
            model.test()
    else:
        for i, data in enumerate(dataset):
            model.set_input(data)
            ncorrect, nexamples = model.test()
            writer.update_counter(ncorrect, nexamples)
        writer.print_acc(epoch, writer.acc)
        return writer.acc



if __name__ == '__main__':
    run_test()
