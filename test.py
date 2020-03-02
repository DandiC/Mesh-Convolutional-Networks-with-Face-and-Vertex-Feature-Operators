from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
import glob
import shutil
import os

def run_test(epoch=-1):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle

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
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()



if __name__ == '__main__':
    run_test()
