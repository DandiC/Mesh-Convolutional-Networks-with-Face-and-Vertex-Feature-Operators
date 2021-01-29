from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
from util.util import clean_data
import os
import json


def run_test(epoch=-1, import_opt=False):
    print('Running Test')

    if import_opt:
        test_opt = TestOptions().parse()
        expr_dir = os.path.join(test_opt.checkpoints_dir, test_opt.name)
        opt = TestOptions().parse()
        with open(os.path.join(expr_dir, 'opt.json'), 'r') as f:
            opt.__dict__ = json.load(f)

        opt.results_dir ='./results/'
        opt.phase = 'test'
        opt.which_epoch='latest'
        opt.num_aug=1
        opt.gpu_ids = test_opt.gpu_ids
        opt.export_segments = test_opt.export_segments
    else:
        opt = TestOptions().parse()

    opt.clean_data = False
    opt.serial_batches = True  # no shuffle
    opt.is_train = False

    if opt.clean_data:
        clean_data(opt)

    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)

    # test
    writer.reset_counter()

    for i, data in enumerate(dataset):
        model.set_input(data)
        ncorrect, nexamples = model.test()
        writer.update_counter(ncorrect, nexamples)
    writer.print_acc(epoch, writer.acc)
    return writer.acc


if __name__ == '__main__':
    run_test(import_opt=True)
