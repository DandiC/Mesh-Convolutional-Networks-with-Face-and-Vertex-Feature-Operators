import torch.utils.data
from data.base_dataset import collate_fn

def CreateDataset(opt):
    """loads dataset class"""

    if opt.dataset_mode == 'segmentation':
        from data.segmentation_data import SegmentationData
        dataset = SegmentationData(opt)
    elif opt.dataset_mode == 'classification':
        from data.classification_data import ClassificationData
        dataset = ClassificationData(opt)
    return dataset


class DataLoader:
    """multi-threaded data loading"""

    def __init__(self, opt):
        self.opt = opt
        # Create dataset with features
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads),
            collate_fn=collate_fn)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
