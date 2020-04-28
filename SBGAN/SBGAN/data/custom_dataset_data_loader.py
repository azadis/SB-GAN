import torch.utils.data

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sbgan_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, sbgan_dir) 

from SBGAN.data.base_data_loader import BaseDataLoader


def CreateDataset(opt, train=True):
    dataset = None

    if opt.dataset=='cityscapes' or opt.dataset=='ade_indoor' or opt.dataset=='cityscapes_full_weighted':
        from SBGAN.data.custom_dataset import SupDataset
        dataset = SupDataset(opt, train)

    else:
        from SBGAN.data.custom_dataset import CustomDataset
        dataset = CustomDataset(opt)


    print("dataset [%s] was created" % (dataset.name()))

    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=opt.train,
            num_workers=opt.num_workers)
        self.opt = opt

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)


    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data
