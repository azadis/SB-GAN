from data.base_dataset import BaseDataset
import numpy as np
import PIL
import torch
import torchvision.transforms as transforms
import os
from scipy import misc
import random

class RandomCropLongEdge(object):
  """Crops the given PIL Image on the long edge with a random start point.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """
  def __call__(self, img):
    """
    Args:
        img (PIL Image): Image to be cropped.
    Returns:
        PIL Image: Cropped image.
    """
    size = (min(img.size), min(img.size))
    # Only step forward along this edge if it's the long edge
    i = (0 if size[0] == img.size[0] 
          else np.random.randint(low=0,high=img.size[0] - size[0]))
    j = (0 if size[1] == img.size[1]
          else np.random.randint(low=0,high=img.size[1] - size[1]))
    return transforms.functional.crop(img, j, i, size[0], size[1])

  def __repr__(self):
    return self.__class__.__name__


class SupDataset(BaseDataset):
    #taking sample from a custom dataset
    def __init__(self, opt, train, transform_im=None, transform_lbl=None):
        print(opt.dataset)
        if opt.dataset=='cityscapes' or opt.dataset=='cityscapes_full_weighted' or opt.dataset=='ade_indoor':
            inpath = 'datasets/%s'%opt.dataset
            if train:
                self.datalist_im = "%s_im_info_train.txt"%inpath
                self.datalist_lbl = "%s_lbl_info_train.txt"%inpath
            else:
                self.datalist_im = "%s_im_info_val.txt"%inpath
                self.datalist_lbl = "%s_lbl_info_val.txt"%inpath

            with open(self.datalist_im) as f:
                # datalist is a txt file containing paths of images X, and their labels 
                self.info_im = (f.readlines())
            with open(self.datalist_lbl) as f:
                self.info_lbl = (f.readlines())
            
            if not opt.not_sort:
                self.info_im = sorted(self.info_im)
                self.info_lbl = sorted(self.info_lbl)

            self.X_paths_im = [x.strip().split(' ')[0] for x in self.info_im]
            self.X_paths_lbl = [x.strip().split(' ')[0] for x in self.info_lbl]

            self.Ys = [int(x.strip().split(' ')[1]) for x in self.info_im]
            print("number of images:", len(self.Ys))
            if train:
                randinds = np.random.permutation(len(self.Ys))
            else:
                randinds = range(len(self.Ys))
            self.X_paths_im = [self.X_paths_im[i] for i in randinds]
            self.X_paths_lbl = [self.X_paths_lbl[i] for i in randinds]
            self.Ys = [int(self.Ys[i]) for i in randinds]
        else:
            self.X_paths_im = []
            self.X_paths_lbl = []
            # self.Ys = []
        self.transform_im = transform_im
        self.transform_lbl = transform_lbl
        self.dataset_name = opt.dataset
        self.opt = opt
    def __getitem__(self, index):


        transform_im = self.transform_im.copy()
        transform_lbl = self.transform_lbl.copy()
        flip = random.random() > 0.5
        if not flip:
            if 'cityscapes' in self.opt.dataset:
                del transform_im[1]
                del transform_lbl[1]
            else:
                del transform_im[1]
                del transform_lbl[1]


        transform_im = transforms.Compose(transform_im)
        transform_lbl = transforms.Compose(transform_lbl)

        X_path_im = self.X_paths_im[index]
        X_path_lbl = self.X_paths_lbl[index]
        Y = self.Ys[index]
        
        X_im = misc.imread(X_path_im)
        X_im = PIL.Image.fromarray(X_im.astype('uint8'))

        X_lbl = PIL.Image.open(X_path_lbl)
        if 'weighted' in self.dataset_name:
            if Y==1:
                X_lbl = transforms.functional.crop(X_lbl, 0, 256, 768, 1536)

        if 'cityscapes' not in self.opt.dataset:

            # if X_lbl and X_im do not have the same size: resize one of them
            if X_lbl.size[0] > X_im.size[0] or X_lbl.size[1] > X_im.size[1]:
                t_presize = transforms.Resize(X_lbl.size[::-1], PIL.Image.BILINEAR)
                X_im = t_presize(X_im)
            elif X_lbl.size[0] < X_im.size[0] or X_lbl.size[1] < X_im.size[1]:
                t_presize = transforms.Resize(X_im.size[::-1], PIL.Image.NEAREST)
                X_lbl = t_presize(X_lbl)

            #crop both images with the same index
            size = (min(X_lbl.size), min(X_lbl.size))
            # Only step forward along this edge if it's the long edge
            i = (0 if size[0] == X_lbl.size[0] 
                  else np.random.randint(low=0,high=X_lbl.size[0] - size[0]))
            j = (0 if size[1] == X_lbl.size[1]
                  else np.random.randint(low=0,high=X_lbl.size[1] - size[1]))

            X_lbl =transforms.functional.crop(X_lbl, j, i, size[0], size[1])

        X_ = np.asarray(X_lbl) 
        if transform_lbl is not None:
          X_lbl = transform_lbl(X_lbl)
        X_lbl = X_lbl * 255.0
        if self.dataset_name=='ade_sorted_lbl':
            X_lbl += 1
            X_lbl[X_lbl==self.opt.label_nc] = 0 # 'unknown' is 0
        elif self.dataset_name == 'ade_indoor_lbl':
            X_lbl = X_lbl
        else:
            X_lbl[X_lbl == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc
        
        if 'weighted' in self.dataset_name:
            if Y==1:
                X_im = transforms.functional.crop(X_im, 0, 256, 768, 1536)
        if 'cityscapes' not in self.opt.dataset:
            X_im =transforms.functional.crop(X_im, j, i, size[0], size[1])

        if transform_im is not None:
            X_im = transform_im(X_im)

        Y = torch.Tensor(np.array([Y]))
        return X_lbl, X_im, Y

    def __len__(self):
        return len(self.X_paths_im)

    def name(self):
        return 'SupDataset'


class CustomDataset(BaseDataset):
    #taking sample from a custom dataset
    def __init__(self, opt, transform=None):
        print(opt.dataset)
        inpath = 'datasets/%s'%opt.dataset
        if opt.train:
            self.datalist = "%s_info_train.txt"%inpath
        else:
            self.datalist = "%s_info_val.txt"%inpath
        with open(self.datalist) as f:
            # datalist is a txt file containing paths of images X, and their labels 
            self.info = f.readlines()
        self.X_paths = [x.strip().split(' ')[0] for x in self.info]
        self.Ys = [int(x.strip().split(' ')[1]) for x in self.info]
        print("number of images:", len(self.Ys))
        if opt.train:
            randinds = np.random.permutation(len(self.Ys))
        else:
            randinds = range(len(self.Ys))
        self.X_paths = [self.X_paths[i] for i in randinds]
        self.Ys = [int(self.Ys[i]) for i in randinds]
            # self.Ys = []
        self.transform = transform
        self.dataset_name = opt.dataset
        self.opt = opt
    def __getitem__(self, index):
        X_path = self.X_paths[index]
        Y = self.Ys[index]
        

        if 'lbl' in self.dataset_name:
            X = PIL.Image.open(X_path)
            if 'weighted' in self.dataset_name:
                if Y==1:
                    X = transforms.functional.crop(X, 0, 256, 768, 1536)
            X_ = np.asarray(X) 
            if self.transform is not None:
              X = self.transform(X)
            X = X * 255.0
            if self.dataset_name == 'ADE_indoor_lbl':
                X = X
            else:
                X[X == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc
        else:
            X = misc.imread(X_path)
            X = PIL.Image.fromarray(X.astype('uint8'))
            if 'weighted' in self.dataset_name:
                if Y==1:
                    X = transforms.functional.crop(X, 0, 256, 768, 1536)
            if self.transform is not None:
              X = self.transform(X)

        Y = torch.Tensor(np.array([Y]))
        return X,Y

    def __len__(self):
        return len(self.X_paths)

    def name(self):
        return 'CustomDataset'

