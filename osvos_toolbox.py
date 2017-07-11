from __future__ import division
import sys
sys.path.append("/home/eec/Documents/external/deep_learning/pytorch/build/lib.linux-x86_64-2.7")  # Custom PyTorch
import numpy as np
import cv2
from scipy.misc import imresize
import os
import random
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt


def im_normalize(im):
    """
    Normalize image
    """
    imn = (im - im.min()) / (im.max() - im.min())
    return imn


def construct_name(p, prefix):
    """
    Construct the name of the model
    p: dictionary of parameters
    prefix: the prefix
    name: the name of the model - manually add ".pth" to follow the convention
    """
    name = prefix
    for key in p.keys():
        if (type(p[key]) != tuple) and (type(p[key]) != list):
            name = name + '_' + str(key) + '-' + str(p[key])
        else:
            name = name + '_' + str(key) + '-' + str(p[key][0])
    return name


def overlay_mask(img, mask, transparency=0.5):
    """
    Overlay a h x w x 3 mask to the image
    img: h x w x 3 image
    mask: h x w x 3 mask
    transparency: between 0 and 1
    """
    im_over = np.ndarray(img.shape)
    im_over[:, :, 0] = (1 - mask[:, :, 0]) * img[:, :, 0] + mask[:, :, 0] * (
    255 * transparency + (1 - transparency) * img[:, :, 0])
    im_over[:, :, 1] = (1 - mask[:, :, 1]) * img[:, :, 1] + mask[:, :, 1] * (
    255 * transparency + (1 - transparency) * img[:, :, 1])
    im_over[:, :, 2] = (1 - mask[:, :, 2]) * img[:, :, 2] + mask[:, :, 2] * (
    255 * transparency + (1 - transparency) * img[:, :, 2])
    return im_over


class DAVISDataset(Dataset):
    """Tool Dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 inputRes=None,
                 db_root_dir='/media/eec/external/Databases/Segmentation/DAVIS',
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892)):
        """Loads image to label pairs for tool pose estimation
        db_elements: the names of the video files
        db_root_dir: dataset directory with subfolders "frames" and "Annotations"
        """
        self.train = train
        self.inputRes = inputRes
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.meanval = meanval

        if self.train:
            fname = 'train'
        else:
            fname = 'val'

        with open(os.path.join(db_root_dir, fname + '.txt')) as f:
            names = f.readlines()
            img_list = ['JPEGImages/480p/' + x.strip() + '.jpg' for x in names]
            labels = ['Annotations/480p/' + x.strip() + '.png' for x in names]

        assert (len(labels) == len(img_list))

        self.img_list = img_list  # [:1]
        self.labels = labels  # [:1]

        print('Done initializing ' + fname + ' Dataset')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # print(idx)
        img, gt = self.make_img_gt_pair(idx)

        sample = {'image': img, 'gt': gt}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx]))
        label = cv2.imread(os.path.join(self.db_root_dir, self.labels[idx]), 0)

        if self.inputRes is not None:
            # inputRes = list(reversed(self.inputRes))
            img = imresize(img, self.inputRes)
            label = imresize(label, self.inputRes, interp='nearest')

        img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))

        gt = np.array(label, dtype=np.float32)
        gt = gt/np.max([gt.max(), 1e-8])

        return img, gt

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]))

        return list(img.shape[:2])


class ScaleNRotate(object):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    Args:
        maxRot (float): maximum rotation angle to be added
        maxScale (float): maximum scale to be added
    """
    def __init__(self, rots=(-30, 30), scales=(.75, 1.25)):
        self.rots = rots
        self.scales = scales

    def __call__(self, sample):

        rot = (self.rots[1] - self.rots[0]) * random.random() - \
              (self.rots[1] - self.rots[0])/2

        sc = (self.scales[1] - self.scales[0]) * random.random() - \
             (self.scales[1] - self.scales[0]) / 2 + 1

        img, gt = sample['image'], sample['gt']
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, rot, sc)
        img_ = cv2.warpAffine(img, M, (w, h))

        h_gt, w_gt = gt.shape[:2]
        center_gt = (w_gt / 2, h_gt / 2)
        M = cv2.getRotationMatrix2D(center_gt, rot, sc)
        gt_ = cv2.warpAffine(gt, M, (w_gt, h_gt), flags=cv2.INTER_NEAREST)
        gt_ = gt_/np.max([gt_.max(), 1e-8])

        return {'image': img_, 'gt': gt_}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']

        if len(gt.shape) == 2:
            gt = gt[:, :, np.newaxis]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        gt = gt.transpose((2, 0, 1))
        # print(image.shape)
        # print(gt.shape)
        sys.stdout.flush()

        return {'image': torch.from_numpy(image),
                'gt': torch.from_numpy(gt)}


if __name__ == 'main':

    a = DAVISDataset(train=True, transform=ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)))
    # a = DAVISDataset(train=True)


    b = a[77]
    plt.imshow(im_normalize(b['image']))
    plt.imshow(b['gt'])
    b['gt'].max()
    np.unique(b['gt'])
