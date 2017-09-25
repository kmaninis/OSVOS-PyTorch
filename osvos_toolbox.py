from __future__ import division
import sys
from mypath import Path
if Path.is_custom_pytorch():
    sys.path.append(Path.custom_pytorch())  # Custom PyTorch
if Path.is_custom_opencv():
    sys.path.insert(0, Path.custom_opencv())
import numpy as np
import cv2
from scipy.misc import imresize
import os
import random
import torch
from torch.utils.data import Dataset


def im_normalize(im):
    """
    Normalize image
    """
    imn = (im - im.min()) / max(im.max() - im.min(), 1e-8)
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
    im_over[:, :, 0] = (1 - mask) * img[:, :, 0] + mask * (transparency + (1 - transparency) * img[:, :, 0])
    im_over[:, :, 1] = (1 - mask) * img[:, :, 1] + mask * (transparency + (1 - transparency) * img[:, :, 1])
    im_over[:, :, 2] = (1 - mask) * img[:, :, 2] + mask * (transparency + (1 - transparency) * img[:, :, 2])

    return im_over


class DAVISDataset(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 inputRes=None,
                 db_root_dir='/media/eec/external/Databases/Segmentation/DAVIS',
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892),
                 seq_name=None):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """
        self.train = train
        self.inputRes = inputRes
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.meanval = meanval
        self.seq_name = seq_name

        if self.train:
            fname = 'train_seqs'
        else:
            fname = 'val_seqs'

        if self.seq_name is None:

            # Initialize the original DAVIS splits for training the parent network
            with open(os.path.join(db_root_dir, fname + '.txt')) as f:
                seqs = f.readlines()
                img_list = []
                labels = []
                for seq in seqs:
                    images = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', seq.strip())))
                    images_path = map(lambda x: os.path.join('JPEGImages/480p/', seq.strip(), x), images)
                    img_list.extend(images_path)
                    lab = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', seq.strip())))
                    lab_path = map(lambda x: os.path.join('Annotations/480p/', seq.strip(), x), lab)
                    labels.extend(lab_path)
        else:

            # Initialize the per sequence images for online training
            names_img = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', str(seq_name))))
            img_list = map(lambda x: os.path.join('JPEGImages/480p/', str(seq_name), x), names_img)
            name_label = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', str(seq_name))))
            labels = [os.path.join('Annotations/480p/', str(seq_name), name_label[0])]
            labels.extend([None]*(len(names_img)-1))
            if self.train:
                img_list = [img_list[0]]
                labels = [labels[0]]

        assert (len(labels) == len(img_list))

        self.img_list = img_list  # [0:10]
        self.labels = labels  # [0:10]

        print('Done initializing ' + fname + ' Dataset')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # print(idx)
        img, gt = self.make_img_gt_pair(idx)

        sample = {'image': img, 'gt': gt}

        if self.seq_name is not None:
            fname = os.path.join(self.seq_name, "%05d" % idx)
            sample['fname'] = fname

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx]))
        if self.labels[idx] is not None:
            label = cv2.imread(os.path.join(self.db_root_dir, self.labels[idx]), 0)
        else:
            gt = np.zeros(img.shape[:-1], dtype=np.uint8)

        if self.inputRes is not None:
            # inputRes = list(reversed(self.inputRes))
            img = imresize(img, self.inputRes)
            if self.labels[idx] is not None:
                label = imresize(label, self.inputRes, interp='nearest')

        img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))

        if self.labels[idx] is not None:
                gt = np.array(label, dtype=np.float32)
                gt = gt/np.max([gt.max(), 1e-8])

        return img, gt

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]))

        return list(img.shape[:2])


class ScaleNRotate(object):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    Args:
        two possibilities:
        1.  rots (tuple): (minimum, maximum) rotation angle
            scales (tuple): (minimum, maximum) scale
        2.  rots [list]: list of fixed possible rotation angles
            scales [list]: list of fixed possible scales
    """
    def __init__(self, rots=(-30, 30), scales=(.75, 1.25)):
        assert (isinstance(rots, type(scales)))
        self.rots = rots
        self.scales = scales

    def __call__(self, sample):

        if type(self.rots) == tuple:
            # Continuous range of scales and rotations
            rot = (self.rots[1] - self.rots[0]) * random.random() - \
                  (self.rots[1] - self.rots[0])/2

            sc = (self.scales[1] - self.scales[0]) * random.random() - \
                 (self.scales[1] - self.scales[0]) / 2 + 1
        elif type(self.rots) == list:
            # Fixed range of scales and rotations
            rot = self.rots[random.randint(0, len(self.rots)-1)]
            sc = self.scales[random.randint(0, len(self.scales) - 1)]

        img, gt = sample['image'], sample['gt']
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, rot, sc)
        img_ = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
        # plt.imshow(img_)

        h_gt, w_gt = gt.shape[:2]
        center_gt = (w_gt / 2, h_gt / 2)
        M = cv2.getRotationMatrix2D(center_gt, rot, sc)
        gt_ = cv2.warpAffine(gt, M, (w_gt, h_gt), flags=cv2.INTER_NEAREST)
        gt_ = gt_/np.max([gt_.max(), 1e-8])

        sample['image'], sample['gt'] = img_, gt_

        return sample


class Resize(object):
    """Randomly resize the image and the ground truth to specified scales.
    Args:
        scales (list): the list of scales
    """
    def __init__(self, scales=[0.5, 0.8, 1]):
        self.scales = scales

    def __call__(self, sample):

        # Fixed range of scales
        sc = self.scales[random.randint(0, len(self.scales) - 1)]

        img, gt = sample['image'], sample['gt']
        img_ = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        gt_ = cv2.resize(gt, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
        sample['image'], sample['gt'] = img_, gt_

        return sample


class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        image, gt = sample['image'], sample['gt']

        if random.random() < 0.5:
            image = cv2.flip(image, flipCode=1)
            gt = cv2.flip(gt, flipCode=1)

        sample['image'], sample['gt'] = image, gt

        return sample


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
        # sys.stdout.flush()

        sample['image'], sample['gt'] = torch.from_numpy(image), torch.from_numpy(gt)

        return sample


if __name__ == '__main__':

    from torchvision import transforms
    from matplotlib import pyplot as plt

    # transforms = transforms.Compose([RandomHorizontalFlip(),
    #                                  ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25))])
    transforms = transforms.Compose([RandomHorizontalFlip(),
                                     Resize(scales=[0.5, 0.8, 1])])

    db = DAVISDataset(train=True, transform=transforms,
                      seq_name='blackswan')

    sample = db[0]
    plt.imshow(overlay_mask(im_normalize(sample['image']), sample['gt']))

    print('Maximum value of gt: ' + str(sample['gt'].max()))
    print('Unique values of gt: ')
    print(np.unique(sample['gt']))
