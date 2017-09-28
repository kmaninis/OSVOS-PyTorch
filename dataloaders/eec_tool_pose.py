from __future__ import division
from PIL import Image
import scipy.io as sio
import os
import sys
import json

from mypath import Path
if Path.is_custom_pytorch():
    sys.path.append(Path.custom_pytorch())  # Custom PyTorch
if Path.is_custom_opencv():
    sys.path.insert(0, Path.custom_opencv())

from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from helpers import *


class ToolDataset(Dataset):
    """Tool Dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 inputRes=(256, 512),
                 outputRes=(64, 128),
                 sigma=5,
                 db_elements=('1', '2', '3'),
                 db_root_dir='/media/eec/external/Databases/EEC/Tool/',
                 transform=None,
                 split_perc=None):
        """Loads image to label pairs for tool pose estimation
        db_elements: the names of the video files
        db_root_dir: dataset directory with subfolders "frames" and "Annotations"
        """
        self.train = train
        self.db_root_dir = db_root_dir
        self.inputRes = inputRes
        self.outputRes = outputRes
        self.sigma = sigma
        self.transform = transform
        labels = []
        img_list = []
        for ii in range(0, len(db_elements)):

            # for 'right' and 'left' sequences
            if '-left' in db_elements[ii]:
                chk_lr = ('left',)
                seq = db_elements[ii].replace('-left', '')
            elif '-right' in db_elements[ii]:
                chk_lr = ('right',)
                seq = db_elements[ii].replace('-right', '')
            else:
                chk_lr = ('left', 'right')
                seq = db_elements[ii]

            for lr in chk_lr:

                # check if the Annotation file exists
                anno_fname = os.path.join(db_root_dir, 'Annotations', seq + '-' + lr + '.json')
                if os.path.isfile(anno_fname):

                    # Read Annotations
                    with open(anno_fname) as anno_file:
                        temp_labels = json.load(anno_file)
                        labels = labels + temp_labels

                    # Read Image names
                    temp_img_list = []
                    for f in sorted(os.listdir(os.path.join(db_root_dir, 'frames', seq, lr))):
                        if f.endswith(".png"):
                            temp_img_list = temp_img_list + [os.path.join(seq, lr, f)]
                    img_list = img_list + temp_img_list

                    assert (len(temp_labels) == len(temp_img_list))

        self.img_list = img_list
        self.labels = labels
        if self.train:
            if split_perc is not None:
                split_ind = int(round(split_perc * len(img_list)))
                self.img_list = img_list[:split_ind]
                self.labels = labels[:split_ind]
            print('Done initializing Training Dataset, with ' + str(len(self.img_list)) + ' images')
        else:
            if split_perc is not None:
                split_ind = int(round(split_perc * len(img_list)))
                self.img_list = img_list[split_ind:]
                self.labels = labels[split_ind:]
            print('Done initializing Testing Dataset, with ' + str(len(self.img_list)) + ' images')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img, gt = self.make_img_gt_pair(idx)

        sample = {'image': img, 'gt': gt}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        img = Image.open(os.path.join(self.db_root_dir, 'frames', self.img_list[idx]))
        img_orig_shape = img.size
        label = jsonline2mat(self.labels[idx])
        if self.inputRes is not None:
            inputRes = list(reversed(self.inputRes))
            img = img.resize(inputRes)
            label[:, 0] = np.round(label[:, 0] * self.outputRes[1] / float(img_orig_shape[0]))
            label[:, 1] = np.round(label[:, 1] * self.outputRes[0] / float(img_orig_shape[1]))

        img = np.array(img, dtype=np.float32)
        gt = self.make_gt(img, label)

        if gt.shape[2] == 0:
            randidx = random.randint(1, len(self.img_list) - 1)
            img, gt = self.make_img_gt_pair(randidx)

        return img, gt

    def make_gt(self, img, labels):
        """ Make the ground-truth for each landmark.
        img: the original color image
        labels: the json labels with the Gaussian centers {'x': x, 'y': y}
        sigma: sigma of the Gaussian.
        """

        if self.outputRes is not None:
            h, w = self.outputRes
        else:
            h, w = img.shape
        # print (h, w, len(labels))
        gt = np.zeros((int(h), int(w), len(labels)), np.float32)

        for land in range(0, labels.shape[0]):
            gt[:, :, land] = (make_gaussian((h, w), self.sigma, (labels[land, 0], labels[land, 1])))
        return gt
    
    def store_gt_asmatfile(self):
        gt_tool = np.zeros((2, 3, len(self.img_list)), dtype=np.float32)
        for i in range(0, len(self.img_list)):
            temp = jsonline2mat(self.labels[i])
            if temp.shape[0] == 0:
                gt_tool[:, :, i] = np.nan
            else:
                gt_tool[:, :, i] = np.transpose(temp)

        a = {'gt_tool': gt_tool}
        sio.savemat('gt_tool', a)

    def get_img_size(self, idx=0):
        img = Image.open(os.path.join(self.db_root_dir, 'frames', self.img_list[idx]))

        return list(reversed(img.size))


if __name__ == '__main__':
    import custom_transforms as tr
    import torch
    from torchvision import transforms

    transforms = transforms.Compose([tr.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)), tr.ToTensor()])
    dataset = ToolDataset(train=True, transform=transforms, inputRes=(512, 1024), outputRes=(512, 1024))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    for i, data in enumerate(dataloader):
        plt.figure()
        plt.imshow(overlay_mask_tool(tens2image(data['image']), tens2image(data['gt'])))
        if i == 10:
            break

    plt.show(block=True)
