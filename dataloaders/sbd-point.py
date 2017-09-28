from __future__ import print_function, division

import errno
import hashlib
import os
import sys
import tarfile
from helpers import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import scipy.io

from mypath import Path
if Path.is_custom_pytorch():
    sys.path.append(Path.custom_pytorch())  # Custom PyTorch
if Path.is_custom_opencv():
    sys.path.insert(0, Path.custom_opencv())

import torch.utils.data as data
from PIL import Image
from six.moves import urllib


class SBDSegmentationPoint(data.Dataset):
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv/monitor', 'ambigious'
    ]

    URL = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz"
    FILE = "benchmark.tgz"
    MD5 = '82b4d87ceb2ed10f6038a1cba92111cb'

    def __init__(self,
                 root='/media/eec/external/Databases/Segmentation/PASCAL',
                 split='val',
                 train=True,
                 inputRes=None,
                 outputRes=None,
                 sigma=10,
                 transform=None,
                 download=False,
                 preprocess=False,
                 area_thres=0):

        self.root = root
        self.dataset_dir = os.path.join(self.root, 'benchmark_RELEASE','dataset')
        _mask_dir = os.path.join(self.dataset_dir, 'inst')
        _cat_dir = os.path.join(self.dataset_dir, 'cls')
        _image_dir = os.path.join(self.dataset_dir, 'img')
        self.transform = transform
        self.split = split
        self.train = train
        self.sigma = sigma
        self.inputRes = inputRes
        self.outputRes = outputRes
        self.area_thres = area_thres

        if self.area_thres == 0:
            self.fname = os.path.join(self.dataset_dir, self.split + '_instances.txt')
        else:
            self.fname = os.path.join(self.dataset_dir, self.split + '_instances_area_thres-' + str(area_thres) + '.txt')

        if download:
            self._download()

        # if not self._check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You can use download=True to download it')

        # train/val/test splits are pre-cut
        _splits_dir = self.dataset_dir
        _split_f = os.path.join(_splits_dir, split+'.txt')

        self.im_ids = []
        self.images = []
        self.categories = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n') + ".jpg")
                _cat = os.path.join(_cat_dir, line.rstrip('\n') + ".mat")
                _mask = os.path.join(_mask_dir, line.rstrip('\n') + ".mat")
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                assert os.path.isfile(_mask)
                self.im_ids.append(line.rstrip('\n'))
                self.images.append(_image)
                self.categories.append(_cat)
                self.masks.append(_mask)

        assert (len(self.images) == len(self.masks))
        assert (len(self.images) == len(self.categories))

        # Precompute the list of objects and their categories for each image 
        if (not self._check_preprocess()) or preprocess:
            print('Preprocessing the dataset, this will take long, but it will be done only once.')
            self._preprocess()
            
        # Build the list of objects
        self.obj_list = []
        for ii in range(len(self.im_ids)):
            for jj in range(len(self.obj_dict[self.im_ids[ii]])):
                if self.obj_dict[self.im_ids[ii]][jj] != -1:
                    self.obj_list.append([ii, jj])

        # Display stats
        if self.train:
            print('Done initializing Training Dataset')
        else:
            print('Done initializing Testing Dataset')

        print('Number of images: {:d}\nNumber of objects: {:d}'.format(len(self.im_ids), len(self.obj_list)))

    def __getitem__(self, index):

        _img, _target, _heat_point = self._make_img_gt_point_pair(index)

        if self.inputRes is not None:
            _inputRes = tuple(reversed(self.inputRes))
            _img = cv2.resize(_img, _inputRes)
            _heat_point = cv2.resize(_heat_point, _inputRes, interpolation=cv2.INTER_NEAREST)

        if self.outputRes is not None:
            _outputRes = tuple(reversed(self.outputRes))
            _target = cv2.resize(_target, _outputRes)

        sample = {'image': _img, 'gt': _target, 'point': _heat_point}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.obj_list)

    def _check_integrity(self):
        _fpath = os.path.join(self.root, self.FILE)
        if not os.path.isfile(_fpath):
            print("{} does not exist".format(_fpath))
            return False
        _md5c = hashlib.md5(open(_fpath, 'rb').read()).hexdigest()
        if _md5c != self.MD5:
            print(" MD5({}) did not match MD5({}) expected for {}".format(
                _md5c, self.MD5, _fpath))
            return False
        return True

    def _check_preprocess(self):
        _obj_list_file = self.fname
        if not os.path.isfile(_obj_list_file):
            return False
        else:
            self.obj_dict = json.load(open(_obj_list_file, 'r'))
            return list(np.sort([str(x) for x in self.obj_dict.keys()])) == self.im_ids

    def _preprocess(self):
        self.obj_dict = {}
        for ii in range(len(self.im_ids)):
            # Read object masks and get number of objects
            tmp = scipy.io.loadmat(self.masks[ii])
            _mask = tmp["GTinst"][0]["Segmentation"][0]
            _cat_ids = tmp["GTinst"][0]["Categories"][0].astype(int)

            _mask_ids = np.unique(_mask)
            n_obj = _mask_ids[-1]
            assert(n_obj == len(_cat_ids))
            
            for jj in range(n_obj):
                temp = np.where(_mask == jj + 1)
                obj_area = len(temp[0])
                if obj_area < self.area_thres:
                    _cat_ids[jj] = -1
                    
            self.obj_dict[self.im_ids[ii]] = np.squeeze(_cat_ids, 1).tolist()

        with open(self.fname, 'w') as outfile:
            outfile.write('{{\n\t"{:s}": {:s}'.format(self.im_ids[0], json.dumps(self.obj_dict[self.im_ids[0]])))
            for ii in range(1, len(self.im_ids)):
                outfile.write(',\n\t"{:s}": {:s}'.format(self.im_ids[ii], json.dumps(self.obj_dict[self.im_ids[ii]])))
            outfile.write('\n}\n')

        print('Preprocessing finished')

    def _download(self):
        _fpath = os.path.join(self.root, self.FILE)

        try:
            os.makedirs(self.root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        else:
            print('Downloading ' + self.URL + ' to ' + _fpath)

            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> %s %.1f%%' %
                                 (_fpath, float(count * block_size) /
                                  float(total_size) * 100.0))
                sys.stdout.flush()

            urllib.request.urlretrieve(self.URL, _fpath, _progress)

        # extract file
        cwd = os.getcwd()
        print('Extracting tar file')
        tar = tarfile.open(_fpath)
        os.chdir(self.root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('Done!')

    def _make_img_gt_point_pair(self, index):
        _im_ii = self.obj_list[index][0]
        _obj_ii = self.obj_list[index][1]

        # Read Image
        _img = np.array(Image.open(self.images[_im_ii]).convert('RGB')).astype(np.float32) / 255

        # Read Taret object
        _mask = scipy.io.loadmat(self.masks[_im_ii])["GTinst"][0]["Segmentation"][0]
        _target = (_mask == (_obj_ii + 1)).astype(np.float32)

        # Construct point heatmap
        _point = point_in_segmentation(_target, .5)
        _heat_point = make_gt(_img, _point, sigma=self.sigma)

        return _img, _target, _heat_point

    def get_img_size(self, idx=0):
        img = Image.open(os.path.join(self.db_root_dir, 'JPEGImages', self.img_list[idx] + '.jpg'))
        return list(reversed(img.size))


if __name__ == '__main__':
    import custom_transforms as tr
    import torch
    from torchvision import transforms

    transforms = transforms.Compose([tr.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)), tr.ToTensor()])
    dataset = SBDSegmentationPoint(root='/media/eec/external/Databases/Segmentation/PASCAL', split='trainval',
                                   transform=transforms, inputRes=(512, 512), outputRes=(512, 512), area_thres=40000)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    for i, data in enumerate(dataloader):
        plt.figure()
        plt.imshow(overlay_mask(tens2image(data['image']), tens2image(data['gt'])))
        plt.figure()
        plt.imshow(overlay_mask(tens2image(data['image']), tens2image(data['point']) > .5))
        if i == 10:
            break

    plt.show(block=True)
