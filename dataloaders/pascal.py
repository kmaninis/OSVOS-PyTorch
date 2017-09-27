import errno
import hashlib
import os
import sys
import tarfile

import numpy as np
import matplotlib.pyplot as plt
import json

import torch.utils.data as data
from PIL import Image

from six.moves import urllib


class VOCSegmentation(data.Dataset):
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv/monitor', 'ambigious'
    ]

    URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    FILE = "VOCtrainval_11-May-2012.tar"
    MD5 = '6cd6e144f989b92b3379bac3b3de84fd'
    BASE_DIR = 'VOCdevkit/VOC2012'

    def __init__(self,
                 root,
                 split='val',
                 transform=None,
                 target_transform=None,
                 download=False,
                 preprocess=False):

        self.root = root
        _voc_root = os.path.join(self.root, self.BASE_DIR)
        _mask_dir = os.path.join(_voc_root, 'SegmentationObject')
        _cat_dir = os.path.join(_voc_root, 'SegmentationClass')
        _image_dir = os.path.join(_voc_root, 'JPEGImages')
        self.transform = transform
        self.target_transform = target_transform
        self.split = split

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_voc_root, 'ImageSets', 'Segmentation')
        _split_f = os.path.join(_splits_dir, split+'.txt')

        self.im_ids = []
        self.images = []
        self.categories = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n') + ".jpg")
                _cat = os.path.join(_cat_dir, line.rstrip('\n') + ".png")
                _mask = os.path.join(_mask_dir, line.rstrip('\n') + ".png")
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
                self.obj_list.append([ii, jj])

        # Display stats
        print('Number of images: {:d}\nNumber of objects: {:d}'.format(len(self.im_ids), len(self.obj_list)))

    def __getitem__(self, index):
        _im_ii = self.obj_list[index][0]
        _obj_ii = self.obj_list[index][1]

        _img = Image.open(self.images[_im_ii]).convert('RGB')
        _target = (np.array(Image.open(self.masks[_im_ii])) == (_obj_ii+1)).astype(np.float).reshape([_img.size[1], _img.size[0], 1])

        if self.transform is not None:
            _img = self.transform(_img)
        if self.target_transform is not None:
            _target = self.target_transform(_target)

        return _img, _target

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
        _obj_list_file = os.path.join(self.root, self.BASE_DIR, 'ImageSets', 'Segmentation', self.split+'_instances.txt')
        if not os.path.isfile(_obj_list_file):
            return False
        else:
            self.obj_dict = json.load(open(_obj_list_file, 'r'))
            return list(self.obj_dict.keys()) == self.im_ids

    def _preprocess(self):
        self.obj_dict = {}
        for ii in range(len(self.im_ids)):
            # Read object masks and get number of objects
            _mask = np.array(Image.open(self.masks[ii]))
            _mask_ids = np.unique(_mask)
            if _mask_ids[-1] == 255:
                n_obj = _mask_ids[-2]
            else:
                n_obj = _mask_ids[-1]

            # Get the categories from these objects
            _cats = np.array(Image.open(self.categories[ii]))
            _cat_ids = []
            for jj in range(n_obj):
                tmp = np.where(_mask == jj+1)
                _cat_ids.append(int(_cats[tmp[0][0], tmp[1][0]]))

            self.obj_dict[self.im_ids[ii]] = _cat_ids

        with open(os.path.join(self.root, self.BASE_DIR, 'ImageSets', 'Segmentation', self.split+'_instances.txt'), 'w') as outfile:
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


if __name__ == '__main__':
    import helpers
    import torch
    import torchvision.transforms as transforms
    transform = transforms.ToTensor()
    dataset = VOCSegmentation('/Users/jpont/Workspace/gt_dbs/Pascal/', split='trainval', transform=transform, target_transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    for i, data in enumerate(dataloader):
        plt.figure()
        plt.imshow(helpers.overlay_mask(helpers.tens2image(data[0]), helpers.tens2image(data[1])))
        if i == 10:
            break

    plt.show(block=True)
