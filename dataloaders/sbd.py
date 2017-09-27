import errno
import hashlib
import os
import sys
import tarfile

import numpy as np
import json
import scipy.io

import torch.utils.data as data
from PIL import Image

from six.moves import urllib


class SBDSegmentation(data.Dataset):

    URL = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz"
    FILE = "benchmark.tgz"
    MD5 = '82b4d87ceb2ed10f6038a1cba92111cb'

    def __init__(self,
                 root,
                 split='val',
                 transform=None,
                 target_transform=None,
                 download=False,
                 preprocess=False):

        # Store parameters
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split

        # Where to find things according to the author's structure
        self.dataset_dir = os.path.join(self.root, 'benchmark_RELEASE', 'dataset')
        _mask_dir = os.path.join(self.dataset_dir, 'inst')
        _image_dir = os.path.join(self.dataset_dir, 'img')

        # Download dataset?
        if download:
            self._download()
            if not self._check_integrity():
                raise RuntimeError('Dataset file downloaded is corrupted.')

        # Get list of all images from the split and check that the files exist
        self.im_ids = []
        self.images = []
        self.masks = []
        with open(os.path.join(self.dataset_dir, split+'.txt'), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n') + ".jpg")
                _mask = os.path.join(_mask_dir, line.rstrip('\n') + ".mat")
                assert os.path.isfile(_image)
                assert os.path.isfile(_mask)
                self.im_ids.append(line.rstrip('\n'))
                self.images.append(_image)
                self.masks.append(_mask)

        assert (len(self.images) == len(self.masks))

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
        _mask = scipy.io.loadmat(self.masks[_im_ii])["GTinst"][0]["Segmentation"][0]
        _target = (_mask == (_obj_ii+1)).astype(np.float).reshape([_img.size[1], _img.size[0], 1])

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
        # Check that the file with categories is there and with correct size
        _obj_list_file = os.path.join(self.dataset_dir, self.split+'_instances.txt')
        if not os.path.isfile(_obj_list_file):
            return False
        else:
            self.obj_dict = json.load(open(_obj_list_file, 'r'))
            return list(self.obj_dict.keys()) == self.im_ids

    def _preprocess(self):
        # Get all object instances and their category
        self.obj_dict = {}
        for ii in range(len(self.im_ids)):
            # Read object masks and get number of objects
            tmp = scipy.io.loadmat(self.masks[ii])
            _mask = tmp["GTinst"][0]["Segmentation"][0]
            _cat_ids = tmp["GTinst"][0]["Categories"][0]

            _mask_ids = np.unique(_mask)
            n_obj = _mask_ids[-1]
            assert(n_obj == len(_cat_ids))

            self.obj_dict[self.im_ids[ii]] = np.squeeze(_cat_ids, 1).tolist()

        # Save it to file for future reference
        with open(os.path.join(self.dataset_dir, self.split+'_instances.txt'), 'w') as outfile:
            outfile.write('{{\n\t"{:s}": {:s}'.format(self.im_ids[0], json.dumps(self.obj_dict[self.im_ids[0]])))
            for ii in range(1, len(self.im_ids)):
                outfile.write(',\n\t"{:s}": {:s}'.format(self.im_ids[ii], json.dumps(self.obj_dict[self.im_ids[ii]])))
            outfile.write('\n}\n')

        print('Pre-processing finished')

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
    import matplotlib.pyplot as plt
    import helpers
    import torch
    import torchvision.transforms as transforms
    transform = transforms.ToTensor()
    dataset = SBDSegmentation('/Users/jpont/Workspace/gt_dbs/SBD', split='train', transform=transform, target_transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    for i, data in enumerate(dataloader):
        plt.figure()
        plt.imshow(helpers.overlay_mask(helpers.tens2image(data[0]), helpers.tens2image(data[1])))
        if i == 10:
            break

    plt.show(block=True)
