# Package Includes
from __future__ import division
# import matlab.engine
import sys
import os
import socket
import timeit
if 'experiments' in os.getcwd():
    sys.path.append('../../OSVOS-PyTorch')
else:
    sys.path.append('OSVOS-PyTorch')

from mypath import Path

if Path.is_custom_pytorch():
    sys.path.append(Path.custom_pytorch())  # Custom PyTorch
if Path.is_custom_opencv():
    sys.path.insert(0, Path.custom_opencv())

# Custom includes
import visualize as viz
import osvos_toolbox as tb
import Hourglass as nt
from custom_layers import class_balanced_cross_entropy_loss
import numpy as np
import scipy.misc as sm
from datetime import datetime

# PyTorch includes
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

# Tensorboard include
from tensorboardX import SummaryWriter

# Select which GPU, -1 if CPU
if socket.gethostname() == 'eec':
    gpu_id = 1
elif 'SGE_GPU' not in os.environ.keys() and socket.gethostname() != 'reinhold':
    gpu_id = -1
else:
    gpu_id = int(os.environ['SGE_GPU'])

print('Using GPU: {} '.format(gpu_id))

# Setting of parameters
# Parameters in p are used for the name of the model
p = {
    'inputRes': (512, 896),  # Input Resolution
    'trainBatch': 3,  # Number of Images in each mini-batch
    'numHG': 4,  # Number of Stacked Hourglasses
    'Block': 'ConvBlock',  # Select: 'ConvBlock', 'BasicBlock', 'BottleNeck'
    }

# # Setting other parameters
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
nEpochs = 240  # Number of epochs for training (500.000/2079)
numHGScales = 4  # How many times to downsample inside each HourGlass
useTest = 1  # See evolution of the test set when training?
testBatch = 1  # Testing Batch
nTestInterval = 5  # Run on test set every nTestInterval epochs
db_root_dir = Path.db_root_dir()
save_dir_root = Path.save_root_dir()

if 'experiments' in os.getcwd():
    save_dir = os.path.join(save_dir_root, 'experiments', exp_name)
else:
    save_dir = './models'

if not os.path.exists(save_dir):
    os.makedirs(os.path.join(save_dir))
vis_net = 0  # Visualize the network?
snapshot = 40  # Store a model every snapshot epochs
nAveGrad = 10
side_supervision = [1.0] * 72
side_supervision.extend([0.5] * 72)
side_supervision.extend([0.0] * 96)
load_caffe_vgg = 1
resume_epoch = 0  # Default is 0, change if want to resume

# Network definition
modelName = exp_name  # tb.construct_name(p, "OSVOS_parent_exact")
if resume_epoch == 0:
    net = nt.Net_SHG(p['numHG'], numHGScales, p['Block'], 128, 1)
else:
    net = nt.Net_SHG(p['numHG'], numHGScales, p['Block'], 128, 1)
    print("Updating weights from: {}".format(
        os.path.join(save_dir, modelName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
    net.load_state_dict(
        torch.load(os.path.join(save_dir, modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'),
                   map_location=lambda storage, loc: storage))

# Logging into Tensorboard
log_dir = os.path.join(save_dir, 'runs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
writer = SummaryWriter(log_dir=log_dir, comment='-parent')

# Visualize the network
if vis_net:
    x = torch.randn(1, 3, 512, 896)
    x = Variable(x)
    y = net.forward(x)
    g = viz.make_dot(y, net.state_dict())
    g.view()

if gpu_id >= 0:
    torch.cuda.set_device(device=gpu_id)
    net.cuda()

# Use the following optimizer
lr = 1e-5
optimizer = optim.RMSprop(net.parameters(), lr=lr, alpha=0.99, momentum=0.0)

# Preparation of the data loaders
# Define augmentation transformations as a composition
composed_transforms = transforms.Compose([tb.RandomHorizontalFlip(),
                                          # tb.Resize(),
                                          tb.ScaleNRotate(rots=(-20, 20), scales=(0.75, 1.25)),
                                          tb.ToTensor()])
# Training dataset and its iterator
db_train = tb.DAVISDataset(train=True, inputRes=p['inputRes'], db_root_dir=db_root_dir, transform=composed_transforms)
trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=1)

# Testing dataset and its iterator
db_test = tb.DAVISDataset(train=False, inputRes=p['inputRes'], db_root_dir=db_root_dir, transform=tb.ToTensor())
testloader = DataLoader(db_test, batch_size=testBatch, shuffle=False, num_workers=1)

num_img_tr = len(trainloader)
num_img_ts = len(testloader)
running_loss_tr = [0] * p['numHG']
running_loss_ts = [0] * p['numHG']
loss_tr = []
loss_ts = []
aveGrad = 0

print("Training Network")
# Main Training and Testing Loop
for epoch in range(resume_epoch, nEpochs):
    start_time = timeit.default_timer()
    # Adjust the learning rate

    # One training epoch
    for ii, sample_batched in enumerate(trainloader):

        inputs, gts = sample_batched['image'], sample_batched['gt']

        # Forward-Backward of the mini-batch
        inputs, gts = Variable(inputs), Variable(gts)
        if gpu_id >= 0:
            inputs, gts = inputs.cuda(), gts.cuda()

        outputs = net.forward(inputs)

        # Compute the losses, side outputs and fuse
        losses = [0] * len(outputs)
        for i in range(0, len(outputs)):
            losses[i] = class_balanced_cross_entropy_loss(outputs[i], gts, size_average=False)
            running_loss_tr[i] += losses[i].data[0]
        loss = side_supervision[epoch] * sum(losses[:-1]) + losses[-1]

        # Print stuff
        if ii % num_img_tr == num_img_tr - 1:
            running_loss_tr = [x / num_img_tr for x in running_loss_tr]
            loss_tr.append(running_loss_tr[-1])
            writer.add_scalar('data/total_loss_epoch', running_loss_tr[-1], epoch)
            print('[Epoch: %d, numImages: %5d]' % (epoch, ii + 1))
            for l in range(0, len(running_loss_tr)):
                print('Loss %d: %f' % (l, running_loss_tr[l]))
                running_loss_tr[l] = 0

            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time))

        # Backward the averaged gradient
        loss /= nAveGrad
        loss.backward()
        aveGrad += 1

        # Update the weights once in nAveGrad forward passes
        if aveGrad % nAveGrad == 0:
            writer.add_scalar('data/total_loss_iter', loss.data[0], ii + num_img_tr * epoch)
            optimizer.step()
            optimizer.zero_grad()
            aveGrad = 0

    # Save the model
    if (epoch % snapshot) == snapshot - 1 and epoch != 0:
        torch.save(net.state_dict(), os.path.join(save_dir, modelName + '_epoch-' + str(epoch) + '.pth'))

    # One testing epoch
    if useTest and epoch % nTestInterval == (nTestInterval - 1):
        for ii, sample_batched in enumerate(testloader):
            inputs, gts = sample_batched['image'], sample_batched['gt']

            # Forward pass of the mini-batch
            inputs, gts = Variable(inputs, volatile=True), Variable(gts, volatile=True)
            if gpu_id >= 0:
                inputs, gts = inputs.cuda(), gts.cuda()

            outputs = net.forward(inputs)

            # Compute the losses, side outputs and fuse
            losses = [0] * len(outputs)
            for i in range(0, len(outputs)):
                losses[i] = class_balanced_cross_entropy_loss(outputs[i], gts, size_average=False)
                running_loss_ts[i] += losses[i].data[0]
            loss = side_supervision[epoch] * sum(losses[:-1]) + losses[-1]

            # Print stuff
            if ii % num_img_ts == num_img_ts - 1:
                running_loss_ts = [x / num_img_ts for x in running_loss_ts]
                loss_ts.append(running_loss_ts[-1])

                print('[Epoch: %d, numImages: %5d]' % (epoch, ii + 1))
                writer.add_scalar('data/test_loss_epoch', running_loss_ts[-1], epoch)
                for l in range(0, len(running_loss_ts)):
                    print('***Testing *** Loss %d: %f' % (l, running_loss_ts[l]))
                    running_loss_ts[l] = 0

writer.close()

# Test parent network
net = nt.Net_SHG(p['numHG'], numHGScales, p['Block'], 128, 1)
parentModelName = exp_name  # tb.construct_name(p, 'OSVOS_parent_exact')
net.load_state_dict(torch.load(os.path.join(save_dir, parentModelName + '_epoch-' + str(nEpochs-1) + '.pth'),
                               map_location=lambda storage, loc: storage))
with open(os.path.join(Path.db_root_dir(), 'val_seqs.txt'), 'r') as f:
    seqs = f.readlines()
seqs = map(lambda seq: seq.strip(), seqs)
for seq_name in seqs:
    # Testing dataset and its iterator
    db_test = tb.DAVISDataset(train=False, db_root_dir=db_root_dir, transform=tb.ToTensor(), seq_name=seq_name)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=2)

    save_dir_seq = os.path.join(save_dir, parentModelName, seq_name)
    if not os.path.exists(save_dir_seq):
        os.makedirs(save_dir_seq)

    print('Testing Network')
    # Main Testing Loop
    for ii, sample_batched in enumerate(testloader):

        img, gt, fname = sample_batched['image'], sample_batched['gt'], sample_batched['fname']

        # Forward of the mini-batch
        inputs, gts = Variable(img, volatile=True), Variable(gt, volatile=True)
        if gpu_id >= 0:
            inputs, gts = inputs.cuda(), gts.cuda()

        outputs = net.forward(inputs)

        for jj in range(int(inputs.size()[0])):
            pred = np.transpose(outputs[-1].cpu().data.numpy()[jj, :, :, :], (1, 2, 0))
            pred = 1 / (1 + np.exp(-pred))
            pred = np.squeeze(pred)
            img = sm.imresize(pred, (480, 854))
            img_ = np.transpose(img.numpy()[jj, :, :, :], (1, 2, 0))
            gt_ = np.transpose(gt.numpy()[jj, :, :, :], (1, 2, 0))
            gt_ = np.squeeze(gt)

            # Save the result, attention to the index jj
            sm.imsave(os.path.join(save_dir_seq, os.path.basename(fname[jj]) + '.png'), pred)
