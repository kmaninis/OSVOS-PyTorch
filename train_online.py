# Package Includes
from __future__ import division
import sys
from mypath import Path
if Path.is_custom_pytorch():
    sys.path.append(Path.custom_pytorch())  # Custom PyTorch

import numpy as np
import os
import socket
import timeit

# Custom includes
import scipy.misc as sm
import visualize as viz
import osvos_toolbox as tb
import vgg_osvos as vo
from custom_layers import class_balanced_cross_entropy_loss
import scipy.io

# PyTorch includes
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import DataLoader

# Tensorboard include
from tensorboardX import SummaryWriter

# Setting of parameters
if 'SEQ_NAME' not in os.environ.keys():
    seq_name = 'drift-chicane'
else:
    seq_name = str(os.environ['SEQ_NAME'])

db_root_dir = Path.db_root_dir()
save_root_dir = Path.save_root_dir()
exp_dir = Path.exp_dir()
vis_net = 0  # Visualize the network?
vis_res = 0  # Visualize the results?
nAveGrad = 5
nEpochs = 2000 * nAveGrad  # Number of epochs for training
snapshot = nEpochs  # Store a model every snapshot epochs
parentEpoch = 240

# Parameters in p are used for the name of the model
p = {
    'trainBatch': 1,  # Number of Images in each mini-batch
    }

parentModelName = tb.construct_name(p, 'OSVOS_parent_exact')
if 'SGE_GPU' not in os.environ.keys() and socket.gethostname() != 'reinhold':
    gpu_id = 1  # Select which GPU, -1 if CPU
else:
    gpu_id = int(os.environ['SGE_GPU'])

# Network definition
net = vo.OSVOS(pretrained=0)
# net.load_state_dict(torch.load(os.path.join('models', parentModelName+'_epoch-'+str(parentEpoch)+'.pth'),
#                                map_location=lambda storage, loc: storage))

# Load weights from Caffe
caffe_weights = scipy.io.loadmat('/home/csergi/scratch/caffe_versions/OSVOS-caffe/parent_network_weights.mat')
caffe_forward = scipy.io.loadmat('/home/csergi/scratch/caffe_versions/OSVOS-caffe/drift-chicane_forward.mat')
# Core network
caffe_ind = 0
for ind, layer in enumerate(net.stages.parameters()):
    if ind % 2 == 0:
        c_w = torch.from_numpy(caffe_weights['valueSet_w'][0][caffe_ind].transpose())
        assert(layer.data.shape == c_w.shape)
        layer.data = c_w
    else:
        c_b = torch.from_numpy(caffe_weights['valueSet_b'][0][caffe_ind][:, 0])
        assert (layer.data.shape == c_b.shape)
        layer.data = c_b
        caffe_ind+=1
# 16 channel conversion
for ind, layer in enumerate(net.side_prep.parameters()):
    if ind % 2 == 0:
        c_w = torch.from_numpy(caffe_weights['valueSet_w'][0][caffe_ind].transpose())
        assert(layer.data.shape == c_w.shape)
        layer.data = c_w
    else:
        c_b = torch.from_numpy(caffe_weights['valueSet_b'][0][caffe_ind][:, 0])
        assert (layer.data.shape == c_b.shape)
        layer.data = c_b
        caffe_ind+=1

for ind, layer in enumerate(net.fuse.parameters()):
    if ind % 2 == 0:
        c_w = torch.from_numpy(caffe_weights['valueSet_w'][0][-1].transpose())
        c_w = c_w.unsqueeze(0)
        assert (layer.data.shape == c_w.shape)
        layer.data = c_w
    else:
        c_b = torch.from_numpy(caffe_weights['valueSet_b'][0][-1][:, 0])
        assert (layer.data.shape == c_b.shape)
        layer.data = c_b
# Logging into Tensorboard
writer = SummaryWriter(comment='-'+seq_name)
y = net.forward(Variable(torch.randn(1, 3, 480, 854)))
writer.add_graph(net, y[-1])

if gpu_id >= 0:
    torch.cuda.set_device(device=gpu_id)
    net.cuda()

# Visualize the network
if vis_net:
    x = torch.randn(1, 3, 480, 854)
    x = Variable(x)
    if gpu_id >= 0:
        x = x.cuda()
    y = net.forward(x)
    g = viz.make_dot(y, net.state_dict())
    g.view()


# Use the following optimizer
lr = 1e-8
wd = 0.0002
optimizer = optim.SGD([
    {'params': [pr[1] for pr in net.stages.named_parameters() if 'weight' in pr[0]], 'weight_decay': wd},
    {'params': [pr[1] for pr in net.stages.named_parameters() if 'bias' in pr[0]], 'lr': lr * 2},
    {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'weight' in pr[0]], 'weight_decay': wd},
    {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'bias' in pr[0]], 'lr': lr*2},
    {'params': [pr[1] for pr in net.upscale.named_parameters() if 'weight' in pr[0]], 'lr': 0},
    {'params': [pr[1] for pr in net.upscale_.named_parameters() if 'weight' in pr[0]], 'lr': 0},
    {'params': net.fuse.weight, 'lr': lr/100, 'weight_decay': wd},
    {'params': net.fuse.bias, 'lr': 2*lr/100},
    ], lr=lr, momentum=0.9)

# Preparation of the data loaders
# Define augmentation transformations as a composition
composed_transforms = transforms.Compose([tb.RandomHorizontalFlip(),
                                          tb.Resize(),
                                          # tb.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
                                          tb.ToTensor()])
# Training dataset and its iterator
db_train = tb.DAVISDataset(train=True, db_root_dir=db_root_dir, transform=composed_transforms, seq_name=seq_name)
trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=2)

# Testing dataset and its iterator
db_test = tb.DAVISDataset(train=False, db_root_dir=db_root_dir, transform=tb.ToTensor(), seq_name=seq_name)
testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=2)


num_img_tr = len(trainloader)
num_img_ts = len(testloader)
loss_tr = []
aveGrad = 0

print("Start of Online Training, sequence: " + seq_name)
start_time = timeit.default_timer()
# Main Training and Testing Loop
for epoch in range(0, nEpochs):
    # One training epoch
    running_loss_tr = 0
    for ii, sample_batched in enumerate(trainloader):

        inputs, gts = sample_batched['image'], sample_batched['gt']

        # Forward-Backward of the mini-batch
        inputs, gts = Variable(inputs), Variable(gts)
        if gpu_id >= 0:
            inputs, gts = inputs.cuda(), gts.cuda()

        outputs = net.forward(inputs)

        # Compute the fuse loss
        loss = class_balanced_cross_entropy_loss(outputs[-1], gts, size_average=False)
        running_loss_tr += loss.data[0]

        # Print stuff
        if epoch % (nEpochs//20) == (nEpochs//20 - 1):
            running_loss_tr /= num_img_tr
            loss_tr.append(running_loss_tr)

            print('[Epoch: %d, numImages: %5d]' % (epoch+1, ii + 1))
            print('Loss: %f' % running_loss_tr)
            writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)

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
        torch.save(net.state_dict(), os.path.join(exp_dir, 'models', seq_name + '_epoch-'+str(epoch+1) + '.pth'))

stop_time = timeit.default_timer()
print('Online training time: ' + str(stop_time - start_time))


# Testing Phase
if vis_res:
    import matplotlib.pyplot as plt
    plt.close("all")
    plt.ion()
    f, ax_arr = plt.subplots(1, 3)

save_dir = os.path.join(save_root_dir, 'OSVOS_online', seq_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

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

        # Save the result, attention to the index jj
        sm.imsave(os.path.join(save_dir, os.path.basename(fname[jj]) + '.png'), pred)

        if vis_res:
            img_ = np.transpose(img.numpy()[jj, :, :, :], (1, 2, 0))
            gt_ = np.transpose(gt.numpy()[jj, :, :, :], (1, 2, 0))
            gt_ = np.squeeze(gt)
            # Plot the particular example
            ax_arr[0].cla()
            ax_arr[1].cla()
            ax_arr[2].cla()
            ax_arr[0].set_title('Input Image')
            ax_arr[1].set_title('Ground Truth')
            ax_arr[2].set_title('Detection')
            ax_arr[0].imshow(tb.im_normalize(img_))
            ax_arr[1].imshow(gt_)
            ax_arr[2].imshow(tb.im_normalize(pred))
            plt.pause(0.001)
