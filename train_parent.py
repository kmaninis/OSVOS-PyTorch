# Includes
from __future__ import division
import numpy as np
import os
import matplotlib.pyplot as plt
import visualize as viz
import osvos_toolbox as tb
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import vgg_osvos as vo
import timeit
from custom_layers import class_balanced_cross_entropy_loss

# Setting of parameters
# Parameters in p are used for the name of the model
p = {
    'trainBatch': 1,  # Number of Images in each mini-batch
    }

# # Setting other parameters
nEpochs = 200  # Number of epochs for training
useTest = 1  # See evolution of the test set when training?
testBatch = 1  # Testing Batch
nTestInterval = 20  # Run on test set every nTestInterval epochs
db_root_dir = '/media/eec/external/Databases/Segmentation/DAVIS/'
save_dir = '/home/eec/Desktop/pytorch_experiments/osvos/'
vis_net = 1  # Visualize the network?
gpu_id = 0  # Select which GPU, -1 if CPU
snapshot = 20  # Store a model every snapshot epochs
nAveGrad = 10


# Network definition
net = vo.OSVOS(pretrained=1)
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
    {'params': [pr[1] for pr in net.score_dsn.named_parameters() if 'weight' in pr[0]], 'lr': lr/10, 'weight_decay': wd},
    {'params': [pr[1] for pr in net.score_dsn.named_parameters() if 'bias' in pr[0]], 'lr': 2*lr/10},
    {'params': net.fuse.weight, 'lr': lr/100, 'weight_decay': wd},
    {'params': net.fuse.bias, 'lr': 2*lr/100},
    ], lr=lr, momentum=0.9)


# Preparation of the data loaders
# Define augmentation transformations as a composition
composed_transforms = transforms.Compose([tb.RandomHorizontalFlip(),
                                          tb.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
                                          tb.ToTensor()])
# Training dataset and its iterator
db_train = tb.DAVISDataset(train=True, inputRes=None, db_root_dir=db_root_dir, transform=composed_transforms)
trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=1, num_workers=1)

# Testing dataset and its iterator
db_test = tb.DAVISDataset(train=False, db_root_dir=db_root_dir, transform=tb.ToTensor())
testloader = DataLoader(db_test, batch_size=testBatch, shuffle=False, num_workers=1)

num_img_tr = len(trainloader)
num_img_ts = len(testloader)
running_loss_tr = [0] * 5
running_loss_ts = [0] * 5
loss_tr = []
loss_ts = []
aveGrad = 0

modelName = tb.construct_name(p, "OSVOS_parent")

print("Training Network")

# Main Training and Testing Loop
for epoch in range(0, nEpochs):
    start_time = timeit.default_timer()
    # One training epoch
    for ii, sample_batched in enumerate(trainloader):

        inputs, gts = sample_batched['image'], sample_batched['gt']

        # Forward-Backward of the mini-batch
        inputs, gts = Variable(inputs), Variable(gts)
        if gpu_id >= 0:
            inputs, gts = inputs.cuda(), gts.cuda()

        outputs = net.forward(inputs)

        # Compute the losses, side outputs and fuse
        losses = [None] * len(outputs)
        for i in range(0, len(outputs)):
            losses[i] = class_balanced_cross_entropy_loss(outputs[i], gts, size_average=False)
            running_loss_tr[i] += losses[i].data[0]
        loss = (1 - epoch / nEpochs)*sum(losses[:-1]) + losses[-1]

        # Print stuff
        if ii % num_img_tr == num_img_tr-1:
            running_loss_tr = [x / num_img_tr for x in running_loss_tr]
            loss_tr.append(running_loss_tr[-1])

            print('[Epoch: %d, numImages: %5d]' % (epoch+1, ii + 1))
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
            optimizer.step()
            optimizer.zero_grad()
            aveGrad = 0

    # Save the model
    if (epoch % snapshot) == snapshot - 1 and epoch != 0:
        torch.save(net.state_dict(), os.path.join(save_dir, modelName+'_epoch-'+str(epoch)+'.pth'))

    # One testing epoch
    if useTest and epoch % nTestInterval == (nTestInterval-1):
        for ii, sample_batched in enumerate(testloader):
            inputs, gts = sample_batched['image'], sample_batched['gt']

            # Forward pass of the mini-batch
            inputs, gts = Variable(inputs, volatile=True), Variable(gts, volatile=True)
            if gpu_id >= 0:
                inputs, gts = inputs.cuda(), gts.cuda()

            outputs = net.forward(inputs)

            # Compute the losses, side outputs and fuse
            losses = [None] * len(outputs)
            for i in range(0, len(outputs)):
                losses[i] = class_balanced_cross_entropy_loss(outputs[i], gts, size_average=False)
                running_loss_ts[i] += losses[i].data[0]
            loss = (1 - epoch / nEpochs) * sum(losses[:-1]) + losses[-1]

            # Print stuff
            if ii % num_img_ts == num_img_ts-1:
                running_loss_ts = [x / num_img_ts for x in running_loss_ts]
                loss_ts.append(running_loss_ts[-1])

                print('[Epoch: %d, numImages: %5d]' % (epoch + 1, ii + 1))
                for l in range(0, len(running_loss_ts)):
                    print('***Testing *** Loss %d: %f' % (l, running_loss_ts[l]))
                    running_loss_ts[l] = 0