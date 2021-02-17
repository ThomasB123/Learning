
# this code is based on https://stackoverflow.com/questions/57913825/how-to-select-specific-labels-in-pytorch-mnist-datasets

# imports
import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

# hyperparameters
batch_size  = 64
n_channels  = 3
latent_size = 512
dataset = 'stl10'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class YourSampler(torch.utils.data.sampler.Sampler): 
    def __init__(self, mask, data_source):
        self.mask = mask
        self.data_source = data_source
    
    def __iter__(self):
        return iter([i.item() for i in torch.nonzero(mask)])
    
    def __len__(self):
        return len(self.data_source)

if dataset == 'cifar10':
    cifar = torchvision.datasets.CIFAR10('drive/My Drive/training/cifar10', train=True, download=True, transform=torchvision.transforms.ToTensor())
    mask = [1 if cifar[i][1] == 2 or cifar[i][1] == 7 else 0 for i in range(len(cifar))]
    mask = torch.tensor(mask)
    sampler = YourSampler(mask,cifar)
    train_loader = torch.utils.data.DataLoader(cifar, batch_size=batch_size, sampler=sampler, drop_last=True)
    class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

if dataset == 'stl10':
    stl = torchvision.datasets.STL10('drive/My Drive/training/stl10', split='train', download=True, transform=torchvision.transforms.ToTensor())
    mask = [1 if stl[i][1] == 1 or stl[i][1] == 6 else 0 for i in range(len(stl))]
    mask = torch.tensor(mask)
    sampler = YourSampler(mask,stl)
    train_loader = torch.utils.data.DataLoader(stl, batch_size=batch_size, sampler=sampler, drop_last=True)
    class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

train_iterator = iter(cycle(train_loader))

class Block(nn.Module):
    def __init__(self, in_f, out_f):
        super(Block, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(in_f, out_f, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_f),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self,x):
        return self.f(x)

class Autoencoder(nn.Module):
    def __init__(self, f=16):
        super().__init__()

        self.encode = nn.Sequential(
            Block(n_channels, f),
            nn.MaxPool2d(kernel_size=(2,2)), # output = 48x48
            Block(f  ,f*2),
            nn.MaxPool2d(kernel_size=(2,2)), # output = 24x24
            Block(f*2,f*4),
            nn.MaxPool2d(kernel_size=(2,2)), # output = 12x12
            Block(f*4,f*4),
            nn.MaxPool2d(kernel_size=(2,2)), # output = 6x6
            Block(f*4,f*4),
            nn.MaxPool2d(kernel_size=(2,2)), # output = 3x3
            Block(f*4,f*4),
            nn.MaxPool2d(kernel_size=(3,3)), # output = 1x1
            Block(f*4,latent_size),
        )

        self.decode = nn.Sequential(
            nn.Upsample(scale_factor=3), # output = 3x3
            Block(latent_size,f*4),
            nn.Upsample(scale_factor=2), # output = 6x6
            Block(f*4,f*4),
            nn.Upsample(scale_factor=2), # output = 12x12
            Block(f*4,f*4),
            nn.Upsample(scale_factor=2), # output = 24x24
            Block(f*4,f*2),
            nn.Upsample(scale_factor=2), # output = 48x48
            Block(f*2,f  ),
            nn.Upsample(scale_factor=2), # output = 96x96
            nn.Conv2d(f,n_channels, 3,1,1),
            nn.Sigmoid()
        )

A = Autoencoder().to(device)
print(f'> Number of autoencoder parameters {len(torch.nn.utils.parameters_to_vector(A.parameters()))}')
optimiser = torch.optim.Adam(A.parameters(), lr=0.001)
epoch = 0

while (epoch<100):
    loss_arr = np.zeros(0)
    for i in range(100):
        x,t = next(train_iterator)
        x,t = x.to(device), t.to(device)

        z = A.encode(x)
        x_hat = A.decode(z)
        loss = ((x-x_hat)**2).mean()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        loss_arr = np.append(loss_arr, loss.item())
    
    z = torch.randn_like(z)
    g = A.decode(z)

    print('loss ' + str(loss.mean()))
    plt.rcParams['figure.dpi'] = 100
    plt.grid(False)
    plt.imshow(torchvision.utils.make_grid(g[:8]).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
    plt.show()
    plt.pause(0.0001)

    epoch = epoch+1

plt.rcParams['figure.dpi'] = 175
plt.grid(False)
plt.imshow(torchvision.utils.make_grid(g).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
plt.show()
