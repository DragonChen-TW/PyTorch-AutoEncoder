# Author: Dragon Chen
# Title: Dataset of handwritten digits
# Date: 2020/10/12
# Version: 0.8
# Contain: cifar10, mnistm, mnist, svhn, syn_number

import platform, os
from PIL import Image
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import numpy as np
from scipy.io import loadmat

batch_size = 64
num_workers = 16
download = False

def get_cifar10(data_dir='/home/dragonchen/data/'):
    trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset_train = datasets.CIFAR10(data_dir, train=True,
                            transform=trans_train, download=download)
    dataset_test = datasets.CIFAR10(data_dir, train=False,
                            transform=trans_test, download=download)

    train_data = DataLoader(dataset=dataset_train,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True)
    test_data = DataLoader(dataset=dataset_test,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False)

    return train_data, test_data

def get_mnist(data_dir='/home/dragonchen/data/', to_rgb=False):
    trans = []
    if to_rgb:
        trans.append(GrayscaleToRgb())
    trans.append(transforms.ToTensor())

    trans = transforms.Compose(trans)

    dataset_train = datasets.MNIST(root=data_dir, train=True,
                            transform=trans,
                            download=download)
    dataset_test = datasets.MNIST(root=data_dir, train=False,
                            transform=trans,
                            download=download)

    train_data = DataLoader(dataset=dataset_train,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True)
    test_data = DataLoader(dataset=dataset_test,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False)

    return train_data, test_data

class mnistmData(data.Dataset):
    def __init__(self, root, mode='train', to_gray=False):
        '''
        Get Image Path.
        '''
        self.mode = mode # 'train' or 'test'

        if mode == 'train':
            folder = os.path.join(root, 'mnist_m_train')
        else:
            folder = os.path.join(root, 'mnist_m_test')

        # generate imgs path
        imgs = [os.path.join(folder, img).replace('\\', '/') \
                for img in os.listdir(folder)]
        imgs = sorted(imgs, key=lambda img: int(img.split('/')[-1].split('.')[0]))
        imgs_num = len(imgs)
        self.imgs = imgs

        # generate labels
        label_file = 'mnist_m_{}_labels.txt'.format(mode)
        label_file = os.path.join(root, label_file)
        with open(label_file) as f:
            self.labels = [l[:-1].split(' ')[1] for l in f]

        # shape is 32x32
        # transforms
        trans = []
        if mode == 'train':
            trans += [
                transforms.RandomCrop(28),
            ]
        else:
            trans += [
                transforms.CenterCrop(28),
            ]

        if to_gray == True:
            trans += [
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]
        else:
            trans += [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]

        self.transforms = transforms.Compose(trans)

    def __getitem__(self, index):
        '''
        return one image's data
        if in test dataset, return image's id
        '''
        img_path = self.imgs[index]
        label = self.labels[index]
        label = int(label)

        data = Image.open(img_path)
        data = self.transforms(data)

        return data, label

    def __len__(self):
        return len(self.imgs)

def get_mnistm(data_dir='/home/dragonchen/data/mnist_m/', to_gray=False):
    dataset_train = mnistmData(data_dir, 'train', to_gray=to_gray)
    dataset_test = mnistmData(data_dir, 'test', to_gray=to_gray)

    train_data = DataLoader(dataset=dataset_train,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True, drop_last=True)
    test_data = DataLoader(dataset=dataset_test,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           shuffle=False, drop_last=True)

    return train_data, test_data

class SynNumberData(data.Dataset):
    def __init__(self, root, mode='train'):
        self.mode = mode

        if mode == 'train':
            file_path = os.path.join(root, 'synth_train_32x32.mat')
        else:
            file_path = os.path.join(root, 'synth_test_32x32.mat')

        data = loadmat(file_path)
        self.imgs = data['X'].transpose(3, 2, 0, 1)
        self.labels = data['y'].astype(np.int64).squeeze()

#         SVHN assign class 10 to number 0, convert it
        np.place(self.labels, self.labels == 10, 0)

        if mode == 'train':
            trans = [
                transforms.RandomCrop(28),
                transforms.ToTensor(),
            ]
        else:
            trans = [
                transforms.CenterCrop(28),
                transforms.ToTensor(),
            ]

        self.transforms = transforms.Compose(trans)

    def __getitem__(self, index):
        img = self.imgs[index]
        label = int(self.labels[index])

        img = Image.fromarray(img.transpose(1, 2, 0))
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.imgs)

def get_syn_number(data_dir='/home/dragonchen/data/synth_digits'):
    dataset_train = SynNumberData(data_dir, 'train')
    dataset_test = SynNumberData(data_dir, 'test')

    train_data = DataLoader(dataset=dataset_train,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True)
    test_data = DataLoader(dataset=dataset_test,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False)

    return train_data, test_data

def get_svhn(data_dir='/home/dragonchen/data/SVHN/'):
    train_transforms = transforms.Compose([
        transforms.RandomCrop(28),
        transforms.ToTensor(),
    ])
    test_transforms = transforms.Compose([
        transforms.CenterCrop(28),
        transforms.ToTensor(),
    ])

    dataset_train = datasets.SVHN(root=data_dir, split='train',
                            transform=train_transforms,
                            download=download)
    dataset_test = datasets.SVHN(root=data_dir, split='test',
                            transform=test_transforms,
                            download=download)

    train_data = DataLoader(dataset=dataset_train,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True)
    test_data = DataLoader(dataset=dataset_test,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False)

    return train_data, test_data

class RandomSVHNDataset(data.Dataset):
    def __init__(self, length, mode='train'):
        data_dir = '/home/dragonchen/data/SVHN/'
        if mode == 'train':
            train_transforms = transforms.Compose([
                transforms.RandomCrop(28),
                transforms.ToTensor(),
            ])
            self.svhn = datasets.SVHN(root=data_dir, split='train',
                                transform=train_transforms,
                                download=download)
            print(mode, 'svhn has ', len(self.svhn))
        else:
            test_transforms = transforms.Compose([
                transforms.CenterCrop(28),
                transforms.ToTensor(),
            ])
            self.svhn = datasets.SVHN(root=data_dir, split='test',
                                transform=test_transforms,
                                download=download)

        # record len of Syn Digits
        self.length = length
        self.rng = np.random.RandomState(1340)

    def __getitem__(self, i):
        idx = self.rng.choice(len(self.svhn))
        return self.svhn[idx]

    def __len__(self):
        return self.length

def get_svhn_random():
#     inject len of SVHN into
    dataset_train = RandomSVHNDataset(479400, 'train')
    dataset_test = RandomSVHNDataset(9553, 'test')

    train_data = DataLoader(dataset=dataset_train,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True)
    test_data = DataLoader(dataset=dataset_test,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False)
    return train_data, test_data

def get_dataset(data_name, **kwargs):
    if data_name == 'cifar10':
        train_data, test_data = get_cifar10(**kwargs)
    elif data_name == 'mnistm':
        train_data, test_data = get_mnistm(**kwargs)
    elif data_name == 'mnistm_1channel':
        train_data, test_data = get_mnistm(to_gray=True, **kwargs)
    elif data_name == 'mnist':
        train_data, test_data = get_mnist(**kwargs)
    elif data_name == 'mnist_3channel':
        train_data, test_data = get_mnist(to_rgb=True, **kwargs)
    elif data_name == 'svhn':
        train_data, test_data = get_svhn(**kwargs)
    elif data_name == 'svhn_random':
        train_data, test_data = get_svhn_random(**kwargs)
    elif data_name == 'syn_number':
        train_data, test_data = get_syn_number(**kwargs)

    return train_data, test_data

if __name__ == '__main__':
    print('cifar10')
    train_d, test_d = get_cifar10()
    print(next(iter(train_d))[0].shape)
    print(next(iter(test_d))[0].shape)

#     print('mnist')
#     train_d, test_d = get_mnist()
#     print(next(iter(train_d))[0].shape)
#     print(next(iter(test_d))[0].shape)
#     train_d, test_d = get_mnist(to_rgb=True)
#     print(next(iter(train_d))[0].shape)
#     print(next(iter(test_d))[0].shape)

#     print('mnist-m')
#     train_d, test_d = get_mnistm()
#     print(next(iter(train_d))[0].shape)
#     print(next(iter(test_d))[0].shape)
#     train_d, test_d = get_mnistm(to_gray=True)
#     print(next(iter(train_d))[0].shape)
#     print(next(iter(test_d))[0].shape)

#     print('svhn')
#     train_d, test_d = get_svhn()
#     print(next(iter(train_d))[0].shape)
#     print(next(iter(test_d))[0].shape)

#     print('synth_digits')
#     dataset = SynNumberData(mode='train')
#     print(len(dataset))
#     dataset = SynNumberData(mode='test')
#     print(len(dataset))
#     img, label = dataset[0]
#     print(img.shape, label)
#     train_d, test_d = get_syn_number()
#     print(next(iter(train_d))[0].shape)
#     print(next(iter(test_d))[0].shape)
