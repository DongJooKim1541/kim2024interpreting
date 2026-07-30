''' Make data

This code contains modified parts of the original work(https://github.com/johnsk95/PT4AL)

Reference:
[1] Yi, J. S. K., Seo, M., Park, J. & Choi, D.-G. Pt4al: Using self-supervised pretext tasks for active learning. In European Conference on Computer Vision (Tel Aviv, Israel, 2022)

'''

import torch
import torchvision
import os

class save_dataset(torch.utils.data.Dataset):

  def __init__(self, dataset, split='train'):
    self.dataset = dataset
    self.split = split

  def __getitem__(self, idx):
      data, label = self.dataset[idx]
      class_dir = f'./DATA/{self.split}/{label}'

      if not os.path.isdir(class_dir):
          os.makedirs(class_dir, exist_ok=True)

      path = f'{class_dir}/{idx}.png'
      data.save(path)

  def __len__(self):
    return len(self.dataset)


def prepare_directories():
    for path in ['./DATA', './DATA/train', './DATA/test']:
        os.makedirs(path, exist_ok=True)


def convert_cifar_to_png(train_dataset, test_dataset):
    for idx in range(len(train_dataset)):
        train_dataset[idx]

    for idx in range(len(test_dataset)):
        test_dataset[idx]


if __name__ == '__main__':
    prepare_directories()

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)

    train_dataset = save_dataset(trainset, split='train')
    test_dataset = save_dataset(testset, split='test')

    convert_cifar_to_png(train_dataset, test_dataset)