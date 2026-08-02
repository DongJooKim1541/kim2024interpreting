''' Make data batches

This code contains modified parts of the original work(https://github.com/johnsk95/PT4AL)

Reference:
[1] Yi, J. S. K., Seo, M., Park, J. & Choi, D.-G. Pt4al: Using self-supervised pretext tasks for active learning. In European Conference on Computer Vision (Tel Aviv, Israel, 2022)

'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from typing import Optional, Union, List, Tuple, Dict, Any

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random
import numpy as np

from .models import *
from .loader import Loader, RotationLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = RotationLoader(is_train=False,  transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

net = ResNet18()
net.linear = nn.Linear(512, 4)
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint = torch.load('./checkpoint/rotation.pth')
net.load_state_dict(checkpoint['net'])

criterion = nn.CrossEntropyLoss()

def test() -> None:
    global best_acc
    net.eval()
    test_loss: float = 0
    correct: int = 0
    total: int = 0
    with torch.no_grad():
        for batch_idx, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3, path) in enumerate(testloader):
            inputs, inputs1, targets, targets1 = inputs.to(device), inputs1.to(device), targets.to(device), targets1.to(device)
            inputs2, inputs3, targets2, targets3 = inputs2.to(device), inputs3.to(device), targets2.to(device), targets3.to(device)
            outputs = net(inputs)
            outputs1 = net(inputs1)
            outputs2 = net(inputs2)
            outputs3 = net(inputs3)
            loss1 = criterion(outputs, targets)
            loss2 = criterion(outputs1, targets1)
            loss3 = criterion(outputs2, targets2)
            loss4 = criterion(outputs3, targets3)
            loss = (loss1+loss2+loss3+loss4)/4.
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loss = loss.item()
            s = str(float(loss)) + '_' + str(path[0]) + "\n"

            with open('loss/rotation_loss.txt', 'a') as f:
                f.write(s)


def parse_loss_file(filepath: str) -> Tuple[List[str], List[str]]:
    with open(filepath, 'r') as f:
        losses = f.readlines()

    loss_values: List[str] = []
    image_paths: List[str] = []
    for line in losses:
        parts = line.strip().split('_', 1)
        if len(parts) == 2:
            loss_values.append(parts[0])
            image_paths.append(parts[1])

    return loss_values, image_paths


def create_groups(loss_values: List[str], image_paths: List[str], num_groups: int = 10, samples_per_group: int = 5000) -> None:
    os.makedirs('loss', exist_ok=True)

    loss_array = np.array(loss_values, dtype=float)
    sort_indices = np.argsort(-loss_array)

    for group_id in range(num_groups):
        start_idx = group_id * samples_per_group
        end_idx = (group_id + 1) * samples_per_group
        group_indices = sort_indices[start_idx:end_idx]

        filepath = f'loss/batch_{group_id}.txt'
        with open(filepath, 'w') as f:
            for idx in group_indices:
                f.write(f'{image_paths[int(idx)]}\n')


if __name__ == "__main__":
    test()

    loss_values, image_paths = parse_loss_file('loss/rotation_loss.txt')
    create_groups(loss_values, image_paths)
