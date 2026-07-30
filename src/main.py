import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os
import numpy as np
from scipy.stats import beta, bernoulli
from models import ResNet18
from loader import Loader, Loader2
from config import *

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
criterion = nn.CrossEntropyLoss()

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = Loader(is_train=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

# Model
print('==> Building model..')
net = ResNet18().to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


def sampling_valid(labeled_data_per_AL_cycle):
    valid_ = []
    for group in range(0, GROUPS):
        with open(f'./loss/batch_{group}.txt', 'r') as f:
            samples = f.readlines()

            samples = np.array(samples)
            sample_per_groups = int(labeled_data_per_AL_cycle[-1] * V) // GROUPS
            sample_val = samples[[j * (len(samples) // sample_per_groups) for j in range(sample_per_groups)]]

            valid_.extend(sample_val)
    print("len(valid_): ", len(valid_))
    return valid_


def get_learning_amount_B():
    iterations_per_cycle_B = []
    learning_amount_per_cycle_B = []

    for i in range(0, CYCLES):
        iterations_per_cycle_B.append((labeled_data_per_AL_cycle[i] / BATCH_SIZE) * EPOCHS)
    print("iterations_per_cycle_B: ", iterations_per_cycle_B)

    for i in range(0, CYCLES):
        learning_amount_per_cycle_B.append(iterations_per_cycle_B[i] * labeled_data_per_AL_cycle[i])
    print("learning_amount_per_cycle_B: ", learning_amount_per_cycle_B)

    return learning_amount_per_cycle_B


def compute_per_cycle_labeling_data(labeling_data_per_cycle, num_sub_cycles):
    per_cycle = []
    remainder_total = 0

    for cycle_data in labeling_data_per_cycle:
        base_count = cycle_data // num_sub_cycles
        remainder = cycle_data % num_sub_cycles

        for _ in range(num_sub_cycles):
            per_cycle.append(base_count)

        for i in range(remainder):
            per_cycle[remainder_total + i] += 1
        remainder_total += num_sub_cycles

    return per_cycle


def compute_cumulative_labeling_data(per_cycle):
    cumulative = []
    total = 0
    for count in per_cycle:
        total += count
        cumulative.append(total)
    return cumulative


def compute_learning_amount_A(learning_amount_B, labeling_data_cycle_A, per_cycle_data):
    learning_amount_A = []
    idx = 0

    for cycle_idx in range(CYCLES):
        cycle_total = 0
        for sub_cycle_idx in range(sampling_ratio_A):
            iterations = math.ceil(labeling_data_cycle_A[idx] / BATCH_SIZE)
            cycle_total += iterations * labeling_data_cycle_A[idx]
            idx += 1

        learning_amount_A.append(cycle_total)

    return learning_amount_A


def get_learning_amount_A():
    per_cycle = compute_per_cycle_labeling_data(labeling_data_per_cycle, sampling_ratio_A)
    cumulative = compute_cumulative_labeling_data(per_cycle)

    print("labeling_data_cycle_A_per_cycle: ", per_cycle)
    print("labeling_data_cycle_A: ", cumulative)

    learning_amount_A = compute_learning_amount_A(learning_amount_per_cycle_B, cumulative, per_cycle)
    print("learning_amount_per_cycle_A: ", learning_amount_A)

    return learning_amount_A, per_cycle, cumulative


def get_epoch_cycles_A(learning_amount_per_cycle_B, learning_amount_per_cycle_A):
    epoch_A = []
    epoch_total_cycle_A = []

    for i in range(0, CYCLES):
        epoch_A.append(math.ceil(learning_amount_per_cycle_B[i] / learning_amount_per_cycle_A[i]))
    print("proper_epoch: ", epoch_A)

    for i in range(0, CYCLES):
        for j in range(0, sampling_ratio_A):
            epoch_total_cycle_A.append(epoch_A[i])
    print("epoch_total_cycle_A: ", epoch_total_cycle_A)
    return epoch_total_cycle_A


def adjust_labeling_for_validation(labeling_data_cycle_A, per_cycle, valid_count):
    valid_size = len(valid_count) if isinstance(valid_count, list) else valid_count

    if labeling_data_cycle_A[0] - valid_size > 0:
        labeling_data_cycle_A[0] -= valid_size
        per_cycle[0] -= valid_size
    else:
        delete_cycles = valid_size // labeling_data_cycle_A[0]
        remainder = valid_size % labeling_data_cycle_A[0]

        for i in range(delete_cycles):
            labeling_data_cycle_A[i] = 0
            per_cycle[i] = 0

        if delete_cycles < len(labeling_data_cycle_A):
            labeling_data_cycle_A[delete_cycles] -= remainder
            per_cycle[delete_cycles] -= remainder

    print("labeling_data_cycle_A (adjusted): ", labeling_data_cycle_A)
    print("labeling_data_cycle_A_per_cycle (adjusted): ", per_cycle)
    print("start_cycle: ", labeling_data_cycle_A.count(0))

    return labeling_data_cycle_A, per_cycle


def group_search_sampling_data(group_list_empty, Q_list, success_list, failure_list, valid_, labeled, sampling_quantity, model):
    group_empty = True

    while group_empty:
        # sample Q_list from beta distribution
        for i_th_group in range(0, GROUPS):
            if i_th_group in group_list_empty:
                Q_list[i_th_group] = float("-inf")
            else:
                Q_list[i_th_group] = float(
                    beta.rvs(success_list[i_th_group] + 1, failure_list[i_th_group] + 1, size=1))
        print("Q_list: ", Q_list)

        # choose group
        group = np.argmax(Q_list)
        print("group = ", group)

        # open group (sorted high -> low)
        with open(f'./loss/batch_{group}.txt', 'r') as f:
            samples = f.readlines()

        # exclude labeled data from samples
        samples_not_labeled = [x for x in samples if x not in valid_]
        samples_not_labeled = [x for x in samples_not_labeled if x not in labeled]
        print("sampling_quantity: ", sampling_quantity)
        if len(samples_not_labeled) < sampling_quantity:
            group_list_empty.append(group)
            print(f"group {group} empty!!")
            continue

        # interval sampling data for check uncertainty
        if sampling_quantity < len(samples_not_labeled) < sampling_quantity * sampling_ratio_A:
            samples_not_labeled = samples_not_labeled[0: len(samples_not_labeled): (
                    len(samples_not_labeled) // sampling_quantity)]  # interval slicing
        else:
            samples_not_labeled = samples_not_labeled[0: len(samples_not_labeled): sampling_ratio_A]  # interval slicing

        print("samples_not_labeled: ", len(samples_not_labeled))

        # sampling data
        sample_data = get_labels(model, samples_not_labeled, sampling_quantity)
        # add labeled data
        labeled.extend(sample_data)
        group_empty = False
    return group, group_list_empty, Q_list, success_list, failure_list, labeled


# Training
def train(net, criterion, optimizer, epoch, trainloader):
    print('\nEpoch: %d' % (int(epoch) + 1))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)' % (
        train_loss / len(trainloader), 100. * correct / total, correct, total))


# Validation for weight saving and calculate loss
def valid(net, criterion, epoch, total_epoch, cycle, best_acc, validloader):
    net.eval()
    valid_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print('Valid Loss: %.3f | Valid Acc: %.3f%% (%d/%d)' % (
        valid_loss / len(validloader), 100. * correct / total, correct, total))
    # Save checkpoint.
    acc = 100. * correct / total
    epoch_loss = valid_loss / len(validloader)

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/main_{cycle}.pth')
        best_acc = acc

    if epoch == total_epoch -1:
        if not os.path.exists(f'./checkpoint/main_{cycle}.pth'):
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, f'./checkpoint/main_{cycle}.pth')

    return best_acc, epoch_loss


# Test
def test(net, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    test_acc = 100. * correct / total
    print('Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)' % (
        test_loss / len(testloader), 100. * correct / total, correct, total))
    return test_acc


# get init loss for D_1
def get_init_loss(net, criterion, validloader):
    net.eval()
    init_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            init_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print('Init Loss: %.3f | Init Acc: %.3f%% (%d/%d)' % (
        init_loss / len(validloader), 100. * correct / total, correct, total))

    return init_loss / len(validloader)


# Sampling and labeling data
def get_labels(net, samples, num_samples):
    sub = Loader2(is_train=False, transform=transform_test, path_list=samples)
    ploader = torch.utils.data.DataLoader(sub, batch_size=1, shuffle=False, num_workers=4)
    top1_scores = []
    net.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(ploader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            scores, predicted = outputs.max(1)
            # save top1 confidence score
            outputs = F.normalize(outputs, dim=1)
            probs = F.softmax(outputs, dim=1).cpu()
            top1_scores.append(probs[0][predicted.item()])
    idx = np.argsort(top1_scores)
    samples = np.array(samples)

    return samples[idx[:num_samples]]


# get EMA value
def get_ema(alpha, last_ema, last_loss_diff, time_step):
    cur_ema = last_loss_diff * alpha + last_ema * (1 - alpha)
    cur_ema_prime = cur_ema / (1 - ((1 - alpha) ** time_step))

    return cur_ema, cur_ema_prime


# Sigmoid function
def sigmoid_func(input):
    return 1 / (1 + np.exp(input * (-1)))


def calculate_update_reward(group, reward_denominator, reward_numerator, success_list, failure_list, visited_list):
    if reward_denominator < 0:
        if reward_numerator >= reward_denominator:
            reward_prob = 1
        elif reward_numerator < reward_denominator:
            reward_prob = 0
    elif reward_denominator >= 0:
        reward_prob = reward_numerator / reward_denominator
        reward_prob = a * (reward_prob - b)
        reward_prob = sigmoid_func(reward_prob)
    # perform Bernoulli trial
    reward = int(bernoulli.rvs(size=1, p=reward_prob))
    # update S and F
    if reward == 1:
        success_list[group] = gamma * success_list[group] + 1
    else:
        failure_list[group] = gamma * failure_list[group] + 1
    visited_list[group] += 1

    for i in range(0, GROUPS):
        if i != group:
            success_list[i] = gamma * success_list[i]
            failure_list[i] = gamma * failure_list[i]

    with open(f'./main_epoch_end.txt', 'a') as f:
        if cycle == 0:
            f.write('[')
        f.write(str(test_acc) + ', ')
        if cycle == CYCLES * sampling_ratio_A - 1:
            f.write(']\n')
    return success_list, failure_list, visited_list


if __name__ == '__main__':
    labeled = []
    group_list_empty = []
    Q_list = [0.00] * sampling_ratio_A
    success_list = [0] * sampling_ratio_A
    failure_list = [0] * sampling_ratio_A
    visited_list = [0] * sampling_ratio_A
    last_ema = 0

    # for CIFAR-10 labeling
    labeled_data_per_AL_cycle = list(range(2000,20001,2000))
    labeling_data_per_cycle = [2000] * sampling_ratio_A

    valid_ = sampling_valid(labeled_data_per_AL_cycle)

    learning_amount_per_cycle_B = get_learning_amount_B()

    learning_amount_per_cycle_A, labeling_data_cycle_A_per_cycle, labeling_data_cycle_A = get_learning_amount_A()

    epoch_total_cycle_A = get_epoch_cycles_A(learning_amount_per_cycle_B, learning_amount_per_cycle_A)

    labeling_data_cycle_A, labeling_data_cycle_A_per_cycle = adjust_labeling_for_validation(labeling_data_cycle_A, labeling_data_cycle_A_per_cycle, valid_)

    # AL cycle
    for cycle in range(labeling_data_cycle_A.count(0), CYCLES * sampling_ratio_A):
        optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[math.ceil(0.8 * epoch_total_cycle_A[cycle])])
        print("milestones_optim: ", math.ceil((0.8 * epoch_total_cycle_A[cycle])))

        best_acc = 0
        print('\nCycle ', cycle)

        if cycle > labeling_data_cycle_A.count(0):
            print('>> Getting previous checkpoint')
            checkpoint = torch.load(
                f'./checkpoint/main_{cycle - 1}.pth')
            net.load_state_dict(checkpoint['net'])

        group, group_list_empty, Q_list, success_list, failure_list, labeled = \
            group_search_sampling_data(group_list_empty, Q_list, success_list, failure_list, valid_, labeled,
                                       labeling_data_cycle_A_per_cycle[cycle], net)

        print(f'>> Labeled length: {len(labeled) + len(valid_)}')
        trainset = Loader2(is_train=True, transform=transform_train, path_list=labeled)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        validset = Loader2(is_train=False, transform=transform_test, path_list=valid_)
        validloader = torch.utils.data.DataLoader(validset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        # get init loss for D_1
        if cycle == labeling_data_cycle_A_per_cycle.count(0):
            init_loss = get_init_loss(net, criterion, validloader)

        # train, valid, test
        for epoch in range(epoch_total_cycle_A[cycle]):
            train(net, criterion, optimizer, epoch, trainloader)
            best_acc, epoch_loss = valid(net, criterion, epoch, epoch_total_cycle_A[cycle], cycle, best_acc, validloader)
            if epoch == epoch_total_cycle_A[cycle] - 1:
                test_acc = test(net, criterion)
            scheduler.step()

        if cycle > labeling_data_cycle_A.count(0):  # update E and calculate D_t
            reward_numerator = last_cycle_loss - epoch_loss
            update_E, reward_denominator = get_ema(alpha, last_ema, loss_difference, cycle)
            last_ema = update_E
            last_cycle_loss = epoch_loss
            loss_difference = reward_numerator

            success_list, failure_list, visited_list = \
                calculate_update_reward(group, reward_denominator, reward_numerator, success_list, failure_list,
                                        visited_list)
        else:  # calculate initial D_1
            loss_difference = init_loss - epoch_loss
            last_cycle_loss = epoch_loss

        print("len(labeled): ", len(labeled))
        print("len(valid_): ", len(valid_))

        # Check Performance with seperate network
        if (len(labeled) + len(valid_)) in labeled_data_per_AL_cycle:
            print(f"\n>>Check Performace of data {len(labeled)}")

            net_ = ResNet18().to(device)  # initialize new network for check peformance
            optimizer_ = optim.SGD(net_.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
            scheduler_ = torch.optim.lr_scheduler.MultiStepLR(optimizer_, milestones=[math.ceil(0.8 * EPOCHS)])

            trainset_ = Loader2(is_train=True, transform=transform_train, path_list=labeled)
            trainloader_ = torch.utils.data.DataLoader(trainset_, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

            # train, test
            for epoch in range(EPOCHS):
                train(net_, criterion, optimizer_, epoch, trainloader_)
                if epoch == EPOCHS - 1:
                    test_acc = test(net_, criterion)
                scheduler_.step()

            with open(f'./performance_list.txt', 'a') as f:
                f.write(str(test_acc) + ', ')
