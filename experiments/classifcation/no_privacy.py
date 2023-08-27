# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
import wandb

import sys

sys.path.append(os.path.join('..', '..', 'utils'))
sys.path.append(os.path.join('..', '..'))

from conf import global_settings as settings
from utils_train import dataset_loader, get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, get_training_dataloader_permuted, get_test_dataloader_permuted


def train(epoch):

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(training_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        #for name, para in last_layer.named_parameters():
        #    if 'weight' in name:
        #        writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #    if 'bias' in name:
        #        writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(training_loader.dataset)
        ))

        #update training loss for each iteration
        #writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    #for name, param in net.named_parameters():
    #    layer, attr = os.path.splitext(name)
    #    attr = attr[1:]
    #    writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

    return epoch, loss.item(), optimizer.param_groups[0]['lr']

@torch.no_grad()
def eval_training(epoch=0, tb=True):
    global loss_item

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset),
        finish - start
    ))
    print()

    try:
        wandb.log({'epochs': epoch,
                '(train) Total loss': loss_item,
                'LR': optimizer.param_groups[0]['lr'],
                '(test) Metric loss': test_loss / len(test_loader.dataset),
                '(test) Accuracy': correct.float() / len(test_loader.dataset)
                #'(test) Metric loss': loss_ce_test / len(dataloader_test)
             })
    except:
                wandb.log({'epochs': epoch,
                '(train) Total loss': 0,
                'LR': optimizer.param_groups[0]['lr'],
                '(test) Metric loss': test_loss / len(test_loader.dataset),
                '(test) Accuracy': correct.float() / len(test_loader.dataset)
                #'(test) Metric loss': loss_ce_test / len(dataloader_test)
             })

    return correct.float() / len(test_loader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-permute', action='store_true', default=False, help='permute train data or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-save_weights', action='store_true', default=False, help='resume training')
    parser.add_argument('-dataset', type=str, help='dataset used for training')

    args = parser.parse_args()

    ## establishing connection to wandb ...
    wandb.login(key='fe0119224af6709c85541483adf824cec731879e')
    config = {
                "learning_rate": args.lr,
                "epochs": settings.EPOCH,
                "batch_size": args.b,
                "architecture": "normal VGG16 FINETUNED"
            }

    wandb.init(project="privacy shuffling", config=config, reinit=True)

    net = get_network(args)

    mean = settings.CIFAR100_TRAIN_MEAN if args.dataset == 'cifar100' else settings.CIFAR10_TRAIN_MEAN
    std = settings.CIFAR100_TRAIN_STD if args.dataset == 'cifar100' else settings.CIFAR10_TRAIN_STD

    training_loader, test_loader, perm = dataset_loader(
        args.dataset,
        mean,
        std,
        num_workers=4,
        batch_size=args.b,
        shuffle=True,
        shuffle_pixels = args.permute
    )

    loss_function = nn.CrossEntropyLoss()
    if args.gpu:
        loss_function = loss_function.cuda()
    
    #optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-8)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    #writer = SummaryWriter(log_dir=os.path.join(
    #        settings.LOG_DIR, args.net, settings.TIME_NOW))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        print('resume training start epoch {}'.format(resume_epoch))
        resume_epoch = 1


    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        n_epoch, loss_item, n_lr = train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            if args.save_weights:
                torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            if args.save_weights:
                torch.save(net.state_dict(), weights_path)

#    writer.close()