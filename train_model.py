import numpy as np
import torch
import torch.nn as nn
from utils import parse_args_with_config, load_data
from datasets import NpDataset, SequentialNpDataset, PreprocessedNpDataset
import models
from torch.utils.tensorboard import SummaryWriter
import time
import os
from torchvision import transforms
from utils import load_data, FuncList
from torch.utils.data import DataLoader
from train import train, validate
import aug


def main():
    args = parse_args_with_config()
    args.pin_memory = not args.no_memory_pinning

    # get device
    if torch.cuda.is_available() and not args.no_gpu:
        device = f'cuda:{args.gpu}'
    else:
        device = None

    # logging
    log_name = args.run_name + "_" + str(int(time.time()))
    writer = SummaryWriter(os.path.join(args.log_root, log_name))

    # transforms and online data augmentation
    transform_train = FuncList([])
    transform_test = FuncList([])

    # optional time warp
    if args.time_warp:
        def tw(x):
            x = np.expand_dims(x, axis=0)
            x = aug.time_warp(x, args.time_warp_sigma, args.time_warp_knot)
            x = x[0]
            return x
        transform_train.append(tw)

    if args.gaussian_eps > 0:
        def noise(x):
            return x + np.random.randn(*x.shape) * args.gaussian_eps
        transform_train.append(noise)

    if args.random_sample:
        def sample(x):
            a = np.random.choice(len(x), size=args.sample_size, replace=False)
            a.sort()
            return x[a]
        transform_train.append(sample)
        transform_test.append(sample)  # not sure about this

    # load dataset
    data = load_data(args.dataset_root)

    # create target to index mapping
    unique_targets = np.unique(data['y_train_valid'])
    offset = np.min(unique_targets)
    data['y_train_valid'] = data['y_train_valid'] - offset
    data['y_test'] = data['y_test'] - offset

    # datasets
    train_dataset = PreprocessedNpDataset(data['X_train_valid'], data['y_train_valid'], wndsze=args.wndsze,
                                          clipping=not args.no_clipping, sample_size=args.sample_size,
                                          sample_type=args.sample_type, store_as_tensor=False,
                                          transform=transform_train.apply)
    test_dataset = PreprocessedNpDataset(data['X_test'], data['y_test'], wndsze=args.wndsze,
                                         clipping=not args.no_clipping, sample_size=args.sample_size,
                                         sample_type=args.sample_type, store_as_tensor=False,
                                         transform=transform_test.apply)

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=args.pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             pin_memory=args.pin_memory)

    # create model
    model = models.__dict__[args.model](args.dropout)
    if device is not None:
        model = model.to(device)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = None
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                    weight_decay=args.l2_reg)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_reg)
    else:
        raise NotImplementedError("Unknown optimizer")

    # lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_milestones, args.lr_gamma)

    # train loop stats
    best_val_acc = 0
    for e in range(args.epochs):
        # train
        model.train()
        train_loss, train_acc = train(model, criterion, optimizer, train_loader, e, device=device)

        # validate
        model.eval()
        val_loss, val_acc = validate(model, criterion, test_loader, e, device=device)

        # update learning rate
        lr_scheduler.step()

        # log stats
        best_val_acc = max(val_acc, best_val_acc)
        writer.add_scalar("loss/train", train_loss, e)
        writer.add_scalar("acc/train", train_acc, e)
        writer.add_scalar("loss/val", val_loss, e)
        writer.add_scalar("acc/val", val_acc, e)
        writer.add_scalar("acc/val_best", best_val_acc, e)
        writer.add_scalar("optim/lr", lr_scheduler.get_last_lr()[0], e)

    # log hyperparams
    writer.add_hparams({
        "model": args.model,
        "optimizer": args.optimizer,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "l2_reg": args.l2_reg,
        "epochs": args.epochs
    }, {
        "best_val_acc": best_val_acc
    })


if __name__ == "__main__":
    main()

