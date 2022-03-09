import numpy as np
import torch
import torch.nn as nn
from utils import parse_args_with_config, load_data
from datasets import NpDataset
import models
from torch.utils.tensorboard import SummaryWriter
import time
import os
from torchvision import transforms
from utils import load_data
from torch.utils.data import DataLoader
from train import train, validate
import preprocessing


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
    transform_train = None
    transform_test = None

    # load dataset
    data = load_data(args.dataset_root)

    xTrainTemp = data['X_train_valid']
    xTrainTemp = np.swapaxes(xTrainTemp,1,2)
    yTrainTemp = data['y_train_valid']

    xTestTemp = data['X_test']
    xTestTemp = np.swapaxes(xTestTemp,1,2)
    yTestTemp = data['y_test']

    data['X_train_valid'], data['y_train_valid'] = preprocessing.data_prep(xTrainTemp,yTrainTemp,2,2,True)
    data['X_test'], data['y_test'] = preprocessing.data_prep(xTestTemp,yTestTemp,2,2,True)


    # data is channels last by default (n, l, c)
    # conv nets need channels first ie (n, c, l)
    # optionally flip channels here
    if args.channels_first:
        data['X_train_valid'] = np.transpose(data['X_train_valid'], axes=(0, 2, 1))
        data['X_test'] = np.transpose(data['X_test'], axes=(0, 2, 1))

    # optionally reshape the data as a greyscale image
    if args.as_greyscale:
        data['X_train_valid'] = data['X_train_valid'][:, np.newaxis, :, :]
        data['X_test'] = data['X_test'][:, np.newaxis, :, :]

    if args.subsampling:
        n, seq_len, n_features = data['X_train_valid'].shape
        subsamp_filter = seq_len // args.subsample_size
        data['X_train_valid'] = data['X_train_valid'].reshape(n, args.subsample_size, subsamp_filter,
                                                              n_features)
        n, seq_len, n_features = data['X_test'].shape
        data['X_test'] = data['X_test'].reshape(n, args.subsample_size, subsamp_filter, n_features)

        def sample(x):
            s = np.random.choice(subsamp_filter, size=args.subsample_size)
            return x[np.arange(args.subsample_size), s]

        transform_train = sample
        transform_test = sample

    # create target to index mapping
    unique_targets = np.unique(data['y_train_valid'])
    offset = np.min(unique_targets)

    train_dataset = NpDataset(data['X_train_valid'], data['y_train_valid'] - offset, transform=transform_train,
                              store_as_tensor=True)
    test_dataset = NpDataset(data['X_test'], data['y_test'] - offset, transform=transform_test, store_as_tensor=True)

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=args.pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             pin_memory=args.pin_memory)

    # create model
    model = models.__dict__[args.model]()
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
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
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
        #"momentum": args.momentum,
        #"l2_reg": args.l2_reg,
        "epochs": args.epochs
    }, {
        "best_val_acc": best_val_acc
    })


if __name__ == "__main__":
    main()

