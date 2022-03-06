from tqdm import tqdm
import torch
import numpy as np


def train(model, criterion, optimizer, train_loader, epoch, device=None):
    """
    Trains a model for a single epoch.

    :param model: The model.  Must be in train mode and on the proper device.
    :param criterion: The loss function.
    :param optimizer: The optimizer
    :param train_loader: The training dataset dataloader
    :param epoch: the current epoch (used for tqdm)
    :param device: the device to move the data.  If none, no data moving will occur
    :return: train loss, train acc
    """
    # stats
    running_loss = 0
    running_correct = 0
    running_examples = 0
    loop = tqdm(train_loader, desc=f"Train {epoch}", total=len(train_loader))

    # preclear grads
    optimizer.zero_grad()

    # iterate through loader
    for batch_x, batch_y in loop:
        # potentially move data
        if device is not None:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

        # forward pass
        out = model.forward(batch_x, 'train')
        loss = criterion(out, batch_y).mean()

        # backward pass
        loss.backward()
        optimizer.step()

        # collect stats
        batch_size = len(batch_y)
        running_loss += loss.item() * batch_size
        n_correct = torch.sum(out.argmax(dim=-1) == batch_y).item()
        running_correct += n_correct
        running_examples += batch_size

        # clear grads
        optimizer.zero_grad()

        # update tqdm postfix
        loop.set_postfix({
            "loss": f"{running_loss / running_examples : .03f}",
            "acc": f"{running_correct / running_examples : .03f}"
        })

    return running_loss / running_examples, running_correct / running_examples

def validate(model, criterion, test_loader, epoch, device=None):
    """
    Computes the accuracy and loss without updating the model.

    :param model: The model.  Must be in the proper mode and on the proper device.
    :param criterion: The loss function.
    :param test_loader: The test loader.
    :param epoch: The current epoch (for tqdm)
    :param device: The device to move the batches to.  If None no data movement will occur.
    :param classwise: If true, will return test_acc as a vector of size num_classes
    :param num_classes: The number of classes (only needed if classwise is True).
    :return: test_loss, test_acc
    """

    # stats
    running_loss = 0
    running_correct = 0
    running_examples = 0
    loop = tqdm(test_loader, desc=f"Valid {epoch}", total=len(test_loader))

    # no grads
    with torch.no_grad():
        # iterate through loader
        for batch_x, batch_y in loop:
            # optionally move data
            if device is not None:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

            # forward pass
            out = model.forward(batch_x, 'test')
            loss = criterion(out, batch_y).mean()

            # collect stats
            batch_size = len(batch_y)
            running_loss += loss.item() * batch_size
            n_correct = torch.sum(out.argmax(dim=-1) == batch_y).item()
            running_correct += n_correct
            running_examples += batch_size

            # set tqdm postfix
            loop.set_postfix({
                "loss": f"{running_loss / running_examples : .03f}",
                "acc": f"{running_correct / running_examples : .03f}"
            })

    return running_loss / running_examples, running_correct / running_examples


def validate_classwise(model, criterion, test_loader, epoch, num_classes=10, device=None):

    # stats
    running_loss = np.zeros(num_classes)
    running_correct = np.zeros(num_classes)
    running_examples = np.zeros(num_classes)
    loop = tqdm(test_loader, desc=f"Valid {epoch}", total=len(test_loader))

    # no grads
    with torch.no_grad():
        # iterate through loader
        for batch_x, batch_y in loop:
            # optionally move data
            if device is not None:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

            # forward pass
            out = model.forward(batch_x)
            loss = criterion(out, batch_y)

            # collect stats
            correct = (out.argmax(-1) == batch_y)
            for c in range(num_classes):
                running_examples[c] += torch.sum(batch_y == c).item()
                running_correct[c] += torch.sum(correct[batch_y == c]).item()
                running_loss[c] += torch.sum(loss[batch_y == c]).item()

            avg_loss = np.sum(running_loss) / np.sum(running_examples)
            avg_acc = np.sum(running_correct) / np.sum(running_examples)

            # set tqdm postfix
            loop.set_postfix({
                "loss": f"{avg_loss : .03f}",
                "acc": f"{avg_acc : .03f}"
            })

    avg_loss = np.sum(running_loss) / np.sum(running_examples)
    avg_acc = np.sum(running_correct) / np.sum(running_examples)

    return running_loss / running_examples, avg_loss, running_correct / running_examples, avg_acc


