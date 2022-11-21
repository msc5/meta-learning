import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import time
import yaml
import shutil

# Our modules
import arch
import data


class Logger:

    def __init__(self, epochs, batches, path=None):
        self.epochs = epochs        # Total Epochs
        self.batches = batches      # Total Batches
        self.e = 0                  # Current Epoch
        self.b = 0                  # Current Batch
        self.path = path
        self.data = torch.zeros(epochs, batches, 5)

    def log(
        self,
        results,
        elapsed_time
    ):
        data = torch.tensor((*results, elapsed_time))
        self.data[self.e, self.b, :] = data
        means = self.data[self.e, 0:self.b + 1, :].mean(dim=0)
        msg = self.msg(means)
        self.b += 1
        if self.b == self.batches:
            self.b = 0
            self.e += 1
            if self.path is not None:
                torch.save(self.data, self.path)
        return msg

    def msg(self, data):
        train_loss, train_acc, test_loss, test_acc, elapsed_time = data
        msg = (
            f'{"":10}{self.e:<8}{self.b + 1:<3} / {self.batches:<6}'
            f'{train_loss:<10.4f}{train_acc:<10.4f}'
            f'{test_loss:<10.4f}{test_acc:<10.4f}'
            f'{elapsed_time:<10.4f}'
        )
        return msg

    def header(self):
        msg = (
            f'{"":35}{"Train":20}{"Test":20}\n'
            f'{"":10}{"Epoch":8}{"Batch":12}'
            f'{"Loss":10}{"Accuracy":10}'
            f'{"Loss":10}{"Accuracy":10}'
            f'{"Elapsed Time":15}\n'
        )
        return msg


def train(
        name,
        model,
        dataloader,
        callback,
        optimizer,
        scheduler,
        loss_fn,
        epochs,
        device,
):

    num_batches = len(dataloader)
    batch_size = dataloader.batch_size

    print(device)
    print(epochs, num_batches, batch_size)

    # Setup files
    model_type = model.__name__
    type_dir = os.path.join('saves', model_type)
    model_dir = os.path.join(type_dir, name)
    model_path = os.path.join(model_dir, 'model')
    log_path = os.path.join(model_dir, 'log')
    if not os.path.exists('saves'):
        os.makedirs('saves')
    if not os.path.exists(type_dir):
        os.makedirs(type_dir)
    if os.path.exists(model_dir):
        ans = input(
            f'{model_type + ": " + name} has already been trained. Overwrite save files? (y/n)\n',
        )
        if ans == 'y' or ans == 'Y':
            pass
        else:
            return
    else:
        os.makedirs(model_dir)
    config_path = os.path.join(model_dir, 'config.yaml')
    shutil.copyfile('config.yaml', config_path)

    # Initialize Logger
    logger = Logger(epochs, num_batches, log_path)

    # Use GPU or CPU to train model
    model = model.to(device)
    model.zero_grad()

    # Print header
    print(logger.header())
    tic = time.perf_counter()

    for i in range(epochs):

        t = tqdm(
            dataloader,
            colour='cyan',
            bar_format='{desc}|{bar:20}| {rate_fmt}',
            leave=False,
        )
        for j, (train_ds, test_ds) in enumerate(t):
            train_results = callback(
                model,
                train_ds,
                optimizer,
                loss_fn,
                device,
                train=True
            )
            with torch.no_grad():
                test_results = callback(
                    model,
                    test_ds,
                    None,
                    loss_fn,
                    device,
                    train=False
                )
            toc = time.perf_counter()
            log = logger.log((*train_results, *test_results), toc - tic)
            t.set_description(log)

        print(log)
        torch.save(model.state_dict(), model_path)
        scheduler.step()


def test(
        name,
        model,
        dataloader,
        callback,
        loss_fn,
        device,
):

    # Load Model from state_dict
    dir = os.path.join('saves', model.__name__, name)
    if not os.path.exists(dir):
        print(f'{name} does not exist')
    model_path = os.path.join(dir, 'model')
    log_path = os.path.join(dir, 'log_test')
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    batches = len(dataloader)
    print(batches)

    # Initialize Logger
    logger = Logger(1, batches, log_path)

    # Use GPU or CPU to train model
    model = model.to(device)
    model.zero_grad()

    # Print header
    print(logger.header())
    tic = time.perf_counter()

    with torch.no_grad():
        for j, test_ds in enumerate(dataloader):
            results = callback(
                model,
                test_ds,
                None,
                loss_fn, device,
                train=False
            )
            toc = time.perf_counter()
            log = logger.log((0, 0, *results), toc - tic)

    print(log)


def omniglotCallBack(
        model,
        inputs,
        optimizer,
        loss_fn,
        device,
        train=True
):
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    (sup_set, query_set), classes = inputs

    classes = classes.numpy()
    sup_set, query_set = sup_set.to(device), query_set.to(device)
    pred = model(sup_set, query_set)

    num_classes = int(pred.shape[1])
    query_num_examples_per_class = int(pred.shape[0] / num_classes)

    # target_indices = np.array(range(len(classes)))
    # class_to_index = dict(zip(classes, target_indices))
    # is it the case that the first query_num_examples_per_class rows (each row is a query in the prediction)
    # still corresponds to the first class in the classes array after all the reshaping and calculations
    # done in the RelationNetwork model?
    lab = torch.eye(num_classes).repeat_interleave(
        query_num_examples_per_class, dim=0).to(device)

    # Compute Loss
    loss_t = loss_fn(pred, lab)
    loss = loss_t.item()

    # Compute Accuracy
    correct = torch.sum(pred.argmax(dim=1) == lab.argmax(dim=1)).item()
    acc = correct / pred.shape[0]

    # zero gradient per batch or per epoch? usually, zero gradient per batch
    if train:
        optimizer.zero_grad()
        loss_t.backward()
        clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

    return loss, acc


def imagenetCallBack(
        model,
        inputs,
        optimizer,
        loss_fn,
        device,
        train=True
):

    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    (ss, sl), (ts, tl) = inputs

    ss = ss.squeeze(0)
    sl = sl.squeeze(0)
    ts = ts.squeeze(0)
    tl = tl.squeeze(0)

    # print(ss.shape, sl.shape, ts.shape, tl.shape)
    k = sl.shape[1]
    n = int(sl.shape[0] / k)
    m = int(tl.shape[0] / k)

    # fig, ax = plt.subplots(2, max(k * n, k * m))
    # for i in range(k):
    #     for j in range(n):
    #         ax[0, i + j].imshow(ss[i, j].view(84, 84, 3).cpu().numpy())
    #         ax[0, i + j].set_title(sl.argmax(dim=1)[i].item())
    # for i in range(k):
    #     for j in range(m):
    #         ax[1, i + j].imshow(ts[i, j].view(84, 84, 3).cpu().numpy())
    #         ax[1, i + j].set_title(tl.argmax(dim=1)[i].item())
    # plt.show()

    # print(ss.shape, ts.shape)
    pred = model(ss, ts)
    lab = tl

    # print(tl, pred)

    # Compute Loss
    loss_t = loss_fn(pred, lab)
    loss = loss_t.item()

    # Compute Accuracy
    correct = torch.sum(pred.argmax(dim=1) == lab.argmax(dim=1)).item()
    acc = correct / pred.shape[0]

    if train:
        loss_t.backward()
        clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

    return loss, acc


if __name__ == '__main__':

    config_file = open("config.yaml", "r")
    try:
        config = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        print(exc)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Task setup
    k = config['k']           # Number of classes
    n = config['n']           # Number of examples per support class
    m = config['m']           # Number of examples per query class

    if config['dataset'] == 'Omniglot':
        train_ds = data.OmniglotDataset(n, m, device, True)
        test_ds = data.OmniglotDataset(n, m, device, False)
        siamese = data.Siamese(train_ds, test_ds)
        dataloader = DataLoader(
            siamese,
            batch_size=k,
            shuffle=True,
            drop_last=True
        )
        test_dataloader = DataLoader(
            test_ds,
            batch_size=k,
            shuffle=True,
            drop_last=True
        )
        callback = omniglotCallBack
        filters_in = 1
        s = 28

    if config['arch'] == 'RelationNetwork':
        in_feat_rel = 64 if config['dataset'] == 'Omniglot' else 576
        model = arch.RelationNetwork(filters_in, 64, in_feat_rel, k, n, m)
    elif config['arch'] == 'MatchingNetwork':
        model = arch.MatchingNets(device, filters_in, 64)
    elif config['arch'] == 'CustomNetwork':
        model = arch.CustomNetwork(3, s, filters_in, 16, k, n, m, device)

    if config['loss_fn'] == 'MSE':
        loss_fn = nn.MSELoss()
    elif config['loss_fn'] == 'NLL':
        loss_fn = nn.NLLLoss()
    elif config['loss_fn'] == 'CrossEntropy':
        loss_fn = nn.CrossEntropy()

    model_name = config['name']
    model_arch = config['arch']
    lr = config['learning_rate']
    schedule = config['schedule']
    epochs = config['epochs']

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, schedule, gamma=0.5)

    print(
        f'Training {model_arch} {model_name} on {k}-way {n}-shot {m}-query-shot')

    if config['train']:
        train(
            model_name,
            model,
            dataloader,
            callback,
            optimizer,
            scheduler,
            loss_fn,
            epochs,
            device
        )
    if config['test']:
        test(
            model_name,
            model,
            test_dataloader,
            callback,
            loss_fn,
            device,
        )
