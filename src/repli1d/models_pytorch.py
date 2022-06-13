import argparse
import os
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from torch import float32, nn, optim
from torch.utils.data import DataLoader, TensorDataset

loss_func = F.mse_loss


class MLP(nn.Module):
    # def __init__(self):
    def __init__(self, l1=120, l2=84):
        super().__init__()
        self.lin1 = nn.Linear(11, 46)
        self.lin2 = nn.Linear(46, 1)

    def forward(self, xb):
        xb = F.relu(self.lin1(xb))
        xb = self.lin2(xb)
        return xb


def mlp():
    model = MLP()
    return model, optim.Adam(model.parameters())


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--preprocessing', type=str, default='log')
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--cell_line', type=str, default='K562')
    parser.add_argument('--listfile', nargs='+', type=str,
                        default='data/K562_2000_merged_histones_init.csv.gz')
    parser.add_argument('--marks', nargs='+', type=str,
                        default=['H2A.Z', 'H3K27ac', 'H3K79me2', 'H3K27me3',
                                 'H3K9ac', 'H3K4me2', 'H3K4me3', 'H3K9me3',
                                 'H3K4me1', 'H3K36me3', 'H4K20me1'])
    parser.add_argument('--output', type=str, default=['initiation'])
    parser.add_argument('--output_dir', type=str,
                        default='development/')
    parser.add_argument('--image_format', type=str, default='png')

    args = parser.parse_args()

    df = pd.read_csv('{}'.format(args.listfile), compression='gzip')
    masks = pd.read_csv('data/hg19_2000_no_N_inside.csv')
    print('Number of NANs is {}'.format(masks['signal'].sum()))
    df.loc[~masks['signal'].astype(bool)] = np.nan
    df = df.dropna()
    print(df)
    print(torch.cuda.is_available())
    dev = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    if args.preprocessing == 'log FCNN':
        for i in args.marks + args.output:
            df[i] = df[i] + np.min(df[i][(df[i] != 0)])
            df[i] = np.log10(df[i])
        X_train = df.loc[df['chrom'] != 'chr1', args.marks].to_numpy()
        print(X_train.shape)
        y_train = df.loc[df['chrom'] != 'chr1', args.output].to_numpy()
        print(y_train.shape)
        X_test = df.loc[df['chrom'] == 'chr1', args.marks].to_numpy()
        y_test = df.loc[df['chrom'] == 'chr1', args.output].to_numpy()

        X_train, y_train = shuffle(X_train, y_train)
        X_val = torch.tensor(X_train[0:100000], dtype=float32)
        y_val = torch.tensor(y_train[0:100000], dtype=float32)
        X_train = torch.tensor(X_train[100000:], dtype=float32)
        y_train = torch.tensor(y_train[100000:], dtype=float32)
        train_ds = TensorDataset(X_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size)
        valid_ds = TensorDataset(X_val, y_val)
        valid_dl = DataLoader(valid_ds, batch_size=args.batch_size*(2**4))
        model, opt = mlp()
        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

        # Print optimizer's state_dict
        print("Optimizer's state_dict:")
        for var_name in opt.state_dict():
            print(var_name, "\t", opt.state_dict()[var_name])
        model.eval()
        torch.manual_seed(123)
        np.random.seed(123)
        input = torch.rand(1, 11)
        baseline = torch.zeros(1, 11)
        fit(args.max_epoch, model, loss_func, opt, train_dl, valid_dl)
        torch.save(model.state_dict(), '{}{}'.format(
            args.output_dir, 'model_weights.pth'))
        ig = IntegratedGradients(model)
        attributions, delta = ig.attribute(
            input, baseline, target=0, return_convergence_delta=True)
        print('IG Attributions:', attributions)
        print('Convergence Delta:', delta)
    # needs to be corrected
    if args.preprocessing == 'log FCNN Gridsearch':
        for i in args.marks:
            df[i] = df[i] + np.min(df[i][(df[i] != 0)])
            df[i] = np.log10(df[i])
        X_train = df.loc[df['chrom'] != 'chr1', args.marks].to_numpy()
        print(X_train.shape)
        y_train = df.loc[df['chrom'] != 'chr1', args.output].to_numpy()
        print(y_train.shape)
        X_test = df.loc[df['chrom'] == 'chr1', args.marks].to_numpy()
        y_test = df.loc[df['chrom'] == 'chr1', args.output].to_numpy()

        X_train, y_train = shuffle(X_train, y_train)
        X_val = torch.tensor(X_train[0:100000], dtype=float32)
        y_val = torch.tensor(y_train[0:100000], dtype=float32)
        X_train = torch.tensor(X_train[100000:], dtype=float32)
        y_train = torch.tensor(y_train[100000:], dtype=float32)
        train_ds = TensorDataset(X_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size)
        valid_ds = TensorDataset(X_val, y_val)
        valid_dl = DataLoader(valid_ds, batch_size=args.batch_size*(2**4))
        def train_mlp(config, checkpoint_dir='development/', data_dir=None):
            net = MLP(config["l1"], config["l2"])
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda:0"
                if torch.cuda.device_count() > 1:
                    net = nn.DataParallel(net)
            net.to(device)

            criterion = nn.MSELoss()
            optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)
            
            if checkpoint_dir:
                model_state, optimizer_state = torch.load(
                    os.path.join(checkpoint_dir, "checkpoint"))
                net.load_state_dict(model_state)
                optimizer.load_state_dict(optimizer_state)
            for epoch in range(10):  # loop over the dataset multiple times
                running_loss = 0.0
                epoch_steps = 0
                for i, data in enumerate(train_dl, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    epoch_steps += 1
                    if i % 2000 == 1999:  # print every 2000 mini-batches
                        print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                        running_loss / epoch_steps))
                        running_loss = 0.0

                # Validation loss
                val_loss = 0.0
                val_steps = 0
                total = 0
                correct = 0
                for i, data in enumerate(valid_dl, 0):
                    with torch.no_grad():
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)

                        outputs = net(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                        loss = criterion(outputs, labels)
                        val_loss += loss.cpu().numpy()
                        val_steps += 1

                with tune.checkpoint_dir(epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((net.state_dict(), optimizer.state_dict()), path)

                tune.report(loss=(val_loss / val_steps))
                print("Finished Training")
        def test_loss(net, device="cpu"):
            with torch.no_grad():
                outputs = net(X_val)
                outputs = outputs.cpu().numpy()
            return mean_squared_error(outputs, y_val.cpu().numpy())

        def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
            config = {
                "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
                "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
                "lr": tune.loguniform(1e-4, 1e-1),
                "batch_size": tune.choice([2, 4, 8, 16])
            }
            scheduler = ASHAScheduler(
                metric="loss",
                mode="min",
                max_t=max_num_epochs,
                grace_period=1,
                reduction_factor=2)
            reporter = CLIReporter(
                # parameter_columns=["l1", "l2", "lr", "batch_size"],
                metric_columns=["loss", "accuracy", "training_iteration"])
            result = tune.run(
                train_mlp,
                resources_per_trial={"cpu": 10, "gpu": gpus_per_trial},
                config=config,
                num_samples=num_samples,
                scheduler=scheduler,
                progress_reporter=reporter)

            best_trial = result.get_best_trial("loss", "min", "last")
            print("Best trial config: {}".format(best_trial.config))
            print("Best trial final validation loss: {}".format(
                best_trial.last_result["loss"]))
            # print("Best trial final validation accuracy: {}".format(
            #     best_trial.last_result["accuracy"]))

            best_trained_model = MLP(best_trial.config["l1"], best_trial.config["l2"])
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda:0"
                if gpus_per_trial > 1:
                    best_trained_model = nn.DataParallel(best_trained_model)
            best_trained_model.to(device)

            best_checkpoint_dir = best_trial.checkpoint.value
            model_state, optimizer_state = torch.load(os.path.join(
                best_checkpoint_dir, "checkpoint"))
            best_trained_model.load_state_dict(model_state)

            test_acc = test_loss(best_trained_model, device)
            print("Best trial test set loss: {}".format(test_acc))

        main(num_samples=10, max_num_epochs=5, gpus_per_trial=3)
