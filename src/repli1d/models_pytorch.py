import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from sklearn.utils import shuffle
from torch import float32, nn, optim
from torch.utils.data import DataLoader, TensorDataset

loss_func = F.mse_loss


class MLP(nn.Module):
    def __init__(self):
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
    if args.preprocessing == 'log':
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
