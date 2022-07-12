import argparse
import copy
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from torch import float32, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


torch.cuda.empty_cache()

loss_func = F.mse_loss
# loss_func = nn.BCELoss()


class MLP(nn.Module):
    # def __init__(self):
    def __init__(self, units=100):
        super().__init__()
        self.lin1 = nn.Linear(11, units)
        self.lin2 = nn.Linear(units, 1)

    def forward(self, xb):
        xb = F.relu(self.lin1(xb))
        xb = F.relu(self.lin2(xb))
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
    # Early stopping
    last_loss = 1000
    patience = 3
    trigger_times = 0
    val_loss_list = []
    train_loss_list = []
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            train_losses, aaa = loss_batch(model, loss_func, xb, yb, opt)
        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
            train_losses, train_nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in train_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        val_loss_list.append(val_loss)
        train_loss = np.sum(np.multiply(train_losses,
                                        train_nums)) / np.sum(train_nums)
        train_loss_list.append(train_loss)
        print(epoch, val_loss)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        # Early stopping
        current_loss = val_loss
        print('The Current Loss:', current_loss)
        if val_loss <= min(val_loss_list):
            best_model = copy.deepcopy(model)
        if current_loss > last_loss:
            trigger_times += 1
            print('Trigger Times:', trigger_times)

            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                return best_model, val_loss_list, train_loss_list
        else:
            print('trigger times: 0')
            trigger_times = 0
        if epoch + 1 >= epochs:
            return best_model, val_loss_list, train_loss_list
        last_loss = current_loss


def report(predicted, predicted_test, y_train, y_test, min_init, max_init,
           preprocessing, output_dir, image_format, cell_line):

    if preprocessing == 'log to raw' or preprocessing == 'raw to raw':
        print(mean_squared_error(predicted, y_train))
        print(mean_squared_error(predicted_test, y_test))
        plt.scatter(np.log10(y_train.ravel() + 1), np.log10(predicted + 1),
                    s=0.1)
        plt.title('Predicted values with respect to the observed values')
        plt.ylabel('Predicted vlaues')
        plt.xlabel('Observed values')
        plt.axis('square')
        plt.savefig('{}{}distribution_performance.{}'.format(output_dir,
                                                             preprocessing,
                                                             image_format),
                    dpi=300, bbox_inches='tight')
        plt.savefig('{}{}distribution_performance.eps'.format(output_dir,
                                                              preprocessing),
                    dpi=300, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(5, 5))
        plt.plot([p1, p2], [p1, p2], 'w-')
        plt.hist2d(np.log10(y_train.ravel() + 1), np.log10(predicted.ravel() + 1),
                   bins=[300, 300], cmap=plt.cm.nipy_spectral,
                   norm=matplotlib.colors.LogNorm(vmin=None, vmax=None,
                                                  clip=False))
        plt.xlim(0, 2)
        plt.ylim(0, 2)
        plt.clim(vmin=10, vmax=10**3)
        plt.colorbar()
        plt.xlabel('Log(observed values + 1)')
        plt.ylabel('Log(predicted values + 1)')
        plt.title('Predicted values with respect to the observed values for {}'.format(
            cell_line))
        plt.savefig('{}{}{}.{}'.format(output_dir, preprocessing,
                                       cell_line,
                                       image_format),
                    dpi=300, bbox_inches='tight', transparent=False)
        plt.savefig('{}{}{}.eps'.format(output_dir, preprocessing,
                                        cell_line),
                    dpi=300, bbox_inches='tight', transparent=False)
        plt.close()
        plt.plot(y_train[0:50], '-o')
        plt.plot(predicted[0:50], '-o')
        plt.legend(['Observed', 'Predicted'], loc='upper right')
        plt.title('Comparison of observed values and predicted values by FCNN')
        plt.savefig('{}{}{}comaprison_r_p.{}'.format(output_dir, preprocessing,
                                                     cell_line, image_format),
                    dpi=300, bbox_inches='tight', transparent=False)
        plt.savefig('{}{}{}comaprison_r_p.eps'.format(output_dir, preprocessing,
                                                      cell_line),
                    dpi=300, bbox_inches='tight', transparent=False)
        plt.close()

    if preprocessing == 'log to log' or preprocessing == 'raw to log':
        print(mean_squared_error(10**predicted, 10**y_train))
        print(mean_squared_error(10**predicted_test, 10**y_test))
        plt.scatter(y_train, predicted, s=0.1)
        plt.title('Log of predicted values with respect to the log of observed values')
        plt.ylabel('Predicted vlaues')
        plt.xlabel('Observed values')
        plt.axis('square')
        plt.savefig('{}{}distribution_performance.{}'.format(output_dir,
                                                             preprocessing,
                                                             image_format),
                    dpi=300, bbox_inches='tight')
        plt.savefig('{}{}distribution_performance.eps'.format(output_dir,
                                                              preprocessing),
                    dpi=300, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(5, 5))
        plt.plot([p1, p2], [p1, p2], 'w-')
        plt.hist2d(y_train.ravel(), predicted.ravel(),
                   bins=[300, 300],
                   cmap=plt.cm.nipy_spectral, norm=matplotlib.colors.LogNorm(
                    vmin=None, vmax=None, clip=False))
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.clim(vmin=10, vmax=10**3)
        plt.colorbar()
        plt.xlabel('Log(observed values + min(observed values))')
        plt.ylabel('Log(predicted values + min(observed values))')
        plt.title('Predicted values with respect to the observed values for {}'.format(
            cell_line))
        plt.savefig('{}{}{}.{}'.format(output_dir, preprocessing,
                                       cell_line,
                                       image_format),
                    dpi=300, bbox_inches='tight', transparent=False)
        plt.savefig('{}{}{}.eps'.format(output_dir, preprocessing,
                                        cell_line),
                    dpi=300, bbox_inches='tight', transparent=False)
        plt.close()
        plt.plot(y_train[0:50], '-o')
        plt.plot(predicted[0:50], '-o')
        plt.legend(['Observed', 'Predicted'], loc='upper right')
        plt.title('Comparison of observed values and predicted values by FCNN')
        plt.savefig('{}{}{}comaprison_r_p.{}'.format(output_dir, preprocessing,
                                                     cell_line,
                                                     image_format),
                    dpi=300, bbox_inches='tight', transparent=False)
        plt.savefig('{}{}{}comaprison_r_p.eps'.format(output_dir, preprocessing,
                                                      cell_line),
                    dpi=300, bbox_inches='tight', transparent=False)
        plt.close()

    if preprocessing == 'min max normalization':
        scale_denominator = (max_init - min_init)
        unscaled_predicted = predicted * scale_denominator + min_init
        unscaled_y_train = y_train * scale_denominator + min_init
        print(mean_squared_error(unscaled_predicted,
                                 unscaled_y_train))
        print(mean_squared_error(unscaled_predicted,
                                 unscaled_y_train))
        plt.scatter(y_train, predicted, s=0.1)
        plt.title('Normalized predicted values with respect to the normalized observed values')
        plt.ylabel('Predicted vlaues')
        plt.xlabel('Observed values')
        plt.axis('square')
        plt.savefig('{}{}distribution_performance.{}'.format(output_dir,
                    preprocessing, image_format),
                    dpi=300, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(5, 5))
        plt.plot([p1, p2], [p1, p2], 'w-')
        df = pd.read_csv('{}'.format(args.listfile), compression='gzip')
        masks = pd.read_csv('data/hg19_2000_no_N_inside.csv')
        print('Number of NANs is {}'.format(masks['signal'].sum()))
        df.loc[~masks['signal'].astype(bool)] = np.nan
        df = df.dropna()
        unscaled_y = df['initiation'].to_numpy()
        min_init_non_zero = np.min(unscaled_y[np.nonzero(unscaled_y)])
        plt.hist2d(np.log10(unscaled_y_train.ravel() + min_init_non_zero),
                   np.log10(unscaled_predicted.ravel() + min_init_non_zero),
                   bins=[300, 300], cmap=plt.cm.nipy_spectral,
                   norm=matplotlib.colors.LogNorm(
                   vmin=None, vmax=None, clip=False))
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.clim(vmin=10, vmax=10**3)
        plt.colorbar()
        plt.xlabel('Log(observed values + min(observed values))')
        plt.ylabel('Log(predicted values+ min(observed values))')
        plt.title('Predicted values with respect to the observed values for {}'.format(
            cell_line))
        plt.savefig('{}{}{}.{}'.format(output_dir, preprocessing,
                                       cell_line,
                                       image_format),
                    dpi=300, bbox_inches='tight', transparent=False)
        plt.close()
        plt.plot(y_train[0:50], '-o')
        plt.plot(predicted[0:50], '-o')
        plt.legend(['Observed', 'Predicted'], loc='upper right')
        plt.title('Comparison of observed values and predicted values by FCNN')
        plt.savefig('{}{}{}comaprison_r_p.{}'.format(output_dir, preprocessing,
                                                     cell_line,
                                                     image_format),
                    dpi=300, bbox_inches='tight', transparent=False)
        plt.close()


def interpret(model, X, output_dir, cell_line, marks):
    ig = IntegratedGradients(model)
    rows = []
    baseline = torch.zeros(1, 11).to(device)
    for input in X:
        input.reshape(1, 11)
        attributions = ig.attribute(
            input, baseline, target=0, return_convergence_delta=False)
        rows.append([attributions])
    attributions = pd.DataFrame(rows, columns=marks).to_csv(
        '{}{}_attributions.csv'.format(output_dir,
                                       cell_line))
    print('IG Attributions:', attributions)
    # print('Convergence Delta:', delta)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--method', type=str, default='FCNN')
    parser.add_argument('--preprocessing', type=str, default='log to raw')
    parser.add_argument('--max_epoch', type=int, default=300)
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
    min_init = np.min(df['initiation'])
    max_init = np.max(df['initiation'])
    print(df)
    print(torch.cuda.is_available())
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    writer = SummaryWriter(args.output_dir)
    if args.method == 'FCNN':
        if args.preprocessing == 'log to raw':
            for i in args.marks:
                df[i] = df[i] + np.min(df[i][(df[i] != 0)])
                df[i] = np.log10(df[i])
        if args.preprocessing == 'log to log':
            for i in args.marks + ['initiation']:
                df[i] = df[i] + np.min(df[i][(df[i] != 0)])
                df[i] = np.log10(df[i])
        if args.preprocessing == 'min max normalization':
            for i in args.marks + ['initiation']:
                df[i] = (df[i] - np.min(df[i])) / (
                    np.max(df[i]) - np.min(df[i]))
        if args.preprocessing == 'raw to log':
            df['initiation'] = df['initiation'] + np.min(df['initiation'][(df['initiation'] != 0)])
            df['initiation'] = np.log10(df['initiation'])
        if args.preprocessing == 'raw to raw':
            pass
        X_train = df.loc[df['chrom'] != 'chr1', args.marks].to_numpy()
        print(X_train.shape)
        y_train = df.loc[df['chrom'] != 'chr1', args.output].to_numpy()
        print(y_train.shape)
        X_test = df.loc[df['chrom'] == 'chr1', args.marks].to_numpy()
        y_test = df.loc[df['chrom'] == 'chr1', args.output].to_numpy()

        X_train, y_train = shuffle(X_train, y_train)
        X_val = torch.tensor(X_train[0:100000], dtype=float32).to(device)
        y_val = torch.tensor(y_train[0:100000], dtype=float32).to(device)
        X_train = torch.tensor(X_train[100000:], dtype=float32).to(device)
        X_test = torch.tensor(X_test, dtype=float32).to(device)
        y_train = torch.tensor(y_train[100000:], dtype=float32).to(device)
        train_ds = TensorDataset(X_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size)
        valid_ds = TensorDataset(X_val, y_val)
        valid_dl = DataLoader(valid_ds, batch_size=args.batch_size)
        model, opt = mlp()
        model = nn.DataParallel(model)
        model = model.to(device)
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
        # input = torch.rand(1, 11).to(device)
        model, val_loss_list, train_loss_list = fit(args.max_epoch, model,
                                                    loss_func, opt, train_dl,
                                                    valid_dl)
        torch.save(model.state_dict(), '{}{}'.format(
            args.output_dir, 'model_weights.pth'))
        plt.plot(train_loss_list)
        plt.plot(val_loss_list)
        plt.title('Loss during training')
        plt.ylabel('Mean Squared Error')
        plt.xlabel('Epochs')
        plt.scatter(np.argmin(val_loss_list),
                    np.min(val_loss_list), facecolors='none',
                    edgecolors='chocolate', s=50)
        plt.legend(['training', 'validation'], loc='upper right')
        plt.savefig('{}loss.{}'.format(args.output_dir, args.image_format),
                    dpi=300, bbox_inches='tight')
        plt.close()
        X = df[args.marks].to_numpy()
        X = torch.tensor(X, dtype=float32).to(device)
        predicted_test = model(X_test).cpu().detach().numpy()
        predicted = model(X_train).cpu().detach().numpy()
        y_train = y_train.cpu().detach().numpy()
        pd.DataFrame(predicted, columns=['predictions']).to_csv(
            '{}{}_predicted_train.csv'.format(args.output_dir,
                                              args.cell_line))
        pd.DataFrame(predicted_test, columns=['predictions']).to_csv(
            '{}{}_predicted_test.csv'.format(args.output_dir,
                                             args.cell_line))
        pd.DataFrame(y_train, columns=['observed_values']).to_csv(
            '{}{}_observed_train.csv'.format(args.output_dir,
                                             args.cell_line))
        pd.DataFrame(y_test, columns=['observed_values']).to_csv(
            '{}{}_observed_test.csv'.format(args.output_dir,
                                            args.cell_line))
        # p1 = max(max(predicted), max(y_train))
        # p2 = min(min(predicted), min(y_train))
        p1 = -2
        p2 = 2
        predicted1 = model(X).cpu().detach().numpy()
        df['predicted'] = predicted1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(df.index[df['chrom'] == 'chr1']*2000),
                      y=list(df.loc[df['chrom'] == 'chr1', 'initiation']),
                      name='PODLS'))
        fig.add_trace(go.Scatter(x=list(df.index[df['chrom'] == 'chr1']*2000),
                      y=list(df.loc[df['chrom'] == 'chr1', 'predicted']),
                      name='Predictions from FCNN'))
        fig.add_trace(go.Scatter(x=list(df.index[df['chrom'] == 'chr1']*2000),
                      y=list(np.abs(df.loc[df['chrom'] == 'chr1', 'predicted'
                                           ].to_numpy() - df.loc[
                                            df['chrom'] == 'chr1',
                                            'initiation'].to_numpy())),
                                            name='Absolute error'))
        fig.write_html("development/profile.html")
        plt.plot([p1, p2], [p1, p2], '-', color='orange')
        report(predicted, predicted_test, y_train, y_test, min_init, max_init,
               preprocessing=args.preprocessing, output_dir=args.output_dir,
               image_format=args.image_format, cell_line=args.cell_line)
        # interpret(model, X, output_dir=args.output_dir, cell_line=args.cell_line, marks=args.marks)
    if args.method == 'log FCNN Gridsearch':
        for i in args.marks:
            df[i] = df[i] + np.min(df[i][(df[i] != 0)])
            df[i] = np.log10(df[i])
        X_train = df.loc[df['chrom'] != 'chr1', args.marks].to_numpy()
        print(X_train.shape)
        y_train = df.loc[df['chrom'] != 'chr1', args.output].to_numpy()
        print(y_train.shape)
        X_test = df.loc[df['chrom'] == 'chr1', args.marks].to_numpy()
        y_test = df.loc[df['chrom'] == 'chr1', args.output].to_numpy()
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_val = torch.tensor(X_train[0:100000], dtype=float32)
        y_val = torch.tensor(y_train[0:100000], dtype=float32)
        X_train = torch.tensor(X_train[100000:], dtype=float32)
        y_train = torch.tensor(y_train[100000:], dtype=float32)
        train_ds = TensorDataset(X_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size)
        valid_ds = TensorDataset(X_val, y_val)
        valid_dl = DataLoader(valid_ds, batch_size=args.batch_size)

        def train_mlp(config, checkpoint_dir='development/', data_dir=None):
            # net = MLP(config["units"], config["l1"], config["l2"])
            # net = MLP(config["units"])
            net = MLP(config["units"])
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda:0"
                if torch.cuda.device_count() > 1:
                    net = nn.DataParallel(net)
            net.to(device)

            criterion = nn.MSELoss()
            # optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)
            # optimizer = optim.Adam(net.parameters(), lr=config["lr"])
            # optimizer = optim.Adam(net.parameters())
            optimizer = optim.Adam()

            if checkpoint_dir:
                model_state, optimizer_state = torch.load(
                    os.path.join(checkpoint_dir, "checkpoint"))
                net.load_state_dict(model_state)
                optimizer.load_state_dict(optimizer_state)
            for epoch in range(160):  # loop over the dataset multiple times
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
                                                        running_loss /
                                                        epoch_steps))
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
                    torch.save((net.state_dict(), optimizer.state_dict()),
                               path)

                tune.report(loss=(val_loss / val_steps))
                print("Finished Training")

        def test_loss(net, device="cpu"):
            with torch.no_grad():
                outputs = net(X_val)
                outputs = outputs.cpu().numpy()
            return mean_squared_error(outputs, y_val.cpu().numpy())

        # def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
        #     config = {
        #         "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        #         "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        #         "lr": tune.loguniform(1e-4, 1e-1),
        #         "units": tune.qrandint(40, 100),
        #         "batch_size": tune.choice([2, 4, 8, 16])
        #     }
        def main1(num_samples=20, max_num_epochs=160, gpus_per_trial=4):
            config = {
                "units": tune.qrandint(40, 1000),
                "batch_size": tune.choice([32, 64, 128]),
                # "lr": tune.loguniform(1e-4, 1e-1)
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

            # best_trained_model = MLP(best_trial.config["units"] ,best_trial.config["l1"], best_trial.config["l2"])
            best_trained_model = MLP(best_trial.config["units"])
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

        main1(num_samples=20, max_num_epochs=160, gpus_per_trial=4)
        