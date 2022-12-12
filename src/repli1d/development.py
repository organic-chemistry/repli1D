import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.metrics import make_scorer
from repli1d.models import mlp
import pickle


def histogram(preprocessing, cell_line, output_dir, image_format):

    observed_test = pd.read_csv(
        'development/{}_observed_test.csv'.format(
            args.cell_line))['observed_values'].to_numpy()
    predicted_test = pd.read_csv(
        'development/{}_predicted_test.csv'.format(
            args.cell_line))['predictions'].to_numpy()
    df = pd.read_csv('{}'.format(args.listfile), compression='gzip')
    observed_train = pd.read_csv(
        'development/{}_observed_train.csv'.format(
            args.cell_line))['observed_values'].to_numpy()
    predicted_train = pd.read_csv(
        'development/{}_predicted_train.csv'.format(
            args.cell_line))['predictions'].to_numpy()
    # df_ch = pd.read_csv(
    #     'development/{}_df_predicted.csv'.format(
    #         args.cell_line))
    # chr2_observed = df_ch.loc[df_ch['chrom'] == 'chr2', 'initiation'].to_numpy()
    # chr2_predicted = df_ch.loc[df_ch['chrom'] == 'chr2', 'predicted'].to_numpy()
    # print(chr2_observed.shape)
    # print(chr2_predicted.shape)
    observed_train, predicted_train = shuffle(observed_train, predicted_train, random_state=42)
    df = pd.read_csv('{}'.format(args.listfile), compression='gzip')
    masks = pd.read_csv('data/hg19_2000_no_N_inside.csv')
    print('Number of NANs is {}'.format(masks['signal'].sum()))
    df.loc[~masks['signal'].astype(bool)] = np.nan
    df = df.dropna()
    y_train = df['initiation'].to_numpy()
    min_init_non_zero = np.min(y_train[np.nonzero(y_train)])
    max_init = np.max(df['initiation'])
    min_init = np.min(df['initiation'])
    if preprocessing == 'min max normalization':
        scale_denominator = (max_init - min_init)
        unscaled_predicted_test = predicted_test * scale_denominator + min_init
        unscaled_y_test = observed_test * scale_denominator + min_init
        unscaled_predicted_train = predicted_train * scale_denominator + min_init
        unscaled_y_train = observed_train * scale_denominator + min_init
        log_un_y_test = np.log10(unscaled_y_test.ravel() + min_init_non_zero)
        log_un_predicted_test = np.log10(unscaled_predicted_test.ravel() +
                                    min_init_non_zero)
        log_un_y_train = np.log10(unscaled_y_train.ravel() + min_init_non_zero)
        log_un_predicted_train = np.log10(unscaled_predicted_train.ravel() +
                                    min_init_non_zero)
    if preprocessing == 'raw to raw' or preprocessing == 'log to raw':
        unscaled_predicted_train = predicted_train
        unscaled_y_train = observed_train
        unscaled_predicted_test = predicted_test
        unscaled_y_test = observed_test
        log_un_y_test = np.log10(observed_test.ravel() + min_init_non_zero)
        log_un_predicted_test = np.log10(predicted_test.ravel() + min_init_non_zero)
        log_un_y_train = np.log10(observed_train.ravel() + min_init_non_zero)
        log_un_predicted_train = np.log10(predicted_train.ravel() + min_init_non_zero)
    if preprocessing == 'raw to log' or preprocessing == 'log to log':
        unscaled_predicted_train = 10**predicted_train
        unscaled_y_train = 10**observed_train
        unscaled_predicted_test = 10**predicted_test
        unscaled_y_test = 10**observed_test
        log_un_y_test = observed_test
        log_un_predicted_test = predicted_test
        log_un_y_train = observed_train
        log_un_predicted_train = predicted_train
    unscaled_observed_val = unscaled_y_train[0:100000]
    unscaled_predicted_val = unscaled_predicted_train[0:100000]
    unscaled_y_train = unscaled_y_train[200000:]
    unscaled_predicted_train = unscaled_predicted_train[200000:]
    print('validation loss:', mean_squared_error(unscaled_predicted_val, unscaled_observed_val))
    print('test loss:', mean_squared_error(unscaled_predicted_test, unscaled_y_test))
    # print('chromosome 2 loss:', mean_squared_error(chr2_observed, chr2_predicted))
    print('train loss:', mean_squared_error(unscaled_predicted_train, unscaled_y_train))
    print('loss of log values in test set:', mean_squared_error(log_un_predicted_test, log_un_y_test))
    print('loss of log values in train set:', mean_squared_error(log_un_predicted_train, log_un_y_train))
    print('correlation:', np.corrcoef(log_un_predicted_test, log_un_y_test))
    plt.figure(figsize=(5, 5))
    p1 = -2
    p2 = 2
    plt.plot([p1, p2], [p1, p2], 'w-')
    plt.hist2d(log_un_y_test,
               log_un_predicted_test,
               bins=[400, 400], cmap=plt.cm.nipy_spectral,
               norm=matplotlib.colors.LogNorm(
                vmin=None, vmax=None, clip=False))

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.clim(vmin=1, vmax=10**3)
    plt.colorbar()
    plt.xlabel('Log(observed values + min(observed values))')
    plt.ylabel('Log(predicted values + min(observed values))')
    plt.title('Predicted values with respect to the observed values for {}'.format(
        cell_line))
    plt.savefig('{}{}{}.{}'.format(output_dir, preprocessing,
                                   cell_line,
                                   image_format),
                dpi=300, bbox_inches='tight', transparent=False)
    plt.close()

def histogram_vs(preprocessing, cell_line, output_dir, image_format):

    predicted_test_log_log = pd.read_csv(
        'development/{}_predicted_test_log_to_log.csv'.format(
            args.cell_line))['predictions'].to_numpy()
    predicted_test_log_raw = pd.read_csv(
        'development/{}_predicted_test_log_to_raw.csv'.format(
            args.cell_line))['predictions'].to_numpy()

    df = pd.read_csv('{}'.format(args.listfile), compression='gzip')
    masks = pd.read_csv('data/hg19_2000_no_N_inside.csv')
    print('Number of NANs is {}'.format(masks['signal'].sum()))
    df.loc[~masks['signal'].astype(bool)] = np.nan
    df = df.dropna()
    y_train = df['initiation'].to_numpy()
    min_init_non_zero = np.min(y_train[np.nonzero(y_train)])
    predicted_test_log_raw = np.log10(predicted_test_log_raw.ravel() + min_init_non_zero)
    predicted_test_log_log = np.log10(predicted_test_log_log.ravel() + min_init_non_zero)
    plt.figure(figsize=(6, 6))
    p1 = -2
    p2 = 2
    plt.plot([p1, p2], [p1, p2], 'w-')
    plt.hist2d(predicted_test_log_log,
               predicted_test_log_raw,
               bins=[400, 400], cmap=plt.cm.nipy_spectral,
            
            
               norm=matplotlib.colors.LogNorm(
                vmin=None, vmax=None, clip=False))

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.clim(vmin=1, vmax=10**3)
    plt.colorbar()
    plt.xlabel('RF Log to raw: Log(predicted values + min(observed values))')
    plt.ylabel('FCNN Log to raw: Log(predicted values + min(observed values))')
    plt.title('Comparison of predicted values for two preprocessing methods for {}'.format(
        cell_line))
    plt.savefig('{}{}.{}'.format(output_dir,
                                   cell_line,
                                   image_format),
                dpi=300, bbox_inches='tight', transparent=False)
    plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--preprocessing', type=str, default='log')
    parser.add_argument('--method', type=str, default='evaluation hist2d')
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
    print(df)

    if args.preprocessing == 'log to log RF Gridsearch':
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
        regr = RandomForestRegressor(n_jobs=1, random_state=0)
        params = {
            'max_depth': [2, 3, 5, 10, 20],
            'min_samples_leaf': [5, 10, 20, 50, 100, 200],
            'n_estimators': [10, 25, 30, 50, 100, 200]
        }
        mse = make_scorer(mean_squared_error, greater_is_better=False)
        grid_search = GridSearchCV(estimator=regr,
                                   param_grid=params,
                                   cv=4,
                                   n_jobs=80, verbose=1,
                                   scoring=mse)
        grid_search.fit(X_train, y_train.ravel())
        print(grid_search.best_score_)
        # print(mean_squared_error(regr.predict(X_train), y_train))
        print(grid_search.best_estimator_)

    if args.preprocessing == 'log to raw RF Gridsearch':
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
        regr = RandomForestRegressor(n_jobs=1, random_state=0)
        params = {
            'max_depth': [2, 3, 5, 10, 20],
            'min_samples_leaf': [5, 10, 20, 50, 100, 200],
            'n_estimators': [10, 25, 30, 50, 100, 200]
        }
        mse = make_scorer(mean_squared_error, greater_is_better=False)
        grid_search = GridSearchCV(estimator=regr,
                                   param_grid=params,
                                   cv=4,
                                   n_jobs=80, verbose=1,
                                   scoring=mse)
        grid_search.fit(X_train, y_train.ravel())
        print(grid_search.best_score_)
        # print(mean_squared_error(regr.predict(X_train), y_train))
        print(grid_search.best_estimator_)

    if args.preprocessing == 'raw to log RF Gridsearch':
        for i in args.output:
            df[i] = df[i] + np.min(df[i][(df[i] != 0)])
            df[i] = np.log10(df[i])
        X_train = df.loc[df['chrom'] != 'chr1', args.marks].to_numpy()
        print(X_train.shape)
        y_train = df.loc[df['chrom'] != 'chr1', args.output].to_numpy()
        print(y_train.shape)
        X_test = df.loc[df['chrom'] == 'chr1', args.marks].to_numpy()
        y_test = df.loc[df['chrom'] == 'chr1', args.output].to_numpy()
        X_train, y_train = shuffle(X_train, y_train)
        regr = RandomForestRegressor(n_jobs=1, random_state=0)
        params = {
            'max_depth': [2, 3, 5, 10, 20],
            'min_samples_leaf': [5, 10, 20, 50, 100, 200],
            'n_estimators': [10, 25, 30, 50, 100, 200]
        }
        mse = make_scorer(mean_squared_error, greater_is_better=False)
        grid_search = GridSearchCV(estimator=regr,
                                   param_grid=params,
                                   cv=4,
                                   n_jobs=80, verbose=1,
                                   scoring=mse)
        grid_search.fit(X_train, y_train.ravel())
        print(grid_search.best_score_)
        # print(mean_squared_error(regr.predict(X_train), y_train))
        print(grid_search.best_estimator_)

    if args.preprocessing == 'raw to raw RF Gridsearch':
        X_train = df.loc[df['chrom'] != 'chr1', args.marks].to_numpy()
        print(X_train.shape)
        y_train = df.loc[df['chrom'] != 'chr1', args.output].to_numpy()
        print(y_train.shape)
        X_test = df.loc[df['chrom'] == 'chr1', args.marks].to_numpy()
        y_test = df.loc[df['chrom'] == 'chr1', args.output].to_numpy()
        X_train, y_train = shuffle(X_train, y_train)
        regr = RandomForestRegressor(n_jobs=1, random_state=0)
        params = {
            'max_depth': [2, 3, 5, 10, 20],
            'min_samples_leaf': [5, 10, 20, 50, 100, 200],
            'n_estimators': [10, 25, 30, 50, 100, 200]
        }
        mse = make_scorer(mean_squared_error, greater_is_better=False)
        grid_search = GridSearchCV(estimator=regr,
                                   param_grid=params,
                                   cv=4,
                                   n_jobs=80, verbose=1,
                                   scoring=mse)
        grid_search.fit(X_train, y_train.ravel())
        print(grid_search.best_score_)
        # print(mean_squared_error(regr.predict(X_train), y_train))
        print(grid_search.best_estimator_)

    if args.preprocessing == 'log to log RF':
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
        regr = RandomForestRegressor(max_depth=20, min_samples_leaf=20,
                                     n_estimators=500, n_jobs=20,
                                     random_state=0)
        regr.fit(X_train, y_train.ravel())
        predicted_test = regr.predict(X_test)
        predicted = regr.predict(X_train)
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
        print(mean_squared_error(10**predicted, 10**y_train))
        print(mean_squared_error(10**predicted_test, 10**y_test))
        print(mean_squared_error(predicted, y_train))
        print(mean_squared_error(predicted_test, y_test))
        print(regr.feature_importances_)
        p1 = max(max(predicted), max(y_train))
        p2 = min(min(predicted), min(y_train))
        plt.plot([p1, p2], [p1, p2], '-', color='orange')
        plt.scatter(y_train.ravel(), predicted, s=0.1, alpha=0.05)
        plt.title(
            'Log of predicted values with respect to the log of observed values')
        plt.ylabel('Predicted vlaues')
        plt.xlabel('Observed values')
        plt.axis('square')
        plt.savefig('{}distribution_performance.{}'.format(args.output_dir,
                    args.image_format),
                    dpi=300, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(12, 10))
        plt.plot([p1, p2], [p1, p2], 'w-')
        plt.hist2d(y_train.ravel(), predicted,
                   bins=[np.histogram_bin_edges(y_train, bins='auto'),
                         np.histogram_bin_edges(predicted,
                                                bins='auto')],
                   cmap=plt.cm.nipy_spectral)
        plt.colorbar()
        plt.xlabel('Observed values')
        plt.ylabel('Predicted values')
        plt.title('Log of predicted values with respect to the log of ' +
                  'observed values for {}'.format(args.cell_line))
        plt.savefig('{}{}.{}'.format(args.output_dir,
                                     args.cell_line,
                                     args.image_format),
                    dpi=300, bbox_inches='tight', transparent=False)
        plt.close()
        plt.plot(y_train[0:50], '-o')
        plt.plot(predicted[0:50], '-o')
        plt.legend(['Real', 'Predicted'], loc='upper right')
        plt.title('comparison of observed values and predicted values by RF')
        plt.savefig('{}{}comaprison_r_p.{}'.format(args.output_dir,
                                                   args.cell_line,
                                                   args.image_format),
                    dpi=300, bbox_inches='tight', transparent=False)
        plt.close()
        filename = 'development/finalized_model.sav'
        pickle.dump(regr, open(filename, 'wb'))

    if args.preprocessing == 'log to raw RF':
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
        regr = RandomForestRegressor(max_depth=20, min_samples_leaf=20,
                                     n_estimators=500, n_jobs=20,
                                     random_state=0)
        regr.fit(X_train, y_train.ravel())
        predicted_test = regr.predict(X_test)
        predicted = regr.predict(X_train)
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
        print(mean_squared_error(predicted, y_train))
        print(mean_squared_error(predicted_test, y_test))
        print(regr.feature_importances_)
        p1 = max(max(predicted), max(y_train))
        p2 = min(min(predicted), min(y_train))
        plt.plot([p1, p2], [p1, p2], '-', color='orange')
        plt.scatter(y_train.ravel(), predicted, s=0.1, alpha=0.05)
        plt.title('Predicted values with respect to the observed values')
        plt.ylabel('Predicted vlaues')
        plt.xlabel('Observed values')
        plt.axis('square')
        plt.savefig('{}distribution_performance.{}'.format(args.output_dir,
                                                           args.image_format),
                    dpi=300, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(10, 10))
        plt.plot([p1, p2], [p1, p2], 'w-')
        plt.hist2d(y_train.ravel(), predicted,
                   bins=[np.histogram_bin_edges(y_train, bins='auto'),
                         np.histogram_bin_edges(predicted,
                                                bins='auto')],
                   cmap=plt.cm.nipy_spectral)
        plt.colorbar()
        plt.xlabel('Observed values')
        plt.ylabel('Predicted values')
        plt.title('Predicted values with respect to the observed values for {}'.format(
            args.cell_line))
        plt.savefig('{}{}.{}'.format(args.output_dir,
                                     args.cell_line,
                                     args.image_format),
                    dpi=300, bbox_inches='tight', transparent=False)
        plt.close()
        plt.plot(y_train[0:50], '-o')
        plt.plot(predicted[0:50], '-o')
        plt.legend(['Observed', 'Predicted'], loc='upper right')
        plt.title('Comparison of observed values and predicted values by RF')
        plt.savefig('{}{}comaprison_r_p.{}'.format(args.output_dir,
                                                   args.cell_line,
                                                   args.image_format),
                    dpi=300, bbox_inches='tight', transparent=False)
        plt.close()
        filename = 'development/finalized_model.sav'
        pickle.dump(regr, open(filename, 'wb'))

    if args.preprocessing == 'raw to log RF':
        for i in args.output:
            df[i] = df[i] + np.min(df[i][(df[i] != 0)])
            df[i] = np.log10(df[i])
        X_train = df.loc[df['chrom'] != 'chr1', args.marks].to_numpy()
        print(X_train.shape)
        y_train = df.loc[df['chrom'] != 'chr1', args.output].to_numpy()
        print(y_train.shape)
        X_test = df.loc[df['chrom'] == 'chr1', args.marks].to_numpy()
        y_test = df.loc[df['chrom'] == 'chr1', args.output].to_numpy()
        X_train, y_train = shuffle(X_train, y_train)
        regr = RandomForestRegressor(max_depth=20, min_samples_leaf=20,
                                     n_estimators=500, n_jobs=20,
                                     random_state=0)
        regr.fit(X_train, y_train.ravel())
        predicted_test = regr.predict(X_test)
        predicted = regr.predict(X_train)
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
        print(mean_squared_error(10**predicted, 10**y_train))
        print(mean_squared_error(10**regr.predict(X_test), 10**y_test))
        print(mean_squared_error(predicted, y_train))
        print(mean_squared_error(predicted_test, y_test))
        print(regr.feature_importances_)
        p1 = max(max(predicted), max(y_train))
        p2 = min(min(predicted), min(y_train))
        plt.plot([p1, p2], [p1, p2], '-', color='orange')
        plt.scatter(y_train.ravel(), predicted, s=0.1, alpha=0.05)
        plt.title(
            'Log of predicted values with respect to the log of observed values')
        plt.ylabel('Predicted vlaues')
        plt.xlabel('Observed values')
        plt.axis('square')
        plt.savefig('{}distribution_performance.{}'.format(args.output_dir,
                                                           args.image_format),
                    dpi=300, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(10, 10))
        plt.plot([p1, p2], [p1, p2], 'w-')
        plt.hist2d(y_train.ravel(), predicted,
                   bins=[np.histogram_bin_edges(y_train, bins='auto'),
                         np.histogram_bin_edges(predicted,
                                                bins='auto')],
                   cmap=plt.cm.nipy_spectral)
        plt.colorbar()
        plt.xlabel('Observed values')
        plt.ylabel('Predicted values')
        plt.title('Predicted values with respect to the log of ' +
                  'observed values for {}'.format(args.cell_line))
        plt.savefig('{}{}.{}'.format(args.output_dir,
                                     args.cell_line,
                                     args.image_format),
                    dpi=300, bbox_inches='tight', transparent=False)
        plt.close()
        plt.plot(y_train[0:50], '-o')
        plt.plot(predicted[0:50], '-o')
        plt.legend(['Observed', 'Predicted'], loc='upper right')
        plt.title('Comparison of observed values and predicted values by RF')
        plt.savefig('{}{}comaprison_r_p.{}'.format(args.output_dir,
                                                   args.cell_line,
                                                   args.image_format),
                    dpi=300, bbox_inches='tight', transparent=False)
        plt.close()
        filename = 'development/finalized_model.sav'
        pickle.dump(regr, open(filename, 'wb'))

    if args.preprocessing == 'raw to raw RF':
        X_train = df.loc[df['chrom'] != 'chr1', args.marks].to_numpy()
        print(X_train.shape)
        y_train = df.loc[df['chrom'] != 'chr1', args.output].to_numpy()
        print(y_train.shape)
        X_test = df.loc[df['chrom'] == 'chr1', args.marks].to_numpy()
        y_test = df.loc[df['chrom'] == 'chr1', args.output].to_numpy()
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        regr = RandomForestRegressor(max_depth=20, min_samples_leaf=20,
                                     n_estimators=200, n_jobs=-1,
                                     random_state=0)
        regr.fit(X_train, y_train.ravel())
        predicted_test = regr.predict(X_test)
        predicted = regr.predict(X_train)
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
        print(mean_squared_error(predicted, y_train))
        print(mean_squared_error(predicted_test, y_test))
        print(regr.feature_importances_)
        p1 = max(max(predicted), max(y_train))
        p2 = min(min(predicted), min(y_train))
        plt.plot([p1, p2], [p1, p2], '-', color='orange')
        plt.scatter(y_train.ravel(), predicted, s=0.1, alpha=0.05)
        plt.title('Predicted values with respect to the observed values')
        plt.ylabel('Predicted values')
        plt.xlabel('Observed values')
        plt.axis('square')
        plt.savefig('{}distribution_performance.{}'.format(args.output_dir,
                                                           args.image_format),
                    dpi=300, bbox_inches='tight')
        plt.close()
        # plt.figure(figsize=(10, 10))

        plt.plot([p1, p2], [p1, p2], 'w-')
        plt.hist2d(np.log10(y_train.ravel()+1), np.log10(predicted + 1),
                   bins=[100, 100],
                   cmap=plt.cm.nipy_spectral, norm=matplotlib.colors.LogNorm(
                       vmin=None, vmax=None, clip=False))
        # plt.yscale('log')
        # plt.ylim([0, 4])
        # plt.xlim([0, 4])
        # plt.xscale('log')
        plt.colorbar()
        plt.xlabel('Log(observed values+1)')
        plt.ylabel('Log(predicted values+1)')
        plt.title('Predicted values with respect to the ' +
                  'observed values for {}'.format(args.cell_line))
        plt.savefig('{}{}.{}'.format(args.output_dir,
                                     args.cell_line,
                                     args.image_format),
                    dpi=300, bbox_inches='tight', transparent=False)
        plt.close()
        plt.plot(y_train[0:50], '-o')
        plt.plot(predicted[0:50], '-o')
        plt.legend(['Observed', 'Predicted'], loc='upper right')
        plt.title('Comparison of observed values and predicted values by RF')
        plt.savefig('{}{}comaprison_r_p.{}'.format(args.output_dir,
                                                   args.cell_line,
                                                   args.image_format),
                    dpi=300, bbox_inches='tight', transparent=False)
        plt.show()
        filename = 'development/finalized_model.sav'
        pickle.dump(regr, open(filename, 'wb'))

    if args.preprocessing == 'log to log FCNN':
        for i in args.marks + args.output:
            df[i] = df[i] + np.min(df[i][(df[i] != 0)])
            df[i] = np.log10(df[i])
        # X_train = df.loc[(df['chrom'] != 'chr1') & (df['chrom'] != 'chr2'),
        #                  args.marks].to_numpy()
        # y_train = df.loc[(df['chrom'] != 'chr1') & (df['chrom'] != 'chr2'),
        #                  args.output].to_numpy()
        X_train = df.loc[df['chrom'] != 'chr1', args.marks].to_numpy()
        print(X_train.shape)
        y_train = df.loc[df['chrom'] != 'chr1', args.output].to_numpy()
        print(y_train.shape)
        X_test = df.loc[df['chrom'] == 'chr1', args.marks].to_numpy()
        y_test = df.loc[df['chrom'] == 'chr1', args.output].to_numpy()
        X_train, y_train = shuffle(X_train, y_train)
        X_train = tf.convert_to_tensor(X_train, np.float32)
        y_train = tf.convert_to_tensor(y_train, np.float32)
        X_test = tf.convert_to_tensor(X_test, np.float32)
        y_test = tf.convert_to_tensor(y_test, np.float32)
        # X_val = df.loc[df['chrom'] == 'chr2', args.marks].to_numpy()
        # y_val = df.loc[df['chrom'] == 'chr2', args.output].to_numpy()
        model = mlp(X_train, y_train)
        tf.keras.utils.plot_model(model,
                                  to_file='{}{}FCNN_architecture.png'.format(
                                      args.output_dir,
                                      args.preprocessing),
                                  show_shapes=True)
        checkpoint_filepath = r'{}{}FCNN_K562_marks.mdl_wts.hdf5'.format(
            args.output_dir, args.preprocessing)
        mcp_save = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_best_only=True,
            monitor='val_loss', mode='min')
        model.compile(loss='mse', optimizer='adam',
                      metrics=['mse', 'mae',
                               tf.keras.metrics.RootMeanSquaredError()])
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        history = model.fit(X_train, y_train, epochs=2000,
                            verbose=1, validation_split=0.07,
                            callbacks=[callback, mcp_save],
                            batch_size=128)  # validation_data=(X_val, y_val),
        plt.plot(history.history['loss'], c='red')
        plt.plot(history.history['val_loss'], c='blue')
        plt.scatter(np.argmin(history.history['val_loss']),
                    np.min(history.history['val_loss']), facecolors='none',
                    edgecolors='chocolate', s=50)
        plt.title('Fully Connected Neural Network Loss')
        plt.ylabel('Loss (Mean Squared Error)')
        plt.xlabel('Epoch')
        plt.legend(['training', 'validation'], loc='upper right')
        plt.savefig('{}FCNN_Loss.png'.format(args.output_dir),
                    dpi=300, bbox_inches='tight')
        hist = pd.DataFrame(history.history)
        with open('{}{}history.csv'.format(args.output_dir,
                                           args.preprocessing), mode='w') as f:
            hist.to_csv(f)
    if args.preprocessing == 'log to raw FCNN':
        for i in args.marks:
            df[i] = df[i] + np.min(df[i][(df[i] != 0)])
            df[i] = np.log10(df[i])
        # X_train = df.loc[(df['chrom'] != 'chr1') & (df['chrom'] != 'chr2'),
        #                  args.marks].to_numpy()
        # y_train = df.loc[(df['chrom'] != 'chr1') & (df['chrom'] != 'chr2'),
        #                  args.output].to_numpy()
        X_train = df.loc[df['chrom'] != 'chr1', args.marks].to_numpy()
        print(X_train.shape)
        y_train = df.loc[df['chrom'] != 'chr1', args.output].to_numpy()
        print(y_train.shape)
        X_test = df.loc[df['chrom'] == 'chr1', args.marks].to_numpy()
        y_test = df.loc[df['chrom'] == 'chr1', args.output].to_numpy()
        X_train, y_train = shuffle(X_train, y_train)
        X_train = tf.convert_to_tensor(X_train, np.float32)
        y_train = tf.convert_to_tensor(y_train, np.float32)
        X_test = tf.convert_to_tensor(X_test, np.float32)
        y_test = tf.convert_to_tensor(y_test, np.float32)
        # X_val = df.loc[df['chrom'] == 'chr2', args.marks].to_numpy()
        # y_val = df.loc[df['chrom'] == 'chr2', args.output].to_numpy()
        model = mlp(X_train, y_train)
        tf.keras.utils.plot_model(model,
                                  to_file='{}{}FCNN_architecture.png'.format(
                                      args.output_dir,
                                      args.preprocessing),
                                  show_shapes=True)
        checkpoint_filepath = r'{}{}FCNN_K562_marks.mdl_wts.hdf5'.format(
            args.output_dir, args.preprocessing)
        mcp_save = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_best_only=True,
            monitor='val_loss', mode='min')
        model.compile(loss='mse', optimizer='adam',
                      metrics=['mse', 'mae',
                               tf.keras.metrics.RootMeanSquaredError()])
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        history = model.fit(X_train, y_train, epochs=2000,
                            verbose=1, validation_split=0.07,
                            callbacks=[callback, mcp_save],
                            batch_size=128)  # validation_data=(X_val, y_val),
        plt.plot(history.history['loss'], c='red')
        plt.plot(history.history['val_loss'], c='blue')
        plt.scatter(np.argmin(history.history['val_loss']),
                    np.min(history.history['val_loss']), facecolors='none',
                    edgecolors='chocolate', s=50)
        plt.title('Fully Connected Neural Network Loss')
        plt.ylabel('Loss (Mean Squared Error)')
        plt.xlabel('Epoch')
        plt.legend(['training', 'validation'], loc='upper right')
        plt.savefig('{}FCNN_Loss.png'.format(args.output_dir),
                    dpi=300, bbox_inches='tight')
        hist = pd.DataFrame(history.history)
        with open('{}{}history.csv'.format(args.output_dir,
                                           args.preprocessing), mode='w') as f:
            hist.to_csv(f)
        predicted = model.predict(X_train)
        print(mean_squared_error(predicted, y_train))
        print(mean_squared_error(model.predict(X_test), y_test))

    if args.preprocessing == 'log to log multi-GPU FCNN':
        for i in args.marks + args.output:
            df[i] = df[i] + np.min(df[i][(df[i] != 0)])
            df[i] = np.log10(df[i])
        # X_train = df.loc[(df['chrom'] != 'chr1') & (df['chrom'] != 'chr2'),
        #                  args.marks].to_numpy()
        # y_train = df.loc[(df['chrom'] != 'chr1') & (df['chrom'] != 'chr2'),
        #                  args.output].to_numpy()
        X_train = df.loc[df['chrom'] != 'chr1', args.marks].to_numpy()
        print(X_train.shape)
        y_train = df.loc[df['chrom'] != 'chr1', args.output].to_numpy()
        print(y_train.shape)
        X_test = df.loc[df['chrom'] == 'chr1', args.marks].to_numpy()
        y_test = df.loc[df['chrom'] == 'chr1', args.output].to_numpy()
        # X_val = df.loc[df['chrom'] == 'chr2', args.marks].to_numpy()
        # y_val = df.loc[df['chrom'] == 'chr2', args.output].to_numpy()
        strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))
        X_train, y_train = shuffle(X_train, y_train)
        num_val_samples = 10**5
        X_val = X_train[-num_val_samples:]
        y_val = y_train[-num_val_samples:]
        X_train = X_train[:-num_val_samples]
        y_train = y_train[:-num_val_samples]
        batch_size = 128
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (X_train, y_train)).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (X_val, y_val)).batch(batch_size)
        # Open a strategy scope.
        with strategy.scope():
            # Everything that creates variables should be under the strategy scope.
            # In general this is only model construction & `compile()`.
            model = mlp(X_train, y_train)
            tf.keras.utils.plot_model(model,
                                      to_file='{}{}FCNN_architecture.png'.format(
                                          args.output_dir, args.preprocessing),
                                      show_shapes=True)
            checkpoint_filepath = r'{}{}FCNN_K562_marks.mdl_wts.hdf5'.format(
                args.output_dir, args.preprocessing)
            mcp_save = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_best_only=True,
                monitor='val_loss', mode='min')
            model.compile(loss='mse', optimizer='adam',
                          metrics=['mse', 'mae',
                                   tf.keras.metrics.RootMeanSquaredError()])
            callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                        patience=3)
            history = model.fit(train_dataset, epochs=2000,
                                verbose=1, validation_data=val_dataset,
                                callbacks=[callback, mcp_save])
            # validation_data=(X_val, y_val),
        plt.plot(history.history['loss'], c='red')
        plt.plot(history.history['val_loss'], c='blue')
        plt.scatter(np.argmin(history.history['val_loss']),
                    np.min(history.history['val_loss']), facecolors='none',
                    edgecolors='chocolate', s=50)
        plt.title('Fully Connected Neural Network Loss')
        plt.ylabel('Loss (Mean Squared Error)')
        plt.xlabel('Epoch')
        plt.legend(['training', 'validation'], loc='upper right')
        plt.savefig('{}FCNN_Loss.png'.format(args.output_dir),
                    dpi=300, bbox_inches='tight')
        hist = pd.DataFrame(history.history)
        with open('{}{}history.csv'.format(args.output_dir,
                                           args.preprocessing), mode='w') as f:
            hist.to_csv(f)

    if args.preprocessing == 'min_max normalization':

        # X_train = df.loc[(df['chrom'] != 'chr1') & (df['chrom'] != 'chr2'),
        #                  args.marks].to_numpy()
        # y_train = df.loc[(df['chrom'] != 'chr1') & (df['chrom'] != 'chr2'),
        #                  args.output].to_numpy()
        X_train = df.loc[df['chrom'] != 'chr1', args.marks].to_numpy()
        print(X_train.shape)
        y_train = df.loc[df['chrom'] != 'chr1', args.output].to_numpy()
        print(y_train.shape)
        X_test = df.loc[df['chrom'] == 'chr1', args.marks].to_numpy()
        y_test = df.loc[df['chrom'] == 'chr1', args.output].to_numpy()
        # X_val = df.loc[df['chrom'] == 'chr2', args.marks].to_numpy()
        # y_val = df.loc[df['chrom'] == 'chr2', args.output].to_numpy()
        model = mlp(X_train, y_train)
        tf.keras.utils.plot_model(model,
                                  to_file='{}{}FCNN_architecture.png'.format(
                                      args.output_dir,
                                      args.preprocessing),
                                  show_shapes=True)
        checkpoint_filepath = r'{}{}FCNN_K562_marks.mdl_wts.hdf5'.format(
            args.output_dir, args.preprocessing)
        mcp_save = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_best_only=True,
            monitor='val_loss', mode='min')
        model.compile(loss='mse', optimizer='adam',
                      metrics=['mse', 'mae',
                               tf.keras.metrics.RootMeanSquaredError()])
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        X_train, y_train = shuffle(X_train, y_train)
        history = model.fit(X_train, y_train, epochs=2000,
                            verbose=1, validation_split=0.07,
                            callbacks=[callback, mcp_save],
                            batch_size=128)  # validation_data=(X_val, y_val),
        plt.plot(history.history['loss'], c='red')
        plt.plot(history.history['val_loss'], c='blue')
        plt.scatter(np.argmin(history.history['val_loss']),
                    np.min(history.history['val_loss']), facecolors='none',
                    edgecolors='chocolate', s=50)
        plt.title('Fully Connected Neural Network Loss')
        plt.ylabel('Loss (Mean Squared Error)')
        plt.xlabel('Epoch')
        plt.legend(['training', 'validation'], loc='upper right')
        plt.savefig('{}FCNN_Loss.png'.format(args.output_dir),
                    dpi=300, bbox_inches='tight')
        hist = pd.DataFrame(history.history)
        with open('{}{}history.csv'.format(args.output_dir,
                                           args.preprocessing), mode='w') as f:
            hist.to_csv(f)
    if args.method == 'evaluation hist2d':
        histogram(preprocessing=args.preprocessing, cell_line=args.cell_line,
                  output_dir=args.output_dir, image_format=args.image_format)
    if args.method == 'hist2d vs':
        histogram_vs(preprocessing=args.preprocessing, cell_line=args.cell_line,
                     output_dir=args.output_dir, image_format=args.image_format)
    if args.method == 'profiles for each chromosome':
        df = pd.read_csv("development/K562_df_predicted.csv")
        for ch in df.chrom.unique():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(df.index[df['chrom'] == ch]*2000),
                        y=list(df.loc[df['chrom'] == ch, 'initiation']),
                        name='PODLS'))
            fig.add_trace(go.Scatter(x=list(df.index[df['chrom'] == ch]*2000),
                        y=list(df.loc[df['chrom'] == ch, 'predicted']),
                        name='Predictions from CNN'))
            fig.add_trace(go.Scatter(x=list(df.index[df['chrom'] == ch]*2000),
                        y=list(np.abs(df.loc[df['chrom'] == ch, 'predicted'
                                            ].to_numpy() - df.loc[
                                                df['chrom'] == ch,
                                                'initiation'].to_numpy())),
                                                name='Absolute error'))
            fig.write_html("development/profile_{}.html".format(ch))         