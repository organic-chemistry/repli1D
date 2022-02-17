import argparse

import matplotlib.pyplot as plt
import numpy as np
import dcor
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--preprocessing_used', type=str, default='log')
    parser.add_argument('--cell_line', type=str, default='K562')
    parser.add_argument('--listfile', nargs='+', type=str,
                        default='data/K562_2000_merged_histones_init.csv.gz')
    parser.add_argument('--marks', nargs='+', type=str,
                        default=['H2A.Z', 'H3K27ac', 'H3K79me2', 'H3K27me3',
                                 'H3K9ac', 'H3K4me2', 'H3K4me3', 'H3K9me3',
                                 'H3K4me1', 'H3K36me3', 'H4K20me1'])
    parser.add_argument('--output', type=str, default=['initiation'])
    parser.add_argument('--output_dir', type=str,
                        default='model_behavior/')
    parser.add_argument('--model_dir', type=str,
                        default='development/logFCNN_K562_marks.mdl_wts.hdf5')
    parser.add_argument('--image_format', type=str, default='png')

    args = parser.parse_args()

    df = pd.read_csv('{}'.format(args.listfile), compression='gzip')
    masks = pd.read_csv('data/hg19_2000_no_N_inside.csv')
    print('Number of NANs is {}'.format(masks['signal'].sum()))
    df.loc[~masks['signal'].astype(bool)] = np.nan
    df = df.dropna()
    print(df)
    if args.preprocessing_used == 'log':
        model = tf.keras.models.load_model(args.model_dir)
        for i in args.marks:
            df[i] = df[i] + np.min(df[i][(df[i] != 0)])
            df[i] = np.log10(df[i])
        for i in args.output:
            df[i] = df[i] + np.min(df[i][(df[i] != 0)])
        X_train = df.loc[df['chrom'] != 'chr1', args.marks].to_numpy()
        print(X_train.shape)
        y_train = df.loc[df['chrom'] != 'chr1', args.output].to_numpy()
        print(y_train.shape)
        X_test = df.loc[df['chrom'] == 'chr1', args.marks].to_numpy()
        y_test = df.loc[df['chrom'] == 'chr1', args.output].to_numpy()
        # test it 2
        # Computing the error of the Training Set
        RMSE1 = tf.keras.metrics.RootMeanSquaredError()
        predictions_train = model.predict(X_train)
        predictions_train = np.power(10, predictions_train)
        predictions_test = model.predict(X_test)
        presictions_test = np.power(10, predictions_test)
        print(y_train)
        print(predictions_train)
        results = {}

        for i in range(y_train.shape[1]):
            print('training set score for parameter number {}:'.format(i+1),
                  r2_score(y_train[:, i], predictions_train[:, i]))
            results['training set score'] = r2_score(
                y_train[:, i], predictions_train[:, i])
            print('training set MAE for parameter number {}:'.format(i+1),
                  mean_absolute_error(y_train[:, i], predictions_train[:, i]))
            results['training set MAE'] = mean_absolute_error(
                y_train[:, i], predictions_train[:, i])
            print('training set MSE for parameter number {}:'.format(i+1),
                  mean_squared_error(y_train[:, i], predictions_train[:, i]))
            results['training set MSE'] = mean_squared_error(
                y_train[:, i], predictions_train[:, i])
            print('training set RMSE for parameter number {}:'.format(i+1),
                  np.sqrt(mean_squared_error(y_train[:, i],
                                             predictions_train[:, i])))
            results['training set RMSE'] = np.sqrt(mean_squared_error(
                y_train[:, i], predictions_train[:, i]))
            print('test set score for parameter number {}:'.format(i+1),
                  r2_score(y_test.reshape(-1, 1)[:, i],
                  predictions_test[:, i]))
            results['test set score'] = r2_score(y_test.reshape(-1, 1)[:, i],
                    predictions_test[:, i])
            print('test set MAE for parameter number {}:'.format(i+1),
                  mean_absolute_error(y_test.reshape(-1, 1)[:, i],
                  predictions_test[:, i]))
            results['test set MAE'] = mean_absolute_error(
                y_test.reshape(-1, 1)[:, i], predictions_test[:, i])
            print('test set MSE for parameter number {}:'.format(i+1),
                  mean_squared_error(y_test.reshape(-1, 1)[:, i],
                  predictions_test[:, i]))
            results['test set MSE'] = mean_squared_error(
                y_test.reshape(-1, 1)[:, i], predictions_test[:, i])
            print('test set RMSE for parameter number {}:'.format(i+1),
                  np.sqrt(mean_squared_error(y_test.reshape(-1, 1)[:, i],
                                             predictions_test[:, i])))
            results['test set RMSE'] = np.sqrt(mean_squared_error(y_test[:, i],
                                               predictions_test[:, i]))
            print('test set DCorr for parameter number {}:'.format(i+1),
                  dcor.distance_correlation(predictions_test[:, i],
                  y_test.reshape(-1, 1)[:, i]), '\n')
            results['test set DCorr'] = dcor.distance_correlation(
                predictions_test[:, i], y_test.reshape(-1, 1)[:, i])

            results_df = pd.DataFrame(results, index=[0])
            # results_df.to_excel('/home/amir/my_codes/mv_seq_to_seq_regression/fcnn/fcnn_results/results.xlsx')
