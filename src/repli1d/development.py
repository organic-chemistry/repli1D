import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.metrics import make_scorer
from repli1d.models import mlp

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--preprocessing', type=str, default='log')
    parser.add_argument('--max_epoch', type=int, default=150)
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
                                     n_estimators=500, n_jobs=20, random_state=0)
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
        plt.savefig('{}distribution_performance.{}'.format(args.output_dir, args.image_format),
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
                                     n_estimators=500, n_jobs=20, random_state=0)
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

    if args.preprocessing == 'raw to raw RF':
        X_train = df.loc[df['chrom'] != 'chr1', args.marks].to_numpy()
        print(X_train.shape)
        y_train = df.loc[df['chrom'] != 'chr1', args.output].to_numpy()
        print(y_train.shape)
        X_test = df.loc[df['chrom'] == 'chr1', args.marks].to_numpy()
        y_test = df.loc[df['chrom'] == 'chr1', args.output].to_numpy()
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        regr = RandomForestRegressor(max_depth=20, min_samples_leaf=20,
                                     n_estimators=200, n_jobs=-1, random_state=0)
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
