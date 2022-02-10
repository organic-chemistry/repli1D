import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle

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

    if args.preprocessing == 'log':
        for i in args.marks + args.output:
            df[i] = df[i] + np.min(df[i][(df[i] != 0)])
            df[i] = np.log10(df[i])
        # X_train = df.loc[(df['chrom'] != 'chr1') & (df['chrom'] != 'chr2'),
        #                  args.marks].to_numpy()
        # y_train = df.loc[(df['chrom'] != 'chr1') & (df['chrom'] != 'chr2'),
        #                  args.output].to_numpy()
        X_train = df.loc[df['chrom'] != 'chr1', args.marks].to_numpy()
        y_train = df.loc[df['chrom'] != 'chr1', args.output].to_numpy()
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
                   filepath=checkpoint_filepath,\
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
