import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--correlation', type=str, default='spearman')
    parser.add_argument('--cell_line', type=str, default='K562')
    parser.add_argument('--listfile', nargs='+', type=str,
                        default='data/K562_2000_merged_histones_init.csv.gz')
    parser.add_argument('--marks', nargs='+', type=str,
                        default=['H2A.Z', 'H3K27ac', 'H3K79me2', 'H3K27me3',
                                 'H3K9ac', 'H3K4me2', 'H3K4me3', 'H3K9me3',
                                 'H3K4me1', 'H3K36me3', "H4K20me1"])
    parser.add_argument('--output', type=str, default=['initiation'])
    parser.add_argument('--output_dir', type=str, 
                        default='data_representation/')
    parser.add_argument('--image_format', type=str, default='png')

    args = parser.parse_args()

    if args.correlation == 'spearman log with clustering':
        df = pd.read_csv(args.listfile)
        masks = pd.read_csv('data/hg19_2000_no_N_inside.csv')
        print('Number of NANs is {}'.format(masks['signal'].sum()))
        df.loc[~masks['signal'].astype(bool)] = np.nan
        df = df.dropna()
        print(df)
        corrMatrix = df[args.marks + args.output].corr(method='spearman')
        for i in args.marks + args.output:
            # df.loc[df[i] == 0, i] = np.min(df[i][(df[i] != 0)])
            df[i] = df[i] + np.min(df[i][(df[i] != 0)])
            df[i] = np.log10(df[i])
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(10, 8))
            normalize = mpl.colors.Normalize(vmin=-0.2, vmax=1)
            ax = sns.clustermap(corrMatrix, annot=True, row_cluster=True,
                                col_cluster=True, metric='correlation',
                                cmap=sns.diverging_palette(220, 20, n=50),
                                norm=normalize)  # cbar_pos=(0, .2, .03, .4)
        plt.title("Spearman correlation coefficients for logs of values {} epigenetic markers".format(args.cell_line), x=10, y=1)
        plt.savefig('{}{}{}___{}.{}'.format(args.output_dir,
                                            args.correlation,
                                            args.cell_line,
                                            i, args.image_format),
                    dpi=300, bbox_inches='tight', transparent=False)
        plt.close()

    if args.correlation == 'spearman log':
        df = pd.read_csv(args.listfile)
        masks = pd.read_csv('data/hg19_2000_no_N_inside.csv')
        print('Number of NANs is {}'.format(masks['signal'].sum()))
        df.loc[~masks['signal'].astype(bool)] = np.nan
        df = df.dropna()
        print(df)
        corrMatrix = df[args.marks + args.output].corr(method='spearman')
        for i in args.marks + args.output:
            # df.loc[df[i] == 0, i] = np.min(df[i][(df[i] != 0)])
            df[i] = df[i] + np.min(df[i][(df[i] != 0)])
            df[i] = np.log10(df[i])
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(8, 8))
            normalize = mpl.colors.Normalize(vmin=-0.2, vmax=1)
            ax = sns.heatmap(corrMatrix, annot=True,
                             cmap=sns.diverging_palette(220, 20, n=50),
                             norm=normalize)  # cbar_pos=(0, .2, .03, .4)
        plt.title("Spearman correlation coefficients for logs of values {} epigenetic markers".format(args.cell_line))
        plt.savefig('{}{}{}_{}.{}'.format(args.output_dir,
                                          args.correlation,
                                          args.cell_line,
                                          i, args.image_format),
                    dpi=300, bbox_inches='tight', transparent=False)
        plt.close()

    if args.correlation == 'spearman':
        df = pd.read_csv(args.listfile)
        masks = pd.read_csv('data/hg19_2000_no_N_inside.csv')
        print('Number of NANs is {}'.format(masks['signal'].sum()))
        df.loc[~masks['signal'].astype(bool)] = np.nan
        df = df.dropna()
        print(df)
        corrMatrix = df[args.marks + args.output].corr(method='spearman')
        for i in args.marks + args.output:
            # df.loc[df[i] == 0, i] = np.min(df[i][(df[i] != 0)])
            df[i] = df[i] + np.min(df[i][(df[i] != 0)])
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(8, 8))
            normalize = mpl.colors.Normalize(vmin=-0.2, vmax=1)
            ax = sns.heatmap(corrMatrix, annot=True,
                             cmap=sns.diverging_palette(220, 20, n=50),
                             norm=normalize)  # cbar_pos=(0, .2, .03, .4)
        plt.title("Spearman correlation coefficients for {} epigenetic markers".format(args.cell_line))
        plt.savefig('{}{}{}_{}.{}'.format(args.output_dir,
                                          args.correlation,
                                          args.cell_line,
                                          i, args.image_format),
                    dpi=300, bbox_inches='tight', transparent=False)
        plt.close()
