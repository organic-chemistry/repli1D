
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--distribution', type=str, default='histogram')
    parser.add_argument('--cell_line', type=str, default='K562')
    parser.add_argument('--listfile', nargs='+', type=str,
                        default='data/K562_2000_merged_histones_init.csv')
    parser.add_argument('--marks', nargs='+', type=str,
                        default=['H2A.Z', 'H3K27ac', 'H3K79me2', 'H3K27me3',
                                 'H3K9ac', 'H3K4me2', 'H3K4me3', 'H3K9me3',
                                 'H3K4me1', 'H3K36me3', "H4K20me1"])
    parser.add_argument('--output', type=str, default=['initiation'])
    parser.add_argument('--output_dir', type=str, 
                        default='data_representation/')
    parser.add_argument('--image_format', type=str, default='png')

    args = parser.parse_args()

    if args.distribution == 'histogram':
        df = pd.read_csv(args.listfile)
        masks = pd.read_csv('data/hg19_2000_no_N_inside.csv')
        print('Number of NANs is {}'.format(masks['signal'].sum()))
        df.loc[~masks['signal'].astype(bool)] = np.nan
        df = df.dropna()
        print(df)
        for i in args.marks + args.output:
            # df.loc[df[i] == 0, i] = np.min(df[i][(df[i] != 0)])
            df[i] = df[i] + np.min(df[i][(df[i] != 0)])
            df[i] = np.log10(df[i])
            sns.displot(df, x=i, bins=200)
            plt.title('{} of log of values for {}'.format(args.distribution,
                                                          args.cell_line))
            plt.savefig('{}{}{}_{}.{}'.format(args.output_dir,
                                              args.distribution,
                                              args.cell_line,
                                              i, args.image_format),
                        dpi=300, bbox_inches='tight', transparent=False)
            plt.close()

    if args.distribution == '2D histogram':
        df = pd.read_csv(args.listfile)
        masks = pd.read_csv('data/hg19_2000_no_N_inside.csv')
        print('Number of NANs is {}'.format(masks['signal'].sum()))
        df.loc[~masks['signal'].astype(bool)] = np.nan
        df = df.dropna()
        print(df)
        for a in args.output:
            # df.loc[df[a] == 0,
            #    a] = np.min(df[a][df[a] != 0])
            df[a] = df[a] + np.min(df[a][(df[a] != 0)])
            df[a] = np.log10(df[a])

            for i in args.marks:
                # df.loc[df[i] == 0, i] = np.min(df[i][(df[i] != 0)])
                df[i] = df[i] + np.min(df[i][(df[i] != 0)])
                df[i] = np.log10(df[i])
                plt.figure(figsize=(10, 10))
                plt.hist2d(df[i], df[a],
                           bins=[np.histogram_bin_edges(df[i], bins='auto'),
                                 np.histogram_bin_edges(df[a], bins='auto')],
                           cmap=plt.cm.nipy_spectral)
                plt.colorbar()
                plt.xlabel('{}'.format(i))
                plt.ylabel('{}'.format(a))
                plt.title('{} of log of values for {}'.format(
                          args.distribution, args.cell_line))
                plt.savefig('{}{}{}_{}.{}'.format(args.output_dir,
                                                  args.distribution,
                                                  args.cell_line,
                                                  i, args.image_format),
                            dpi=300, bbox_inches='tight', transparent=False)
                plt.close()

    if args.distribution == '2D histogram and marginals':
        df = pd.read_csv(args.listfile)
        masks = pd.read_csv('data/hg19_2000_no_N_inside.csv')
        print('Number of NANs is {}'.format(masks['signal'].sum()))
        df.loc[~masks['signal'].astype(bool)] = np.nan
        df = df.dropna()
        print(df)
        for a in args.output:
            df[a] = df[a] + np.min(df[a][(df[a] != 0)])
            df[a] = np.log10(df[a])
            for i in args.marks:
                df[i] = df[i] + np.min(df[i][(df[i] != 0)])
                df[i] = np.log10(df[i])
                plt.figure(figsize=(10, 10))
                sns.jointplot(x=df[i], y=df[a], kind='hex',
                              marginal_ticks=True, cmap=plt.cm.nipy_spectral,
                              marginal_kws=dict(binwidth=0.008, fill=True))
                plt.colorbar()
                plt.xlabel('{}'.format(i))
                plt.ylabel('{}'.format(a))
                # plt.title('{} of log of values for {}'.format(
                #           args.distribution, args.cell_line))
                plt.savefig('{}{}{}_{}.{}'.format(args.output_dir,
                                                  args.distribution,
                                                  args.cell_line,
                                                  i, args.image_format),
                            dpi=300, bbox_inches='tight', transparent=False)
                plt.close()
