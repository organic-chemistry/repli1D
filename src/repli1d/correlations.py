import argparse

import dcor
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# crosscorr needs to be checked and modified


def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation. 
        Shifted data filled with NaNs 
        
        Parameters
        ----------
        lag : int, default 0
        datax, datay : pandas.Series objects of equal length
        Returns
        ----------
        crosscorr : float
        """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag), method='spearman')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--correlation', type=str, default='spearman')
    parser.add_argument('--cell_line', type=str, default='K562')
    parser.add_argument('--listfile', nargs='+', type=str,
                        default='data/K562_2000_merged_histones_init.csv.gz')
    parser.add_argument('--marks', nargs='+', type=str,
                        default=['H3K4me1', 'H3K27ac', 'H2A.Z', 'H3K9ac',
                                 'H3K4me2', 'H3K79me2', 'H4K20me1',
                                 'H3K36me3', 'H3K4me3', 'H3K27me3',
                                 'H3K9me3'])
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
        for i in args.marks + args.output:
            # df.loc[df[i] == 0, i] = np.min(df[i][(df[i] != 0)])
            df[i] = df[i] + np.min(df[i][(df[i] != 0)])
            df[i] = np.log10(df[i])
        corrMatrix = df[args.marks + args.output].corr(method='spearman')
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(10, 8))
            normalize = mpl.colors.Normalize(vmin=-0.2, vmax=1)
            ax = sns.clustermap(corrMatrix, annot=True, row_cluster=True,
                                col_cluster=True, metric='correlation',
                                cmap=sns.diverging_palette(220, 20, n=50),
                                norm=normalize)  # cbar_pos=(0, .2, .03, .4)
        plt.title("Spearman correlation coefficients for logs of values {} epigenetic markers".format(
            args.cell_line), x=10, y=1)
        plt.savefig('{}{}{}___{}.{}'.format(args.output_dir,
                                            args.correlation,
                                            args.cell_line,
                                            i, args.image_format),
                    dpi=300, bbox_inches='tight', transparent=False)
        plt.close()
    

    if args.correlation == 'spearman log with clustering':
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
        corrMatrix = df[args.marks + args.output].corr(method='spearman')
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(10, 8))
            normalize = mpl.colors.Normalize(vmin=-0.2, vmax=1)
            ax = sns.clustermap(corrMatrix, annot=True, row_cluster=True,
                                col_cluster=True, metric='correlation',
                                cmap=sns.diverging_palette(220, 20, n=50),
                                norm=normalize)  # cbar_pos=(0, .2, .03, .4)
        plt.title("Spearman correlation coefficients for logs of values {} epigenetic markers".format(
            args.cell_line), x=10, y=1)
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
        for i in args.marks + args.output:
            # df.loc[df[i] == 0, i] = np.min(df[i][(df[i] != 0)])
            df[i] = df[i] + np.min(df[i][(df[i] != 0)])
            df[i] = np.log10(df[i])
        corrMatrix = df[args.marks + args.output].corr(method='spearman')
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(8, 8))
            normalize = mpl.colors.Normalize(vmin=-0.2, vmax=1)
            ax = sns.heatmap(corrMatrix, annot=True,
                             cmap=sns.diverging_palette(220, 20, n=50),
                             norm=normalize)  # cbar_pos=(0, .2, .03, .4)
        plt.title("Spearman correlation coefficients for logs of values {} epigenetic markers".format(
            args.cell_line))
        plt.savefig('{}{}{}_{}.{}'.format(args.output_dir,
                                          args.correlation,
                                          args.cell_line,
                                          i, args.image_format),
                    dpi=300, bbox_inches='tight', transparent=False)
        plt.close()

    if args.correlation == 'pearson':
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
        corrMatrix = df[args.marks + args.output].corr(method='pearson')
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(10, 8))
            normalize = mpl.colors.Normalize(vmin=-0.2, vmax=1)
            ax = sns.heatmap(corrMatrix, annot=True,
                             cmap=sns.diverging_palette(220, 20, n=50),
                             norm=normalize)  # cbar_pos=(0, .2, .03, .4)
        plt.title("Pearson correlation coefficients log of values of {} epigenetic markers".format(
            args.cell_line))
        plt.savefig('{}{}{}_{}.{}'.format(args.output_dir,
                                          args.correlation,
                                          args.cell_line,
                                          i, args.image_format),
                    dpi=300, bbox_inches='tight', transparent=False)
        plt.close()

    if args.correlation == 'pearson log with clustering':
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
        corrMatrix = df[args.marks + args.output].corr(method='pearson')
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(10, 8))
            normalize = mpl.colors.Normalize(vmin=-0.2, vmax=1)
            ax = sns.clustermap(corrMatrix, annot=True, row_cluster=True,
                                col_cluster=True, metric='correlation',
                                cmap=sns.diverging_palette(220, 20, n=50),
                                norm=normalize)  # cbar_pos=(0, .2, .03, .4)
        plt.title("Pearson correlation coefficients for logs of values {} epigenetic markers".format(
            args.cell_line), x=10, y=1, fontsize=18, weight='bold')
        plt.savefig('{}{}{}_{}.{}'.format(args.output_dir,
                                          args.correlation,
                                          args.cell_line,
                                          i, args.image_format),
                    dpi=400, bbox_inches='tight', transparent=False, format='eps')
        plt.close()

    if args.correlation == 'distance log with clustering':
        df = pd.read_csv(args.listfile)
        masks = pd.read_csv('data/hg19_2000_no_N_inside.csv')
        print('Number of NANs is {}'.format(masks['signal'].sum()))
        df.loc[~masks['signal'].astype(bool)] = np.nan
        df = df.dropna()
        print(df)
        for i in args.marks + args.output:
            df[i] = df[i] + np.min(df[i][(df[i] != 0)])
            df[i] = np.log10(df[i])
        size = len(args.marks + args.output)
        res = np.zeros([size, size])
        for index, elem in enumerate(args.marks + args.output):
            x = df[elem].to_numpy(copy=True)
            for inter_index, inter_elem in enumerate(args.marks + args.output):
                y = df[inter_elem].to_numpy(copy=True)
                res[index, inter_index] = dcor.distance_correlation(x, y)
        print(res)
        corrMatrix = pd.DataFrame(res, index=args.marks + args.output,
                                  columns=args.marks + args.output)
        # corrMatrix = dcor.rowwise(dcor.distance_correlation,
        #                           df[args.marks + args.output].to_numpy(copy=True),
        #                           df[args.marks + args.output].to_numpy(copy=True))
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(10, 8))
            normalize = mpl.colors.Normalize(vmin=-0.2, vmax=1)
            ax = sns.clustermap(corrMatrix, annot=True, row_cluster=True,
                                col_cluster=True, metric='correlation',
                                cmap=sns.diverging_palette(220, 20, n=50),
                                norm=normalize)  # cbar_pos=(0, .2, .03, .4)
        plt.title("Distance correlation coefficients for logs of values {} epigenetic markers".format(
            args.cell_line), x=10, y=1, fontsize=18, weight='bold')
        plt.savefig('{}{}{}_{}.{}'.format(args.output_dir,
                                          args.correlation,
                                          args.cell_line,
                                          i, args.image_format),
                    dpi=300, bbox_inches='tight', transparent=False, format='eps')
        plt.close()

# TLCC needs to be checked and modified
    if args.correlation == 'time lagged cross correlation':
        df = pd.read_csv(args.listfile)
        masks = pd.read_csv('data/hg19_2000_no_N_inside.csv')
        print('Number of NANs is {}'.format(masks['signal'].sum()))
        df.loc[~masks['signal'].astype(bool)] = np.nan
        df = df.dropna()
        print(df)
        for i in args.marks + args.output:
            df[i] = df[i] + np.min(df[i][(df[i] != 0)])
            df[i] = np.log10(df[i])
        d1 = df['initiation']
        d2 = df['H3K4me1']
        seconds = 5000
        fps = 1
        rs = [crosscorr(d1,d2, lag) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
        offset = np.floor(len(rs)/2)-np.argmax(rs)
        f,ax=plt.subplots(figsize=(14,3))
        ax.plot(rs)
        ax.axvline(np.ceil(len(rs)/2),color='k',linestyle='--',label='Center')
        ax.axvline(np.argmax(rs),color='r',linestyle='--',label='Peak synchrony')
        ax.set(title=f'Offset = {offset} frames\nS1 leads <> S2 leads', xlabel='Offset',ylabel='Pearson r')
        # ax.set_xticks([0, 50, 100, 151, 201, 251, 301])
        # ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150]);
        plt.legend()
        plt.savefig('{}{}{}_{}.{}'.format(args.output_dir,
                                          args.correlation,
                                          args.cell_line,
                                          i, args.image_format),
                    dpi=300, bbox_inches='tight', transparent=False)
        plt.close()
    
    if args.correlation == 'pearson spearman distance log':
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
        pear_c = df[args.marks + args.output].corr(method='pearson')
        spear_c = df[args.marks + args.output].corr(method='spearman')
        size = len(args.marks + args.output)
        res = np.zeros([size, size])
        for index, elem in enumerate(args.marks + args.output):
            x = df[elem].to_numpy(copy=True)
            for inter_index, inter_elem in enumerate(args.marks + args.output):
                y = df[inter_elem].to_numpy(copy=True)
                res[index, inter_index] = dcor.distance_correlation(x, y)
        print(res)
        dist_c = pd.DataFrame(res, index=args.marks + args.output,
                                columns=args.marks + args.output)
        with sns.axes_style("white"):
            f, axs = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(30,10))
            normalize = mpl.colors.Normalize(vmin=-0.2, vmax=1)
            sns.heatmap(spear_c, annot=True, ax=axs[0],
                                cmap=sns.diverging_palette(220, 20, n=50),
                                norm=normalize)
            sns.heatmap(pear_c, annot=True, cbar=False, ax=axs[1],
                                cmap=sns.diverging_palette(220, 20, n=50),
                                norm=normalize)
            sns.heatmap(dist_c, annot=True, ax=axs[2], cbar=False,
                                cmap=sns.diverging_palette(220, 20, n=50),
                                norm=normalize)  # cbar_pos=(0, .2, .03, .4)
        f.tight_layout()
        plt.savefig('{}{}{}_{}.{}'.format(args.output_dir,
                                        args.correlation,
                                        args.cell_line,
                                        i, args.image_format),
                    dpi=300, bbox_inches='tight', transparent=False,
                    format='eps')
        plt.close()
