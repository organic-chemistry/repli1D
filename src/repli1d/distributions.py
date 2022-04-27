
import argparse
from hashlib import shake_128
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distribution', type=str, default='histogram')
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
    parser.add_argument('--image_format', type=str, default='eps')

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
            # df[i] = df[i] + np.min(df[i][(df[i] != 0)])
            # df[i] = np.log10(df[i])
            sns.displot(df, x=i, bins=200)
            plt.title('{} of log of values for {}'.format(args.distribution,
                                                          args.cell_line), fontsize=18)
            plt.savefig('{}{}{}_{}.{}'.format(args.output_dir,
                                              args.distribution,
                                              args.cell_line,
                                              i, args.image_format),
                        dpi=300, bbox_inches='tight', transparent=False,
                        format='eps')
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
            # df[a] = df[a] + np.min(df[a][(df[a] != 0)])
            # df[a] = np.log10(df[a])
            for i in args.marks:
                # df[i] = df[i] + np.min(df[i][(df[i] != 0)])
                # df[i] = np.log10(df[i])
                plt.figure(figsize=(10, 10))
                sns.jointplot(x=df[i], y=df[a], kind='hex',
                              marginal_ticks=True, cmap=plt.cm.nipy_spectral,
                              marginal_kws=dict(binwidth=0.008, fill=True))
                plt.colorbar()
                plt.xlabel('{}'.format(i))
                plt.ylabel('{}'.format(a))
                plt.title('{} of log of values for {}'.format(
                          args.distribution, args.cell_line))
                plt.savefig('{}{}{}_{}.{}'.format(args.output_dir,
                                                  args.distribution,
                                                  args.cell_line,
                                                  i, args.image_format),
                            dpi=300, bbox_inches='tight',
                            format='eps')
                plt.close()

    if args.distribution == 'raw':
        df = pd.read_csv('data/K562_2000.csv')
        df1 = pd.read_csv(args.listfile)
        masks = pd.read_csv('data/hg19_2000_no_N_inside.csv')
        print('Number of NANs is {}'.format(masks['signal'].sum()))
        df.loc[~masks['signal'].astype(bool)] = np.nan
        df = df.dropna()
        print(df)
        fig, (ax, ax1) = plt.subplots(2, 1,figsize=(16.8, 7*2))
        s = 31500
        e = 34000
        s1 = 32500
        e1 = 33000
        ax.plot(df['MRT'][s:e], c='#007FD4')
        ax.set_xlabel('Genomic position', fontsize=18)
        ax.set_ylabel("MRT", c='#007FD4', fontsize=18)
        ax.invert_yaxis()
        axr = ax.twinx()
        axr.plot(df['OKSeq'][s:e], c='#D45500')
        ax.set_title('K562, Resolution 2kb', fontsize=18, weight='bold')
        axr.set_ylabel("RFD", c='#D45500', fontsize=18)

        ax1.plot(df1['initiation'][s1:e1], c='#D40015')
        ax1.set_xlabel('Genomic position', fontsize=18) # compact it
        ax1.set_ylabel("PODLS", c='#D40015', fontsize=18)
        ax1.set_yscale('log')
        axr1 = ax1.twinx()
        axr1.plot(df1['H3K4me1'][s1:e1], c='#00D4BF')
        axr1.set_ylabel('H3K4me1', c='#00D4BF', fontsize=18)
        axr1.set_yscale('log')
        plt.savefig('{}{}{}.{}'.format(args.output_dir,
                                            args.distribution,
                                            args.cell_line,
                                            args.image_format),
                    dpi=300, bbox_inches='tight', transparent=False,
                    format='eps')
        plt.close()


    if args.distribution == 'raw marker podls':
        df = pd.read_csv('data/K562_2000.csv')
        df1 = pd.read_csv(args.listfile)
        masks = pd.read_csv('data/hg19_2000_no_N_inside.csv')
        print('Number of NANs is {}'.format(masks['signal'].sum()))
        df.loc[~masks['signal'].astype(bool)] = np.nan
        df = df.dropna()
        print(df)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1,figsize=(16.8, 7*4), sharex=True)
        s = 21000
        e = 31000
        # s1 = 20850
        # e1 = 30850
        s1 = 20850
        e1 = 25850
        # s1 = 0
        # e1 = 110000
        ax1.plot(df1['H3K4me1'][s1:e1], c='#00D4BF')
        ax1.set_ylabel("H3K4me1", c='#00D4BF', fontsize=18)
        # ax1.set_yscale('log')
        ax2.plot(df1['H3K27ac'][s1:e1], c='blue')
        ax2.set_ylabel("H3K27ac", c='blue', fontsize=18)
        # ax2.set_yscale('log')
        ax3.plot(df1['H3K9me3'][s1:e1], c='red')
        ax3.set_ylabel("H3K9me3", c='red', fontsize=18)
        # ax3.set_yscale('log')
        ax3.set_xlabel('Genomic position(2kbp)', fontsize=18) # compact it
        # ax1.set_yscale('log')
        # axr1 = ax1.twinx()
        # axr1.plot(df1['H3K27ac'][s1:e1], c='#007FD4')
        # axr1.set_ylabel('H3K27ac', c='#007FD4', fontsize=18)
        # axr1.set_yscale('log')
        plt.savefig('{}{}{}.{}'.format(args.output_dir,
                                            args.distribution,
                                            args.cell_line,
                                            args.image_format),
                    dpi=300, bbox_inches='tight', transparent=False,
                    format='eps')
        plt.close()
        plt.plot(df1['initiation'][s1:e1], c='#D45500')
        plt.ylabel("PODLS", c='#D45500', fontsize=18)
        plt.xlabel('Genomic position(2kbp)', fontsize=18) # compact it
        plt.savefig('{}{}{}__.{}'.format(args.output_dir,
                                            args.distribution,
                                            args.cell_line,
                                            args.image_format),
                    dpi=300, bbox_inches='tight', transparent=False,
                    format='eps')

