
import pickle
from repli1d.retrieve_marks import norm2
from repli1d.expeData import whole_genome
import pandas as pd
import numpy as np
from repli1d.analyse_RFD import propagate_n_false


def load_signals_and_transform_to_ml_format(peak, strain, norm=False,roadmap=False,resolution=5):

    if "highest_correlation.csv" in peak:
        with open(peak,"r") as fich:
            peak = fich.readlines()[0]
            peak.strip()

    print(peak)
    redo = False

    if norm:
        try:
            smark = whole_genome(strain=strain, experiment="CNV",
                                 resolution=resolution, raw=False, oData=False,
                                 bp=True, bpc=False, filename=None, redo=redo, root="/home/jarbona/repli1D/")
            smark = np.concatenate(smark, axis=0)
            smark[smark == 0] = 4
            smark[np.isnan(smark)] = 4
            CNV = smark
        except:
            CNV = 4

    v = []
    Data = {}
    marks = ['H2az', 'H3k27ac', 'H3k79me2', 'H3k27me3', 'H3k9ac', 'H3k4me2',
             'H3k4me3', 'H3k9me3', 'H3k4me1', 'H3k36me3'] + ["H4k20me1"]
    marks = [m + "wig" for m in marks]
    marks += ["DNaseI"]
    marks += ["OKSeq"]
    marks += ["Meth"]
    marks += ["Meth450"]
    marks += ["AT_5","AT_20","AT_30"]
    marks += ["RNA_seq"]
    #marks += ["ORC2"]

    for mark in marks:
        print(mark)
        straint = strain
        if strain == "Gm12878" and mark == "OKSeq":
            straint = "GM06990"

        #print(strain, straint)
        smark = whole_genome(strain=straint, experiment=mark,
                             resolution=resolution, raw=False, oData=False,
                             bp=True, bpc=False, filename=None, redo=redo, root="/home/jarbona/repli1D/")
        smark = np.concatenate(smark, axis=0)


        # print(Signals[mark].dtype)
        if mark == "OKSeq":
            Data["notnan"] = propagate_n_false(~np.isnan(smark), 20)  # 100 kb

        smark[np.isnan(smark)] = 0

        if norm:
            smark /= CNV
        else:
            smark /= 4

        #if mark in ["MRT","RFD"]:
            #mark += "e"
        Data[mark] = smark


    try:
        with open(peak, "rb") as f:
            data = pickle.load(f)
        X = [["chr%i" % i] * len(d) for i, d in enumerate(data, 1)]
        Pos = [range(0, len(d)*5, 5) for i, d in enumerate(data, 1)]
        X = np.concatenate(X).tolist()
        Pos = np.concatenate(Pos).tolist()

        data = np.concatenate(data, axis=0)
        assert(len(X) == len(data))
        data, mean, std = norm2(data)
        data[np.isnan(data)] = 0
        data = data[:None]
        Data["initiation"] = data
    except:
        data = pd.read_csv(peak, "\t")
        data["signalValue"][data["signalValue"] < 0.001] = 0
        Data["initiation"] = norm2(data["signalValue"])[0]


        print(len(Data["OKSeq"]), len(data))

        X = []
        Pos = []

    return pd.DataFrame(Data), X, Pos


if __name__ == "__main__":
    root = "/home/jarbona/repli1D/data/mlformat_whole_sig_standard"

    """
    df, X, Pos = load_signals_and_transform_to_ml_format(
        peak="/home/jarbona/repli1D/results/best/comb_8_14_nooverfit_Hela/Heladec2.peak", strain="Helas3")

    df.to_csv(root+"Hela_dec2.csv", index=False)

    df, X, Pos = load_signals_and_transform_to_ml_format(
        peak="/home/jarbona/repli1D/results/best/comb_sc_GM/GMdec2.peak", strain="Gm12878")
    df.to_csv(root+"GM_dec2.csv", index=False)

    df, X, Pos = load_signals_and_transform_to_ml_format(
        peak="/home/jarbona/repli1D/K562dec2.peak", strain="K562", norm=True)
    df.to_csv(root+"K562_dec2.csv", index=False)

    root1 = "/home/jarbona/projet_yeast_replication/notebooks/DNaseI/repli1d/"
    XC = pd.read_csv(root1 + "coords_K562.csv", sep="\t")
    XC["signalValue"] = df["initiation"]/np.max(df["initiation"])
    print("saving", root+"K562_dec2_with_coord.csv")
    XC.to_csv(root+"K562_dec2_with_coord.csv", index=False, sep="\t")
"""
    """
    df, X, Pos = load_signals_and_transform_to_ml_format(
        peak="/home/jarbona/repli1D/results/K562_RFD_to_init/nn_global_profiles.csv", strain="K562", norm=True)
    df.to_csv(root+"K562_nn.csv", index=False)

    df, X, Pos = load_signals_and_transform_to_ml_format(
        peak="/home/jarbona/repli1D/results/GM_RFD_to_init/nn_global_profiles.csv", strain="Gm12878", norm=False)
    df.to_csv(root+"GM_nn.csv", index=False)

    df, X, Pos = load_signals_and_transform_to_ml_format(
        peak="/home/jarbona/repli1D/results/Hela_RFD_to_init/nn_global_profiles.csv", strain="Helas3", norm=True)
    df.to_csv(root+"Hela_nn.csv", index=False)
    """

    import argparse
    import os

    parser = argparse.ArgumentParser()

    parser.add_argument('--peak', type=str, default=None)
    parser.add_argument('--outfile', type=str, default=None)
    parser.add_argument('--cell', type=str, default=None)
    parser.add_argument('--resolution', type=int, default=5)

    args = parser.parse_args()

    if args.cell == "Hela":
        cell = "Helas3"
    if args.cell == "GM":
        cell = "Gm12878"
    if args.cell == "K562":
        cell = "K562"




    df, X, Pos = load_signals_and_transform_to_ml_format(
        peak=args.peak, strain=cell, norm=True,resolution=args.resolution)
    df.to_csv(args.outfile, index=False)
