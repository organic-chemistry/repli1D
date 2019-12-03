
import pickle
from repli1d.retrieve_marks import norm2
from repli1d.expeData import whole_genome
import pandas as pd
import numpy as np
from repli1d.analyse_RFD import propagate_n_false


def load_signals_and_transform_to_ml_format(peak,file, strain, norm=False,roadmap=False):

    resolution = 5
    redo = False
    """
    if norm:
        smark = whole_genome(strain=strain, experiment="CNV",
                             resolution=resolution, raw=False, oData=False,
                             bp=True, bpc=False, filename=None, redo=redo, root="/home/jarbona/repli1D/")
        smark = np.concatenate(smark, axis=0)
        smark[smark == 0] = 4
        smark[np.isnan(smark)] = 4
        CNV = smark
    """
    v = []
    marks = ["H3k4me1",	"H3k4me3",	"H3k9me3",	"H3k27me3",	"H3k36me3"]
    Data = pd.read_csv(file,sep="\t")

    for c in marks:
        print(c)
        Data[c][np.isnan(Data[c])]=0


    data = pd.read_csv(peak, "\t")
    data["signalValue"][data["signalValue"] < 0.001] = 0
    print(len(Data),len(data))

    Data["initiation"] = norm2(data["signalValue"])[0]
    X = []
    Pos = []

    return pd.DataFrame(Data), X, Pos


if __name__ == "__main__":
    root = "/home/jarbona/repli1D/data/roadmap_"


    df, X, Pos = load_signals_and_transform_to_ml_format(file="/mnt/data/data/roadmap/K562/input_road.csv",
        peak="/home/jarbona/repli1D/results/K562_RFD_to_init/nn_global_profiles.csv", strain="K562", norm=True)
    df.to_csv(root+"K562_nn.csv", index=False)

    df, X, Pos = load_signals_and_transform_to_ml_format(file="/mnt/data/data/roadmap/IMR90/input_road.csv",
        peak="/home/jarbona/repli1D/results/K562_RFD_to_init/nn_global_profiles.csv", strain="IMR90", norm=True)
    df.to_csv(root+"IMR90_nn.csv", index=False)
"""
    df, X, Pos = load_signals_and_transform_to_ml_format(
        peak="/home/jarbona/repli1D/results/GM_RFD_to_init/nn_global_profiles.csv", strain="Gm12878", norm=False)
    df.to_csv(root+"GM_nn.csv", index=False)

    df, X, Pos = load_signals_and_transform_to_ml_format(
        peak="/home/jarbona/repli1D/results/Hela_RFD_to_init/nn_global_profiles.csv", strain="Helas3", norm=True)
    df.to_csv(root+"Hela_nn.csv", index=False)
"""