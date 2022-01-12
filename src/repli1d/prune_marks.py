import itertools
import pickle

import keras
import numpy as np
import pandas as pd
import scipy.signal
from scipy import stats
from scipy.signal import find_peaks

from repli1d.nn import (create_model, load_signal, train_test_split,
                        transform_norm, transform_seq)


def predict_ori(name, marks, return_signal=False):
    multi_layer_keras_model = keras.models.load_model(name)
    # multi_layer_keras_model = keras.models.load_model("../results/whole_pip_last_K562/K562_Epi_nn_ORC2///Noneweights.hdf5")

    # marks = ['H3k4me3', 'H3k4me1']
    # marks = ['H2az', 'H3k27ac', 'H3k79me2', 'H3k27me3', 'H3k9ac',
    #                 'H3k4me2', 'H3k4me3', 'H3k9me3', 'H3k4me1', 'H3k36me3', "H4k20me1"]
    temp_dict = load_signal("./data//prune//mlformat_whole_pipe_K562.csv",
                            marks=marks,
                            targets=["initiation"],
                            t_norm=transform_norm, smm=5, wig=True,
                            augment=False, show=False)
    df, yinit, notnan, mask_borders = temp_dict.values()
    # root = "/home/jarbona/projet_yeast_replication/notebooks/DNaseI/repli1d/"
    # XC = pd.read_csv(root + "coords_K562.csv", sep="\t")  # List of chromosome coordinates
    # traint =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] +[20, 21, 22, 23]
    ch1 = int(245522847 / 5000)
    df = df[:ch1]
    yinit = yinit[:ch1]
    # train, test = train_test_split(XC, traint, valt, notnan)
    X_train_us, y_train_us = df, yinit

    window = 51
    vtrain = transform_seq(X_train_us, y_train_us, 1, window)
    if not return_signal:
        return multi_layer_keras_model.predict(vtrain[0])
    else:
        return multi_layer_keras_model.predict(vtrain[0]), vtrain, multi_layer_keras_model


if __name__ == "__main__":

    which = "initiation"
    all_marks = True
    global_scores = []

    if which == "ORC":
        marks = ['H2az', 'H3k27ac', 'H3k79me2', 'H3k27me3', 'H3k9ac',
                 'H3k4me2', 'H3k4me3', 'H3k9me3', 'H3k4me1', 'H3k36me3',
                 "H4k20me1", "AT_5", "AT_20", "AT_30"]
        name_nn = "data/prune/results___whole_pip_last_K562___K562_Epi_nn_ORC2_AT___Noneweights.hdf5"
        ref, vtrain, multi_layer_keras_model = predict_ori(name_nn,
                                                           marks=marks,
                                                           return_signal=True)

        name = "./data//prune//mlformat_whole_pipe_K562.csv"
        t = pd.read_csv(name)
        ORC2 = np.array(t["ORC2"][25:-25])
    if which == "initiation":
        marks = ['H2az', 'H3k27ac', 'H3k79me2', 'H3k27me3', 'H3k9ac',
                 'H3k4me2', 'H3k4me3', 'H3k9me3', 'H3k4me1', 'H3k36me3', "H4k20me1"]
        weight = "data/prune/results___all_reproducible_normal___all_reproducible_normal___Noneweights.hdf5"
        if all_marks:
            marks += ["RNA_seq","DNaseI"]
            weight =
        ref, vtrain, multi_layer_keras_model = predict_ori(weight,
            marks=marks, return_signal=True)


    MRT = np.array(pd.read_csv("data/prune/results___whole_pip_last_K562___K562_Epi_from_All_normal_wholecell___global_profiles.csv")[
                "MRTe"])[25:-25]

    ref=ref.flatten()
    peaks_ref, _ = find_peaks(ref / np.percentile(ref,95),prominence=0.5)#,width=1 )
    early_p = peaks_ref[MRT[peaks_ref]<0.5]
    late_p = peaks_ref[MRT[peaks_ref]>0.5]
    selected = []
    left = list(range(len(marks)))
    mean = np.mean(vtrain[0].reshape(-1,len(marks)),axis=0)
    global_score =[]
    for combination in itertools.product([0,1], repeat=len(marks)):
        scores=[]

        signal = vtrain[0].copy()
        for mark_to_flat,im in enumerate(combination):
            if im == 0:
                #print("skip")
                signal[::,::,::,mark_to_flat]=mean[mark_to_flat]
        #print(signal.shape)
        #print(signal[0,0,0])
        predict = multi_layer_keras_model.predict(signal)
        predict = predict.flatten()
        if which == "initiation":
            peaks_signal, _ = find_peaks(predict / np.percentile(predict,95),prominence=0.5)#,width=1 )
            peaks_signal_early  =peaks_signal[MRT[peaks_signal]<0.5]
            peaks_signal_late  =peaks_signal[MRT[peaks_signal]>0.5]
            scoring = stats.pearsonr(ref,predict)[0]

            global_scores.append([scoring,np.mean( (predict-ref)**2),
                                  len(set(peaks_ref).intersection(set(peaks_signal)))/len(peaks_ref),
                                    len(set(early_p).intersection(set(peaks_signal_early)))/len(early_p),
                                 len(set(late_p).intersection(set(peaks_signal_late)))/len(late_p)]+list(combination))

        if which == "ORC":
            scoring = stats.pearsonr(np.array(ORC2[:len(predict)]),predict)[0]

            global_scores.append([scoring,np.mean( (predict-ref)**2)]+list(combination))
        print(global_scores[-1])
        #break

    with open(f"data/prune/results_K562_{which}.pick","wb") as f:
        pickle.dump(global_scores,f)
        #
