import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dropout, Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, History
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from repli1d.analyse_RFD import nan_polate, smooth
from keras.models import load_model


def transform_norm(signal):
    s = np.array(signal).copy()
    s -= np.percentile(s, 10)
    s /= np.percentile(s, 50)
    s /= 5
    s[s > 50] = 50
    return np.array(s,dtype=np.float32) #mod


def transform_DNase(signal):
    s = np.array(signal).copy()
    s /= 500
    s[s > 1] = 1
    return s


def transform_norm_meth(signal):
    s = np.array(signal).copy()
    print(np.percentile(s, [10, 95]))
    # s = np.percentile(s,10)
    s /= np.percentile(s, 95)
    s /= 20
    return s

# print(transform_norm)


def load_signal(name,
                marks=['H2az', 'H3k27ac', 'H3k79me2', 'H3k27me3', 'H3k9ac', 'H3k4me2',
                       'H3k4me3', 'H3k9me3', 'H3k4me1', 'H3k36me3', "H4k20me1"],
                targets=["initiation"], t_norm=None, smm=None,wig=True):

    df = pd.read_csv(name)
    #wig = True

    if "signal" in df.columns:
        df["initiation"] = df["signal"]

    if wig:
        lm = ["DNaseI", "initiation", "Meth", "Meth450", "RFDs", "MRTs", "RFDe", "MRTe"]
        marks0 = [m+"wig" for m in marks if m not in lm]
        for sm in lm:
            if sm in marks:
                marks0 += [sm]
        marks = marks0

    if "notnan" in df.columns:
        print("Found notnan")
        notnan = df["notnan"]
    else:
        notnan = []

    df = df[targets+marks]

    yinit = [df.pop(target) for target in targets]
    # print(yinit.shape,"Yinit shape")

    if t_norm is not None:
        transform_norm = t_norm

    for col in df.columns:
        # print(col)
        if col not in ["DNaseI", "initiation", "Meth", "Meth450", "RFDs", "MRTs", "RFDe", "MRTe"]:
            df[col] = transform_norm(df[col])
        elif col == "DNaseI":
            df[col] = transform_DNase(df[col])
        elif col == "initiation":
            df[col] = df[col] / np.max(df[col])
        elif "Meth" in col:
            df[col] = transform_norm_meth(df[col])
        elif "RFD" in col:
            if "RFDe" in col:
                # print("Nanpo")
                df[col] = nan_polate(df[col])
            if smm is not None:
                df[col] = smooth(df[col], smm)
            df[col] = (df[col]+1)/2
        elif "MRT" in col:
            if "MRTe" in col:
                df[col] = nan_polate(df[col])
            pass

    print(np.max(yinit[0]), "max")
    print(df.describe())

    yinit0 = []
    for y, t in zip(yinit, targets):
        if t == "initiation":
            yinit0.append(y/np.max(y))
        if t == "DNaseI":
            yinit0.append(transform_DNase(y))
        if t == "OKSeq":
            yinit0.append((y+1)/2)

    yinit = np.array(yinit0).T
    yinit[np.isnan(yinit)] = 0
    # print(yinit.shape)
    return df, yinit, notnan


def window_stack(a, stepsize=1, width=3):
    # print([[i,1+i-width or None,stepsize] for i in range(0,width)])
    return np.hstack(a[i:1+i-width or None:stepsize] for i in range(0, width))


def transform_seq(Xt, yt, stepsize=1, width=3, impair=True):
    # X = (seq,dim)
    # y = (seq)
    Xt = np.array(Xt)
    yt = np.array(yt)
    print(Xt.shape, yt.shape)

    assert(len(Xt.shape) == 2)
    assert(len(yt.shape) == 2)
    if impair:
        assert(width % 2 == 1)
    X = window_stack(Xt, stepsize, width).reshape(-1, width, Xt.shape[-1])[::, np.newaxis, ::, ::]
    # [::,np.newaxis] #Take the value at the middle of the segment
    Y = window_stack(yt[::, np.newaxis], stepsize, width)[::, width//2]

    print(X.shape, Y.shape)
    # exit()

    return X, Y


def create_model(X_train, targets, nfilters, kernel_length):
    print(X_train.shape,targets,nfilters,kernel_length)

    K.set_image_data_format('channels_last')

    multi_layer_keras_model = Sequential()
    multi_layer_keras_model.add(Conv2D(filters=nfilters, kernel_size=(
        1, kernel_length), input_shape=X_train.shape[1:]))
    multi_layer_keras_model.add(Activation('relu'))
    multi_layer_keras_model.add(Dropout(0.2))

    multi_layer_keras_model.add(Conv2D(filters=nfilters, kernel_size=(
        1, kernel_length), input_shape=X_train.shape[1:]))
    multi_layer_keras_model.add(Activation('relu'))
    multi_layer_keras_model.add(Dropout(0.2))

    multi_layer_keras_model.add(Conv2D(filters=nfilters, kernel_size=(
        1, kernel_length), input_shape=X_train.shape[1:]))
    multi_layer_keras_model.add(Activation('relu'))
    multi_layer_keras_model.add(MaxPooling2D(pool_size=(1, 2)))
    multi_layer_keras_model.add(Dropout(0.2))

    multi_layer_keras_model.add(Flatten())
    multi_layer_keras_model.add(Dense(len(targets)))
    multi_layer_keras_model.add(Activation("sigmoid"))
    # multi_layer_keras_model.compile(optimizer='adadelta',  # 'adam'
    #                                loss='mean_squared_logarithmic_error')
    multi_layer_keras_model.compile(optimizer='adadelta',  # 'adam'
                                    loss='binary_crossentropy')
    multi_layer_keras_model.summary()
    return multi_layer_keras_model


def train_test_split(chrom, ch_train, ch_test, notnan):
    print(list(ch_train), list(ch_test))
    chltrain = ["chr%i" % i for i in ch_train]
    chltest = ["chr%i" % i for i in ch_test]
    if len(notnan) != 0:
        train = [chi in chltrain and notna for chi, notna in zip(chrom.chrom, notnan)]
        test = [chi in chltest and notna for chi, notna in zip(chrom.chrom, notnan)]
    else:
        print("Working on all")
        train = [chi in chltrain for chi in chrom.chrom]
        test = [chi in chltest for chi in chrom.chrom]
    print(np.sum(train), np.sum(test), np.sum(test)/len(test))
    return train, test


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def repad1d(res, window):
    return np.concatenate([np.zeros(window//2), res, np.zeros(window//2)])


if __name__ == "__main__":

    import argparse
    import os

    parser = argparse.ArgumentParser()

    parser.add_argument('--cell', type=str, default=None)
    parser.add_argument('--rootnn', type=str, default=None)
    parser.add_argument('--nfilters', type=int, default=15)
    parser.add_argument('--sm', type=int, default=None)  # Smoothing exp data

    parser.add_argument('--window', type=int, default=51)
    parser.add_argument('--kernel_length', type=int, default=10)
    parser.add_argument('--weight', type=str, default=None)
    parser.add_argument('--marks', nargs='+', type=str, default=[])
    parser.add_argument('--targets', nargs='+', type=str, default=["initiation"])
    parser.add_argument('--listfile', nargs='+', type=str, default=[])
    parser.add_argument('--enrichment', nargs='+', type=float, default=[0.1, 1.0, 5.0])
    parser.add_argument('--roadmap',action="store_true")
    parser.add_argument('--noenrichment',action="store_true")

    parser.add_argument('--restart', type=str, default=None)


    args = parser.parse_args()

    cell = args.cell
    rootnn = args.rootnn
    window = args.window
    marks = args.marks
    if marks == []:
        marks = ['H2az', 'H3k27ac', 'H3k79me2', 'H3k27me3', 'H3k9ac',
                 'H3k4me2', 'H3k4me3', 'H3k9me3', 'H3k4me1', 'H3k36me3', "H4k20me1"]

    lcell = [cell]

    if cell == "all":
        lcell = ["K562", "GM", "Hela"]

    os.makedirs(args.rootnn, exist_ok=True)

    root = "/home/jarbona/projet_yeast_replication/notebooks/DNaseI/repli1d/"
    XC = pd.read_csv(root + "coords_K562.csv", sep="\t")  # List of chromosome coordinates

    if args.listfile == []:
        listfile = []
        for cellt in lcell:
            name = "/home/jarbona/repli1D/data/mlformat_whole_sig_%s_dec2.csv" % cellt
            name = "/home/jarbona/repli1D/data/mlformat_whole_sig_standard%s_dec2.csv" % cellt
            name = "/home/jarbona/repli1D/data/mlformat_whole_sig_standard%s_nn.csv" % cellt
            if args.roadmap:
                namep = "/home/jarbona/repli1D/data/roadmap_%s_nn.csv" % cellt

            listfile.append(name)
    else:
        listfile = args.listfile

    if args.weight is None:
        X_train = []
        for name in listfile:
            print(name)
            df, yinit, notnan = load_signal(
                name, marks, targets=args.targets, t_norm=transform_norm, smm=args.sm)

            traint = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19]
            valt = [4, 18, 21, 22]
            testt = [5, 20]
            traint = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            valt = [1, 20, 21, 22, 23]
            testt = []

            for v in testt:
                assert(v not in traint)
            for v in valt:
                assert(v not in traint)
            train, test = train_test_split(XC, traint, valt, notnan)
            X_train_us, X_test_us, y_train_us, y_test_us = df[train], df[test], yinit[train], yinit[test]

            vtrain = transform_seq(X_train_us, y_train_us, 1, window)
            vtest = transform_seq(X_test_us, y_test_us, 1, window)
            if X_train == []:
                X_train, y_train = vtrain
                X_test, y_test = vtest
            else:
                X_train = np.concatenate([X_train, vtrain[0]])
                y_train = np.concatenate([y_train, vtrain[1]])
                X_test = np.concatenate([X_test, vtest[0]])
                y_test = np.concatenate([y_test, vtest[1]])

        X_train, y_train = unison_shuffled_copies(X_train, y_train)

    if args.weight is not None:
        multi_layer_keras_model = load_model(args.weight)
        multi_layer_keras_model.summary()

    else:
        multi_layer_keras_model = create_model(
            X_train, targets=args.targets, nfilters=args.nfilters, kernel_length=args.kernel_length)

        if args.restart is not None:
            multi_layer_keras_model = load_model(args.restart)

        """
        if (len(args.targets) == 1) and (args.targets[0] == "OKSeq"):

            selpercents = [1.0]
        else:
            selpercents = [0.1, 1.0, 5.0, "all"]
        """

        totenr = args.enrichment + ["all"]
        if args.noenrichment:
            totenr = ["all"]

        print(totenr)
        for selp in totenr :

            print(sum(y_train == 0), sum(y_train != 0))
            if type(selp) == float:
                sel = y_train[::, 0] != 0
                sel[np.random.randint(0, len(sel-1), int(selp*sum(sel)))] = True
            else:

                sel = np.ones_like(y_train[::, 0],dtype=np.bool)
            print(np.sum(sel), sel.shape)
            print(X_train.shape, X_train[sel].shape)
            cp = [EarlyStopping(patience=3)]
            if selp == "all" and False:
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                              patience=3, min_lr=0.1)
                cp = [reduce_lr]

            history_multi_filter = multi_layer_keras_model.fit(x=X_train[sel],
                                                               y=y_train[sel],
                                                               batch_size=128,
                                                               epochs=150,
                                                               verbose=1,
                                                               callbacks=cp+[History(),
                                                                             ModelCheckpoint(save_best_only=True,
                                                                                             filepath=rootnn+"/%sweights.{epoch:02d}-{val_loss:.4f}.hdf5" % cell, verbose=1)],
                                                               validation_data=(X_test,
                                                                                y_test))

            multi_layer_keras_model.save(rootnn+"/%sweights.hdf5" % cell)
            print("Saving on", rootnn+"/%sweights.hdf5" % cell)
    ###################################
    # predict

    if args.listfile == [] or args.roadmap:
        lcell = ["K562", "Hela", "GM"]
        if args.cell is not None and args.weight is not None:
            lcell = [args.cell]
        for cellp in lcell:
            namep = "/home/jarbona/repli1D/data/mlformat_whole_sig_%s_dec2.csv" % cellp
            namep = "/home/jarbona/repli1D/data/mlformat_whole_sig_standard%s_nn.csv" % cellp
            wig=True
            if args.roadmap:
                namep = "/home/jarbona/repli1D/data/roadmap_%s_nn.csv" % cellp
                wig=False
            print("Reading", namep)
            df, yinit, notnan = load_signal(
                namep, marks, targets=args.targets, t_norm=transform_norm,wig=wig)
            X, y = transform_seq(df, yinit, 1, window)
            print(X.shape)
            res = multi_layer_keras_model.predict(X)
            del df, X, y
            print(res.shape, "resshape", yinit.shape)

            for itarget, target in enumerate(args.targets):
                XC["signalValue"] = repad1d(res[::, itarget], window)
                if target == "OKSeq":
                    XC["signalValue"] = XC["signalValue"] * 2-1
            # XC.to_csv("nn_hela_fk.csv",index=False,sep="\t")
                if target == "initiation":
                    ns = rootnn+"/nn_%s_from_%s.csv" % (cellp, cell)
                    s = 0
                    for y1, y2 in zip(yinit, XC["signalValue"]):
                        s += (y1-y2)**2
                    print("Average delta", s/len(yinit))
                else:
                    ns = rootnn+"/nn_%s_%s_from_%s.csv" % (cellp, target, cell)

                print("Saving to", ns)
                XC.to_csv(ns, index=False, sep="\t")
    else:
        for namep in args.listfile:
            marks = ["RFDe", "MRTe"]
            df, yinit, notnan = load_signal(
                namep, marks, targets=args.targets, t_norm=transform_norm, smm=args.sm)
            X, y = transform_seq(df, yinit, 1, window)
            print(X.shape)
            res = multi_layer_keras_model.predict(X)
            del df, X, y
            print(res.shape, "resshape", yinit.shape)

            for itarget, target in enumerate(args.targets):
                XC["signalValue"] = repad1d(res[::, itarget], window)
                if target == "OKSeq":
                    XC["signalValue"] = XC["signalValue"] * 2-1
            # XC.to_csv("nn_hela_fk.csv",index=False,sep="\t")
                if target == "initiation":
                    namew = namep.split("/")[-1][:-4]
                    ns = rootnn+"/nn_%s.csv" % (namew)
                    s = 0
                    for y1, y2 in zip(yinit, XC["signalValue"]):
                        s += (y1-y2)**2
                    print("Average delta", s/len(yinit))
                else:
                    print("Not implemented")
                    #ns = rootnn+"/nn_%s_%s_from_%s.csv" % (cellp, target, cell)

                print("Saving to", ns)
                XC.to_csv(ns, index=False, sep="\t")
