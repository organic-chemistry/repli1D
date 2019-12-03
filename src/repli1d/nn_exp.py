import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adadelta, SGD, RMSprop;
import keras.losses;
from keras.constraints import maxnorm;
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping, History
from keras import backend as K
from keras.callbacks import ModelCheckpoint

def transform_norm(signal):
    s = np.array(signal).copy()
    s -= np.percentile(s,10)
    s /= np.percentile(s,50)
    s /= 5
    s[s>50]=50
    return s

def transform_DNase(signal):
    s = np.array(signal).copy()
    s /= 500
    s[s>1]=1
    return s

def load_signal(name,marks = ['H2az', 'H3k27ac', 'H3k79me2', 'H3k27me3', 'H3k9ac', 'H3k4me2', 'H3k4me3', 'H3k9me3', 'H3k4me1', 'H3k36me3',"H4k20me1"],targets=["initiation"]):

    df = pd.read_csv(name)
    wig = True

    if wig:

        marks0 = [m+"wig" for m in marks if m not in ["DNaseI","initiation"]]
        if "DNaseI" in marks:
            marks0 += ["DNaseI"]
        if "initiation" in marks:
            marks0 += ["initiation"]
        marks = marks0

    notnan = df["notnan"]

    df = df[targets+marks]

    yinit = [df.pop(target) for target in targets]
    #print(yinit.shape,"Yinit shape")

    for col in df.columns:
        #print(col)
        if col not in ["DNaseI","initiation"]:
            df[col] = transform_norm(df[col])
        elif col == "DNaseI":
            df[col] = transform_DNase(df[col])
        elif col == "initiation":
            df[col] = df[col] / np.max(df[col])


    print(np.max(yinit[0]),"max")

    yinit0 = []
    for y,t in zip(yinit,targets):
        if t == "initiation":
            yinit0.append(y/np.max(y))
        if t == "DNaseI":
            yinit0.append(transform_DNase(y))
        if t == "OKSeq":
            yinit0.append((y+1)/2)


    yinit = np.array(yinit0).T
    yinit[np.isnan(yinit)] = 0
    #print(yinit.shape)
    return df,yinit,notnan


def window_stack(a, stepsize=1, width=3):
    #print([[i,1+i-width or None,stepsize] for i in range(0,width)])
    return np.hstack( a[i:1+i-width or None:stepsize] for i in range(0,width) )

def transform_seq(Xt,yt,stepsize=1,width=3):
    # X = (seq,dim)
    # y = (seq)
    Xt = np.array(Xt)
    yt = np.array(yt)

    print(Xt.shape,yt.shape)

    assert(len(Xt.shape)==2)
    assert(len(yt.shape)==2)
    assert(width % 2 == 1)
    X = window_stack(Xt,stepsize,width).reshape(-1,width,Xt.shape[-1])[::,np.newaxis,::,::]
    Y  = window_stack(yt[::,np.newaxis],stepsize,width)[::,width//2]#[::,np.newaxis] #Take the value at the middle of the segment

    print(X.shape,Y.shape)

    return [X,Y]

def create_model_o(X_train,targets,nfilters,kernel_length):



    K.set_image_data_format('channels_last')




    multi_layer_keras_model=Sequential()
    multi_layer_keras_model.add(Conv2D(filters=nfilters,kernel_size=(1,kernel_length),input_shape=X_train.shape[1:]))
    multi_layer_keras_model.add(Activation('relu'))
    multi_layer_keras_model.add(Dropout(0.2))

    multi_layer_keras_model.add(Conv2D(filters=nfilters,kernel_size=(1,kernel_length),input_shape=X_train.shape[1:]))
    multi_layer_keras_model.add(Activation('relu'))
    multi_layer_keras_model.add(Dropout(0.2))

    multi_layer_keras_model.add(Conv2D(filters=nfilters,kernel_size=(1,kernel_length),input_shape=X_train.shape[1:]))
    multi_layer_keras_model.add(Activation('relu'))
    multi_layer_keras_model.add(MaxPooling2D(pool_size=(1,2)))
    multi_layer_keras_model.add(Dropout(0.2))


    multi_layer_keras_model.add(Flatten())
    multi_layer_keras_model.add(Dense(len(targets)))
    multi_layer_keras_model.add(Activation("sigmoid"))
    multi_layer_keras_model.compile(optimizer='adam',
                               loss='mse')
    multi_layer_keras_model.summary()
    return multi_layer_keras_model

def create_model(X_train,targets,nfilters,kernel_length):

    from keras.layers import Input, Dense
    from keras.models import Model

    X = Input(shape=X_train.shape[1:])
    l1 = Conv2D(filters=nfilters,kernel_size=(1,kernel_length),activation="relu",padding="same")(X)
    l1p = Dropout(0.2)(l1)
    l2 = Conv2D(filters=nfilters//2,kernel_size=(1,kernel_length),activation="relu",padding="same")(l1p)
    l2p = Dropout(0.2)(l2)
    l2b = Conv2D(filters=nfilters//4,kernel_size=(1,kernel_length),activation="relu",padding="same")(l2p)
    l2bp = Dropout(0.2)(l2b)
    density_activation = Conv2D(name="density",filters=1,kernel_size=(1,kernel_length),activation="sigmoid",padding="same")(l2bp)
    #l3max = MaxPooling2D(pool_size=(1,2))(density_activation)
    l3p = Dropout(0.2)(density_activation)
    l3pi = Conv2D(filters=30,kernel_size=(1,4*kernel_length),activation="relu")(l3p)
    l3pi = Conv2D(filters=30,kernel_size=(1,4*kernel_length),activation="relu")(l3pi)

    l3pf = Flatten()(l3pi)
    Outputs = Dense(len(targets),activation="sigmoid")(l3pf)

    multi_layer_keras_model = Model(inputs=X,outputs=[Outputs,density_activation])
    multi_layer_keras_model.compile(optimizer='adam',
                               loss='mse')
    multi_layer_keras_model.summary()
    return multi_layer_keras_model


def train_test_split(chrom,ch_train,ch_test,notnan):
    print(list(ch_train),list(ch_test))
    chltrain = ["chr%i"%i for i in ch_train]
    chltest = ["chr%i"%i for i in ch_test]
    train = [chi in chltrain and notna for chi,notna in zip(chrom.chrom,notnan)]
    test = [chi in chltest and notna for chi,notna in zip(chrom.chrom,notnan)]
    print(np.sum(train),np.sum(test),np.sum(test)/len(test))
    return train,test

def unison_shuffled_copies(a, b,c=None):
    assert len(a) == len(b)
    if c is not None:
        assert len(c) == len(a)
    p = np.random.permutation(len(a))
    if c is None:
        return a[p], b[p]
    else:
        return a[p], b[p],c[p]

def repad1d(res,window):
    return np.concatenate([np.zeros(window//2),res,np.zeros(window//2)])


if __name__ == "__main__":

    import argparse
    import os

    parser = argparse.ArgumentParser()

    parser.add_argument('--cell', type=str, default=None)
    parser.add_argument('--rootnn', type=str, default=None)
    parser.add_argument('--nfilters', type=int, default=15)
    parser.add_argument('--window', type=int, default=51)
    parser.add_argument('--kernel_length', type=int, default=10)
    parser.add_argument('--weight', type=str, default=None)
    parser.add_argument('--marks', nargs='+', type=str, default=[])
    parser.add_argument('--targets', nargs='+', type=str, default=["initiation"])



    args = parser.parse_args()

    cell = args.cell
    rootnn = args.rootnn
    window=args.window
    marks = args.marks
    if marks == []:
        marks = ['H2az', 'H3k27ac', 'H3k79me2', 'H3k27me3', 'H3k9ac', 'H3k4me2', 'H3k4me3', 'H3k9me3', 'H3k4me1', 'H3k36me3',"H4k20me1"]


    lcell = [cell]

    if cell == "all":
        lcell = ["K562","GM","Hela"]

    os.makedirs(args.rootnn,exist_ok=True)


    root= "/home/jarbona/projet_yeast_replication/notebooks/DNaseI/repli1d/"
    XC = pd.read_csv(root + "coords_K562.csv",sep="\t")  # List of chromosome coordinates

    if args.weight is None:
        X_train = []
        for cellt in lcell:
            name = "/home/jarbona/repli1D/data/mlformat_whole_sig_%s_dec2.csv" % cellt
            name = "/home/jarbona/repli1D/data/mlformat_whole_sig_standard%s_dec2.csv" % cellt
            print(name)
            df,yinit,notnan = load_signal(name,marks,targets=args.targets)


            traint = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13,14,  15, 16, 17, 19, 21,22]
            valt = [4,18]
            testt = [5,20]
            for v in testt:
                assert(v not in traint)
            for v in valt:
                assert(v not in traint)
            train,test = train_test_split(XC,traint,valt,notnan)
            X_train_us, X_test_us, y_train_us, y_test_us = df[train],df[test],yinit[train],yinit[test]

            print(y_train_us.shape)
            vtrain = transform_seq(X_train_us,y_train_us,1,window)
            vtrain_a = transform_seq(y_train_us,y_train_us[::,:1],1,window)
            #a_train = vtrain_a[0][::,::,(args.kernel_length-1)*2:-(args.kernel_length-1)*2,-1:]
            a_train = vtrain_a[0][::,::,::,-1:]
            vtrain = vtrain + [a_train]

            vtest = transform_seq(X_test_us,y_test_us,1,window)
            vtest_a = transform_seq(y_test_us,y_test_us[::,:1],1,window)
            #a_test = vtest_a[0][::,::,(args.kernel_length-1)*2:-(args.kernel_length-1)*2,-1:]
            a_test = vtest_a[0][::,::,::,-1:]
            vtest = vtest+[a_test]


            if X_train == []:
                X_train,y_train,a_train = vtrain
                X_test,y_test,a_test =  vtest
            else:
                X_train = np.concatenate([X_train,vtrain[0]])
                y_train = np.concatenate([y_train,vtrain[1]])
                a_train = np.concatenate([a_train,vtrain[2]])
                X_test = np.concatenate([X_test,vtest[0]])
                y_test = np.concatenate([y_test,vtest[1]])
                a_test = np.concatenate([a_test,vtest[2]])

        X_train,y_train,a_train = unison_shuffled_copies(X_train,y_train,a_train)



    if args.weight is not None:
        from keras.models import load_model
        multi_layer_keras_model = load_model(args.weight)

    else:
        multi_layer_keras_model = create_model(X_train,targets=args.targets,nfilters=args.nfilters,kernel_length=args.kernel_length)

        if (len(args.targets) == 1) and (args.targets[0] == "OKSeq"):

            selpercents = [1.0]
        else:
            selpercents = [0.1,1.0,5.0,"all"]
            selpercents = [0.1,"all"]

        for selp in selpercents:
            print(sum(y_train==0),sum(y_train!=0))
            if type(selp) == float:
                sel = y_train[::,0] != 0
                sel[np.random.randint(0,len(sel-1),int(selp*sum(sel)))] = True
            else:

                sel[::]=True
            print(np.sum(sel),sel.shape)
            print(X_train.shape,X_train[sel].shape)


            history_multi_filter=multi_layer_keras_model.fit(x=X_train[sel],
                                              y=[y_train[sel],a_train[sel]],
                                              batch_size=64, # 128
                                              epochs=150,
                                              verbose=1,
                                              callbacks=[EarlyStopping(patience=3),
                                                        History(),
                                                        ModelCheckpoint(save_best_only=True,
                                                                        filepath=rootnn+"/%sweights.{epoch:02d}-{val_loss:.4f}.hdf5" % cell,verbose=1)],
                                              validation_data=(X_test,
                                                               [y_test,a_test]))


            multi_layer_keras_model.save(rootnn+"/%sweights.hdf5" % cell)
            print("Saving on",rootnn+"/%sweights.hdf5" % cell)
    ###################################
    #predict
    for cellp in ["K562","Hela","GM"]:
        namep = "/home/jarbona/repli1D/data/mlformat_whole_sig_%s_dec2.csv" % cellp
        namep = "/home/jarbona/repli1D/data/mlformat_whole_sig_standard%s_dec2.csv" % cellp

        df,yinit,notnan = load_signal(namep,marks,targets=args.targets)
        X,y = transform_seq(df,yinit,1,window)
        print(X.shape)
        res = multi_layer_keras_model.predict(X)[0]

        print(res.shape,"resshape")

        for itarget,target in enumerate(args.targets):
            XC["signalValue"] = repad1d(res[::,itarget],window)
            if target == "OKSeq":
                XC["signalValue"] = XC["signalValue"] * 2-1
        #XC.to_csv("nn_hela_fk.csv",index=False,sep="\t")
            if target == "initiation":
                ns = rootnn+"/nn_%s_from_%s.csv" % (cellp,cell)
            else:
                ns = rootnn+"/nn_%s_%s_from_%s.csv" % (cellp,target,cell)

            print("Saving to",ns)
            XC.to_csv(ns,index=False,sep="\t")
