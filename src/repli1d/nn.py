import os

import numpy as np
import pandas as pd

from repli1d.analyse_RFD import nan_polate, smooth
from repli1d.models import jm_cnn_model


def normal_seq(signal, q=99, output_path='../data/'):
    """
    normalization function that transforms each fature in range (0,1)
    and outputs the minimum and maximum of features in a csv file in
    data folder inside the repository, suitable for future transformation
    on new dataset in a trained
    neural network.

    Parameters
    ----------
    signal : numpy array or pandas dataframe
    in the shape of (n_samples, n_features)
    output_path : str, default='../data/'
    q : float, default=99
    the quantile threshold, to act like a lowerpass filter
    to remove the outliers. The q is in percentage, this function substitutes
    (100-q) quantile from reversed sorted data by the quantile of data that
    specified by user. if user set q=None there would be no denoising and it
    would scale the input by its minimum, and its maximum.
    Returns
    -------
    transformed : numpy array
        a normalised sequence or features in the range (0,1)
    """
    max_element = []
    min_element = []
    transformed = []
    if isinstance(signal, pd.DataFrame):
        signal = signal.to_numpy(copy=True)
    elif isinstance(signal, list):
        signal = np.array(signal)
    if signal.ndim == 1:
        if q is not None:
            max_element = np.percentile(signal, q)
        else:
            max_element = max(signal)
        min_element = min(signal)
        signal[signal > max_element] = max_element
        transformed.append((signal-min_element)/(
                            max_element-min_element))
    else:
        if q is not None:
            max_element = np.percentile(signal, q, axis=0)
        else:
            max_element = signal.max(axis=0)
        for i in range(signal.shape[1]):
            min_element.append(min(signal[:, i]))
            signal[signal[:, i] > max_element[i]] = max_element[i]
            transformed.append((signal[:, i]-min_element[i])/(
                                max_element[i]-min_element[i]))
    transformed = np.array(transformed).T  # transpose for correspondence
    if output_path is not None:
        result = pd.DataFrame((min_element, max_element), index=['minimum',
                                                                 'maximum'])
        result.to_csv(output_path + 'min_max_inputs.csv')
    return transformed


def inv_transform(signal, input_path='../data/'):
    """
    Inversre transform is a function for transforming the output of NN to the
    scale of real dataset.

    Parameters
    ----------
    signal : numpy array or pandas dataframe
    in the shape of (n_samples, n_features)
    input_path : str, default='../data/'
    the address of a folder that contains min_max_outputs.csv.
    Returns
    -------
    inv_transformed : numpy array
    """
    if isinstance(signal, pd.DataFrame):
        signal = signal.to_numpy(copy=True)
    scales = pd.read_csv(input_path + 'min_max_outputs.csv')
    min_s = scales.to_numpy(copy=True)[0, 1:]
    max_s = scales.to_numpy(copy=True)[1, 1:]
    scales = max_s - min_s
    scales = scales.reshape(1, -1)
    inv_transformed = np.multiply(signal, scales) + min_s
    return inv_transformed


def dev_transform(signal, input_path='../data/', is_denoised=True):
    """
    normalization function that transforms each fature based on the
    scaling of the trainning set. This transformation should be done on
    test set(developmental set), or any new input for a trained neural
    network. Due to existence of a denoising step in the normal_seq funciton,
    this transformation can not reproduce the exact same of initial sequences,
    instead it transforms to the scale of denoised version of training set.

    Parameters
    ----------
    signal : numpy array or pandas dataframe
    in the shape of (n_samples, n_features)
    input_path : str, default='../data/'
    is_denoised : boolean
    it specifies the state if original sequence is denoised by a threshold,
    if it's set to False it means that user used q=None in normal_seq function.
    Returns
    -------
    transformed : numpy array
        a normalised sequence or features
    """
    transformed = []
    if isinstance(signal, pd.DataFrame):
        signal = signal.to_numpy(copy=True)
    elif isinstance(signal, list):
        signal = np.array(signal)
    scales = pd.read_csv(input_path + 'min_max_inputs.csv')
    max_element = scales.to_numpy(copy=True)[1, 1:]
    min_element = scales.to_numpy(copy=True)[0, 1:]
    if signal.ndim == 1:
        if is_denoised is True:
            signal[signal > max_element] = max_element
        transformed.append((signal-min_element)/(
                            max_element-min_element))
    else:
        for i in range(signal.shape[1]):
            if is_denoised is True:
                signal[signal[:, i] > max_element[i]] = max_element[i]
            transformed.append((signal[:, i]-min_element[i])/(
                                max_element[i]-min_element[i]))
    transformed = np.array(transformed).T  # transpose for correspondence
    return transformed


def transform_norm(signal):
    s = np.array(signal).copy()
    s -= np.percentile(s, 10)
    p = np.percentile(s, 50)
    if p == 0:
        p = np.mean(s)
    s /= p
    s /= 5
    s[s > 50] = 50
    return np.array(s, dtype=np.float32)  # mod


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


def filter_anomalyf(signal, smv, percentile, nf):
    for n in range(nf):
        delta = np.abs(signal-smooth(signal, smv))
        p = np.percentile(np.abs(delta), percentile)
        signal[np.abs(delta) > p] = np.nan
        signal = nan_polate(signal)
    return signal


def load_signal(name,
                marks=["H3K4me1", "H3K4me3", "H3K27me3", "H3K36me3",
                       "H3K9me3", "H2A.Z", "H3K79me2", "H3K9ac", "H3K4me2",
                       "H3K27ac", "H4K20me1"],
                targets=["initiation"], t_norm=None, smm=None, wig=True,
                augment=None, show=True, add_noise=False,
                filter_anomaly=False,
                repertory_scaling_param="../data/"):
    """
    This function does some modification on datset based on its column names
    and also invoke the scaling methods for different features and outputs,
    it also makes a mask for different chromosomes. to be able to
    adapt the method for different chromosomes it is necessary to call
    load_signal, and transform_seq for training set and then invoke them for
    test set or any other set (revoking two consequent load_signal on two
    different dataset then tranform_seq them may return wrong stacked
    sequences), it is necessary due to variable that defines in load_signal.

    Parameters
    ----------
    name : str or pd.Dataframe
    the address of a csv file or pandas dataframe
    marks : list
    a list that contains the names of markers as features for NN.
    targets : list
    a list that contains columns names of desired outputs of NN.
    repertory_scaling_param : str
    the address to save the scaling parameters in it.
    Returns
    -------
    df : numpy array
        a scaled dataset of features
    y_init : numpy array
        a scaled dataset of outputs
    notnan : numpy array
    """
    if type(name) == str:
        df = pd.read_csv(name)

    # wig = True
    mask_borders = np.cumsum(df.chrom.value_counts().to_numpy(copy=True))
    if "signal" in df.columns:
        df["initiation"] = df["signal"]

    if wig:
        lm = ["DNaseI", "initiation", "Meth", "Meth450", "RFDs", "MRTs",
              "RFDe", "MRTe", "AT_20", "RNA_seq", "AT_5", "AT_30"]
        marks0 = [m+"wig" for m in marks if m not in lm]
        for sm in lm:
            if sm in marks:
                marks0 += [sm]

        assert(len(marks) == len(marks0))
        marks = marks0

    if "notnan" in df.columns:
        if show:
            print("Found notnan")
        notnan = df["notnan"]
    else:
        notnan = []

    df = df[targets+marks]
    if show:
        print(df.describe())

    yinit = [df.pop(target) for target in targets]
    # print(yinit.shape,"Yinit shape")

    if t_norm is not None:
        transform_norm = t_norm

    if transform_norm == normal_seq:
        df = pd.DataFrame(transform_norm(df,
                                         output_path=repertory_scaling_param))
    else:
        for col in df.columns:
            if show:
                print(col)
            if col not in ["DNaseI", "initiation", "Meth", "Meth450", "RFDe",
                           "MRTe", "RFDs", "MRTs"]:
                df[col] = transform_norm(df[col])
            elif col == "DNaseI":
                df[col] = transform_DNase(df[col])
            elif col in ["initiation", "Stall"]:
                df[col] = df[col] / np.max(df[col])
            elif "Meth" in col:
                df[col] = transform_norm_meth(df[col])
            elif "RFD" in col:
                if "RFD" in col:
                    if col == "RFDe" and filter_anomaly:
                        df[col] = filter_anomalyf(df[col].copy(), smv=5,
                                                  percentile=98.5, nf=4)

                    # print("Nanpo")
                    df[col] = nan_polate(df[col])
                if add_noise and col == "RFDs":
                    print("Noise: ", int(len(df)*0.01))
                    for p in np.random.randint(0, len(df),
                                               size=int(len(df)*0.01)):  # article 1%
                        df[col][p] = 2*np.random.rand()-1

                if smm is not None:
                    df[col] = smooth(df[col], smm)
                df[col] = (df[col]+1)/2
            elif "MRT" in col:
                if "MRT" in col:
                    df[col] = nan_polate(df[col])
                    if augment == "test":
                        for asm in [10, 50, 200]:
                            df[col+f"_sm_{asm}"] = smooth(nan_polate(df[col]), asm)
                            df[col + f"_sm_{asm}"] -= np.mean(df[col+f"_sm_{asm}"])
                            df[col + f"_sm_{asm}"] /= np.std(df[col + f"_sm_{asm}"])

                pass

            if np.sum(np.isnan(df[col])) != 0:
                raise "NanVal"

    if show:
        print(np.max(yinit[0]), "max")
        print(df.describe())

    yinit0 = []
    min_outputs = []
    max_outputs = []
    for y, t in zip(yinit, targets):
        if t in ["initiation", "Stall"]:
            max_outputs.append(np.max(y))
            min_outputs.append(np.min(y))
            trunc = (y - np.min(y)) / (np.max(y)-np.min(y))  # np.percentile(y,99)
            # trunc[trunc>1] = 1
            result = pd.DataFrame((min_outputs, max_outputs), index=['minimum',
                                                                     'maximum'])
            result.to_csv(os.path.join(repertory_scaling_param,
                                       'min_max_outputs.csv'))
            yinit0.append(trunc)

        elif t == "DNaseI":
            yinit0.append(transform_DNase(y))
        elif t == "OKSeq":
            yinit0.append((y+1)/2)
        elif t == "ORC2":
            yinit0.append(y)
        else:
            raise "Undefined target"

    yinit = np.array(yinit0).T
    yinit[np.isnan(yinit)] = 0
    # print(yinit.shape)
    """
    import pylab
    f=pylab.figure()
    pylab.plot(yinit)
    pylab.plot(df["RFDs"])
    pylab.show()
    """
    dict = {"df": df,
            "yinit": yinit,
            "notnan": notnan,
            "mask_borders": mask_borders}
    return dict


def window_stack(a, mask_borders, stepsize=1, width=3):
    """
    This function makes windows of the size specified as 'width'
    and sweeping over dataset with the specified step size.

    Parameters
    ----------
    a : numpy array
    in the shape of (n_samples, n_features)
    step_size : int
    width : int
    mask_borders : list
    list of end positions of each chromosome as elements along
    the first axis of dataset.
    Returns
    -------
    window_stacked : numpy array or pandas dataframe
        in the shape of (n_windows, n_features*width)
        an array of stacked windows, column wise.
    """
    window_stacked = []
    # print([[i,1+i-width or None,stepsize] for i in range(0,width)])

    for index, elem in enumerate(mask_borders):
        if index != 0:
            boundary = mask_borders[index-1]
        else:
            boundary = 0
        b = a[boundary: elem]
        window_stacked.append(np.hstack([b[i:1+i-width or None:stepsize] for i in range(0, width)]))
    if len(mask_borders) == 1:
        return window_stacked[0]
    else :
        return np.concatenate(window_stacked)
    #return window_stacked


def transform_seq(Xt, yt, mask_borders,stepsize=1, width=3, impair=True):
    """
    This function reshapes the output of window_stack function into a
    suitable shape for neural network.

    Parameters
    ----------
    Xt : numpy array
    in the shape of (n_samples, n_features)
    yt : numpy array
    in the shape of (n_samples, n_features)
    step_size : int
    width : int
    impair : bool
    Returns
    -------
    X : numpy array
        in the shape of (n_windows, 1, width, n_features)
    Y : numpy array
        in the shape of (n_windows, n_outputs)
    """
    # X = (seq,dim)
    # y = (seq)
    # Xt = np.array(Xt, dtype=np.float16)
    yt = np.array(yt, dtype=np.float32)
    # print(Xt.shape, yt.shape)

    assert(len(Xt.shape) == 2)
    assert(len(yt.shape) == 2)
    if impair:
        assert(width % 2 == 1)
    X = window_stack(Xt, mask_borders, stepsize, width).reshape(-1, width, Xt.shape[-1])[::, np.newaxis, ::, ::]
    # [::,np.newaxis] #Take the value at the middle of the segment
    Y = window_stack(yt[::, np.newaxis], mask_borders, stepsize, width)[::, width//2]

    # print(X.shape, Y.shape)
    # exit()

    return X, Y

def window_stack_single(a, stepsize=1, width=3):
    # print([[i,1+i-width or None,stepsize] for i in range(0,width)])
    return np.hstack([a[i:1+i-width or None:stepsize] for i in range(0, width)])

def reshape_for_batch(selected,step_size,window_size):
    return window_stack_single(selected,step_size,window_size).reshape(-1, window_size, selected.shape[-1])

def compute_all_possible_batches(data_x,window_size,step_size,batch_size,drop_remainder=True):

    tot_size=window_size + step_size * (batch_size-1)
    all_possible_batches = []
    for ch in range(len(data_x)):
        for pos in np.arange(0,len(data_x[ch])-tot_size+1,batch_size*step_size):
            all_possible_batches.append([ch,int(pos),tot_size])
        if not drop_remainder:
            left = len(data_x[ch]) - (int(pos)+batch_size)
            if left> 0 :
                all_possible_batches.append([ch,int(pos)+batch_size,left])
            if left < window_size:
                raise

    return tot_size,all_possible_batches

def split(data,mask_borders):
    splited = []
    for index, elem in enumerate(mask_borders):
        if index != 0:
            boundary = mask_borders[index-1]
        else:
            boundary = 0
        splited.append(data[boundary: elem])
    return splited

def n_steps(x,step_size,window_size,batch_size):
    tot_size,all_possible_batches = compute_all_possible_batches(x,window_size,step_size,batch_size)
    return len(all_possible_batches)
def generator(x,y,window_size,step_size,batch_size,random=True,drop_remainder=True):

    # create the list of all possible batch:
    tot_size,all_possible_batches = compute_all_possible_batches(x,window_size,step_size,batch_size,drop_remainder)
    #print(all_possible_batches)
    while True:
        if random:
            perm = np.random.permutation(len(all_possible_batches))
        else:
            perm = np.arange(len(all_possible_batches))
        #print("Number of batches",len(all_possible_batches))
        for p in perm:
            which,start,size_b=all_possible_batches[p]
            yield reshape_for_batch(x[which][start:start+size_b],step_size,window_size)[::, np.newaxis, ::, ::],\
                  reshape_for_batch(y[which][start:start+size_b],step_size,window_size)[:,window_size//2]

#For tf but is really slow
"""
def compute_all_possible_sequences(lengths,window_size,step_size):
    print(lengths,window_size,step_size)
    tot_size=window_size
    all_possible_sequences = []
    shift=0
    for length in lengths:
        #print(lenght)
        for pos in np.arange(0,length-tot_size+1,step_size):
            all_possible_sequences.append(int(pos))
        shift += length
    return all_possible_sequences

def generator_no_batch(data_x,data_y,all_possible_sequences ,window_size,step_size,random=True):

    # create the list of all possible batch:
    #while True:
    tot_size=window_size
    if random:
        perm = np.random.permutation(len(all_possible_sequences))
    else:
        perm = np.arange(len(all_possible_sequences))
    #print("Number of batches",len(all_possible_batches))
    for p in perm:
        start=all_possible_sequences[p]
        yield data_x[start:start+tot_size][np.newaxis,...],\
              data_y[start:start+tot_size][window_size//2]
"""
def train_test_split(chrom, ch_train, ch_test, notnan):
    print(list(ch_train), list(ch_test))

    chltrain = ch_train
    chltest = ch_test
    if len(notnan) != 0:
        train = [chi in chltrain and notna for chi, notna in zip(chrom.chrom, notnan)]
        test = [chi in chltest and notna for chi, notna in zip(chrom.chrom, notnan)]
    else:
        print("Working on all (no nan)")
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
    from tensorflow.keras.callbacks import (EarlyStopping, History, ModelCheckpoint,
                                 ReduceLROnPlateau)
    from repli1d.models import jm_cnn_model as create_model
    from keras.models import  load_model
    import tensorflow as tf


    parser = argparse.ArgumentParser()

    parser.add_argument('--cell', type=str, default=None)
    parser.add_argument('--ml_model', type=str, default=jm_cnn_model)
    parser.add_argument('--rootnn', type=str, default=None)
    parser.add_argument('--nfilters', type=int, default=15)
    parser.add_argument('--resolution', type=int, default=5)
    parser.add_argument('--sm', type=int, default=None)  # Smoothing exp data

    parser.add_argument('--window', type=int, default=51)
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--imp', action="store_true")
    parser.add_argument('--reduce_lr', action="store_true")

    parser.add_argument('--wig', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--kernel_length', type=int, default=10)
    parser.add_argument('--weight', type=str, default=None)
    parser.add_argument('--loss', type=str, default="binary_crossentropy")
    parser.add_argument('--augment', type=str, default="")

    parser.add_argument('--marks', nargs='+', type=str, default=[])
    parser.add_argument('--targets', nargs='+', type=str, default=["initiation"])
    parser.add_argument('--listfile', nargs='+', type=str, default=[])
    parser.add_argument('--enrichment', nargs='+', type=float, default=[0.1, 1.0, 5.0])
    parser.add_argument('--roadmap', action="store_true")
    parser.add_argument('--noenrichment', action="store_true")
    parser.add_argument('--predict_files', nargs='+', type=str, default=[])

    parser.add_argument('--restart', action="store_true")
    parser.add_argument('--datafile', action="store_true")
    parser.add_argument('--add_noise', action="store_true")
    parser.add_argument('--filter_anomaly', action="store_true")
    parser.add_argument('--generator', action="store_true")


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
    if not args.datafile:
        if args.resolution == 5:
            XC = pd.read_csv(root + "coords_K562.csv", sep="\t")  # List of chromosome coordinates
        if args.resolution == 1:
            XC = pd.read_csv("data/Hela_peak_1_kb.csv", sep="\t")

    if args.listfile == []:
        listfile = []
        for cellt in lcell:
            name = "/home/jarbona/repli1D/data/mlformat_whole_sig_%s_dec2.csv" % cellt
            name = "/home/jarbona/repli1D/data/mlformat_whole_sig_standard%s_dec2.csv" % cellt
            name = "/home/jarbona/repli1D/data/mlformat_whole_sig_standard%s_nn.csv" % cellt
            wig = True
            if args.roadmap:
                name = "/home/jarbona/repli1D/data/roadmap_%s_nn.csv" % cellt
                wig = False

            listfile.append(name)
    else:
        listfile = args.listfile
        wig = False

    if args.wig is not None:
        if args.wig == 1:
            wig = True
        else:
            wig = False
    if args.weight is None or args.restart:
        X_train = []
        for name in listfile:
            print(name)
            temp_dict = load_signal(
                name, marks, targets=args.targets, t_norm=transform_norm,
                smm=args.sm, wig=wig, augment=args.augment,
                add_noise=args.add_noise,repertory_scaling_param=args.rootnn+"/")
            df, yinit, notnan, mask_borders = temp_dict.values()
            """
            traint = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19]
            valt = [4, 18, 21, 22]
            testt = [5, 20]
            """
            if not args.datafile:
                traint = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                          18, 19] + [20, 21, 22, 23]
                valt = [20, 21, 22, 23]
                valt = [2]
                testt = [1]  # 1]

                traint = ["chr%i" for i in traint]
                valt = ["chr%i" for i in valt]
                testt = ["chr%i" for i in testt]

            else:
                XC = pd.read_csv(args.listfile[0])
                chs = set(XC.chrom)
                traint = list(chs)
                tests = ["chr1"]
                valt = ["chr2"]
                traint.remove(tests[0])
                traint.remove(valt[0])

                # traint.pop(0)

            if not args.datafile:
                for v in testt:
                    assert(v not in traint)
                for v in valt:
                    assert(v not in traint)
            train, val = train_test_split(XC, traint, valt, notnan)
            X_train_us, X_val_us, y_train_us, y_val_us = df[train], df[val], yinit[train], yinit[val]

            if not args.generator:
                vtrain = transform_seq(X_train_us, y_train_us, mask_borders, 1, window)
                vval = transform_seq(X_val_us, y_val_us, mask_borders, 1, window)
                del X_train_us, X_val_us, y_train_us, y_val_us
                if X_train == []:
                    X_train, y_train = vtrain
                    X_val, y_val = vval
                else:
                    X_train = np.concatenate([X_train, vtrain[0]])
                    y_train = np.concatenate([y_train, vtrain[1]])
                    X_val = np.concatenate([X_val, vval[0]])
                    y_val = np.concatenate([y_val, vval[1]])
            else:
                x = split(df,mask_borders)
                y = split(yinit,mask_borders)
                assert(len(x)==(len(traint)+len(tests)+len(valt)))


                vtrain = generator(x[2:], y[2:],  window,1,args.batch_size,random=True,drop_remainder=True)
                train_steps = len(compute_all_possible_batches(x[2:],window,1,args.batch_size,drop_remainder=True)[1])
                vval = generator(x[1:2], y[1:2], window,1,args.batch_size,random=True,drop_remainder=True)
                val_steps = len(compute_all_possible_batches(x[1:2],window,1,args.batch_size,drop_remainder=True)[1])

                """
                print([len(xx) for xx in x[2:]])
                all_possible_sequences = compute_all_possible_sequences([len(xx) for xx in x[2:]] , window,step_size=1)
                vtrain =  tf.data.Dataset.from_generator(generator_no_batch,
                                                        args=[np.concatenate(x[2:]),
                                                              np.concatenate(y[2:]),
                                                              all_possible_sequences,window,1,False],
                                                        output_types=(tf.float32,tf.float32),
                                                        output_shapes = ((1,window,x[0].shape[-1]),(y[0].shape[-1])))


                vtrain = vtrain.batch(args.batch_size, drop_remainder=True)
                train_steps = len(all_possible_sequences) // args.batch_size
                print("Train steps",train_steps,len(all_possible_sequences))

                #lengths= [len(xx) for xx in x[1:2]]
                all_possible_sequences = compute_all_possible_sequences([len(xx) for xx in x[1:2]] , window,step_size=1)
                vval =  tf.data.Dataset.from_generator(generator_no_batch,
                                                        args=[x[1],y[1],
                                                        all_possible_sequences,window,1,False],
                                                        output_types=(tf.float32,tf.float32),
                                                        output_shapes = ((1,window,x[0].shape[-1]),(y[0].shape[-1])))


                vval = vval.batch(args.batch_size, drop_remainder=True)
                val_steps = len(all_possible_sequences) // args.batch_size
                """

                for x,y in vtrain:
                    break
                print("Shapes",x.shape,y.shape)
        if not args.generator:
            X_train, y_train = unison_shuffled_copies(X_train, y_train)

            n = X_train.shape[0] * X_train.shape[2]
            if n > 1e9:
                nmax = int(0.5e9//X_train.shape[2])
                print(nmax)
                X_train = X_train[:nmax]
                y_train = y_train[:nmax]

            print("Shape", X_train.shape, y_train.shape)

    weight=None
    if (args.weight is not None) or os.path.exists(rootnn+"/%sweights.hdf5" % cell):
        weight= args.weight
        if weight is None:
            weight = rootnn+"/%sweights.hdf5" % cell


        multi_layer_keras_model = load_model(weight)

        multi_layer_keras_model.summary()
        if not args.generator:
            del X_train, y_train

    if not args.restart and weight is not None:
        #load_model(args.weight)
        pass

    else:
        if not args.generator:
            X_train_shape=X_train.shape
        else:
            X_train_shape=(None,1,window,X_train_us.shape[-1])
        if not args.imp:
            multi_layer_keras_model = create_model(
                X_train_shape, targets=args.targets, nfilters=args.nfilters,
                kernel_length=args.kernel_length, loss=args.loss)
        else:
            multi_layer_keras_model = create_model_imp(
                X_train, targets=args.targets, nfilters=args.nfilters,
                kernel_length=args.kernel_length, loss=args.loss)

        if args.restart:
            multi_layer_keras_model = load_model(args.weight)


        totenr = args.enrichment + ["all"]
        if args.noenrichment:
            totenr = ["all"]

        print(totenr)
        for selp in totenr:
            if not args.generator:
                print(sum(y_train == 0), sum(y_train != 0))
                if type(selp) == float:
                    sel = y_train[::, 0] != 0
                    th = np.percentile(y_train[::, 0], 100-selp)
                    print("sepp,th", selp, th)
                    sel = y_train[::, 0] > th
                    # sel = y_train[::, 0] > 0.2
                    """
                    if sum(sel)/len(sel) > selp:
                        th = np.percentile(sel,100-100*selp)
                        print(th)
                        sel = y_train[::, 0] > th
                    """
                    print("top %i , Total %i, selected %i" % (sum(sel), len(sel), int(0.01*selp*len(sel))))
                    sel[np.random.randint(0, len(sel-1), int(0.01*selp*len(sel)))] = True
                    print("Chekc", np.sum(sel))
                else:

                    sel = np.ones_like(y_train[::, 0], dtype=np.bool)
                print(np.sum(sel), sel.shape)
                print(X_train.shape, X_train[sel].shape)

            cp = [EarlyStopping(patience=3)]
            if selp == "all" and False:
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                              patience=3, min_lr=0.0001)
                cp = [reduce_lr]
            if args.reduce_lr:
                cp = [EarlyStopping(patience=5),
                      ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                        patience=3, min_lr=0.0001)]

            if not args.generator:
                validation_data = (X_val, y_val)
                validation_split = 0.
                history_multi_filter = multi_layer_keras_model.fit(x=X_train[sel],
                                                                   y=y_train[sel],
                                                                   batch_size=args.batch_size,
                                                                   epochs=args.max_epoch,
                                                                   verbose=1,
                                                                   callbacks=cp+[History(),
                                                                                 ModelCheckpoint(save_best_only=True,
                                                                                                 filepath=rootnn+"/%sweights.{epoch:02d}-{val_loss:.4f}.hdf5" % cell,
                                                                                                 verbose=1)],
                                                                   validation_data=validation_data,
                                                                   validation_split=validation_split)
            else:
                history_multi_filter = multi_layer_keras_model.fit_generator(vtrain,
                                                                            workers=4,
                                                                            use_multiprocessing=True,
                                                                            steps_per_epoch=train_steps,
                                                                            epochs=args.max_epoch,
                                                                               verbose=1,
                                                                               callbacks=cp+[History(),
                                                                                             ModelCheckpoint(save_best_only=True,
                                                                                                             filepath=rootnn+"/%sweights.{epoch:02d}-{val_loss:.4f}.hdf5" % cell,
                                                                                                             verbose=1)],
                                                                   validation_data=vval,
                                                                   validation_steps=val_steps)


            multi_layer_keras_model.save(rootnn+"/%sweights.hdf5" % cell)
            print("Saving on", rootnn+"/%sweights.hdf5" % cell)
        if not args.generator:
            del X_train, y_train
    ###################################
    # predict
    print("Predict")
    if args.listfile == [] or args.roadmap or (len(args.predict_files) != 0):
        if marks == ["RFDs", "MRTs"]:
            marks = ["RFDe", "MRTe"]
        to_pred = []
        if len(args.predict_files) == 0:
            lcell = ["K562", "Hela", "GM"]
            if args.cell is not None and args.weight is not None:
                lcell = [args.cell]
            for cellp in lcell:
                namep = "/home/jarbona/repli1D/data/mlformat_whole_sig_%s_dec2.csv" % cellp
                namep = "/home/jarbona/repli1D/data/mlformat_whole_sig_standard%s_nn.csv" % cellp
                wig = True
                if args.roadmap:
                    namep = "/home/jarbona/repli1D/data/roadmap_%s_nn.csv" % cellp
                    wig = False
            to_pred.append(namep)
        else:
            to_pred = args.predict_files

        if args.wig is not None:
            if args.wig == 1:
                wig = True
            else:
                wig = False

        for namep in to_pred:

            cellp = os.path.split(namep)[1].split("_")[0]  # namep.split("_")[-1][:-4]

            print("Reading %s, cell %s" % (namep, cellp))
            temp_dict = load_signal(
                namep, marks, targets=args.targets, t_norm=transform_norm,
                wig=wig, smm=args.sm, augment=args.augment,
                filter_anomaly=args.filter_anomaly)
            df, yinit, notnan, mask_borders = temp_dict.values()
            vtrain = generator(X_train_us, y_train_us, mask_borders, 1, window,args.batch_size)
            train_steps = n_steps(X_train_us, mask_borders, 1, window,args.batch_size)
            vtrain = generator(x[2:], y[2:], 1, window,args.batch_size)
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
            temp_dict = load_signal(
                namep, marks, targets=args.targets, t_norm=transform_norm,
                smm=args.sm, augment=args.augment,
                filter_anomaly=args.filter_anomaly)
            df, yinit, notnan, mask_borders = temp_dict.values()
            x = split(df,mask_borders)
            y = split(yinit,mask_borders)
            final = []
            for ch,yt in zip(x,y):
                pred = generator([ch],[yt],  window,1,args.batch_size,random=False,drop_remainder=False)
                steps = len(compute_all_possible_batches([ch],window,1,args.batch_size,drop_remainder=False)[1])


                res = multi_layer_keras_model.predict(pred,steps=steps)
                final.append(res)
                #print(len(res),len(ch))
            #del df, X, y
            #print(res.shape, "resshape", yinit.shape)
            for itarget, target in enumerate(args.targets):
                XC["signalValue"] = np.concatenate([repad1d(res[::, itarget], window) for res in final])
                if target == "OKSeq":
                    XC["signalValue"] = XC["signalValue"] * 2-1
            # XC.to_csv("nn_hela_fk.csv",index=False,sep="\t")
                if target in ["initiation", "Init"]:
                    namew = namep.split("/")[-1][:-4]
                    ns = rootnn+"/nn_%s.csv" % (namew)
                    s = 0
                    for y1, y2 in zip(yinit, XC["signalValue"]):
                        s += (y1-y2)**2
                    print("Average delta", s/len(yinit))
                else:
                    ns = rootnn+"/nn_%s_%s.csv" % (namew, target)
                    # print("Not implemented")
                    # ns = rootnn+"/nn_%s_%s_from_%s.csv" % (cellp, target, cell)

                print("Saving to", ns)
                XC.to_csv(ns, index=False, sep="\t")
