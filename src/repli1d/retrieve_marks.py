
from repli1d.visu_browser import plotly_blocks
from repli1d.analyse_RFD import mapboth
import numpy as np
import glob


def norm2(e, remove_zero=True, cut=None, mean=None, std=None, verbose=False, cutp=None):
    if mean is None:
        mean = np.nanmean(e)
    e /= mean
    ep = e.copy()
    if std is None:
        std = np.nanpercentile(ep, 75)-np.nanpercentile(ep, 25)

    if remove_zero:
        ep[np.isnan(ep)] = 0
    # std = np.std(ep[ep!=0])
    if cut is not None:
        print("Truncating", np.sum(ep > cut*std))
        ep[ep > cut*std] = cut*std
        print(np.nanmean(ep), std, np.nanmean(ep)+cut*std,)
    elif cutp is not None:
        if std is None:
            th = np.percentile(ep, cutp)
            std = th
        else:
            th = std


        print("Removing", np.sum(ep > th), th)
        ep[ep > th] = th
    else:
        if verbose:
            print(np.nanmean(ep), std)
    return ep, mean, std


def fit_observed_value(observed, Exp, signals, best_only=False, n=1000, student=False,sigmoid=False):
    """
    return re

    """
    # Start_values

    maxi = np.max(observed)
    print(maxi)
    #observed /= maxi
    with pm.Model() as model:

        # weight = pm.Uniform("weight", -maxi, maxi, shape=len(Exp))
        weight = pm.Laplace("weight", mu=0, b=0.01, shape=len(Exp))

        print(len(Exp)*(len(Exp)+1)/2)
        # weightp = pm.Uniform("weightp",-maxi,maxi,shape=int(len(Exp)*(len(Exp)+1)/2))


        if sigmoid:
            weightcut = pm.Laplace("weightc", mu=1, b=0.01, shape=len(Exp))
            weightstrength = pm.Laplace("weights", mu=1, b=0.01, shape=len(Exp))

            signal = theano.tensor.sum([weight[i]*(-1+2/(1+theano.tensor.exp(-weightstrength[i]*(signals[iexp]-weightcut[i])))) for i, iexp in enumerate(Exp)], axis=0)

        else:

            signal = theano.tensor.sum([weight[i]*signals[iexp] for i, iexp in enumerate(Exp)], axis=0)
            # signal2 = theano.tensor.sum([weightp[i]*signals[i]*signals[j] for i in range(len(Exp)) for j in range(i,len(Exp))],axis=0)



            # signal =nnet.relu(signal)
            """
            signal = nnet.relu(maxi-signal)
            signal = -signal + maxi
            """
        sigma = pm.HalfNormal("sigma")  #Before
        #sigma = pm.HalfNormal("sigma",tau=1/10)
        signal = nnet.relu(signal)


        if student:
            nu = pm.Uniform('nu', lower=20, upper=100)

        # Define Student T likelihood
            Observed = pm.StudentT('Observed', mu=signal+1e-7, sd=sigma/(1+observed)**0.5, nu=nu,
                                   observed=observed)
        else:
            print("Mean non zero observed",np.mean(observed[observed != 0]))
            #stdf = sigma/(1+observed)**0.05
            #ustdf = observed
            Observed = pm.Normal("Observed", mu=signal+1e-7, sd=sigma*(1+observed)**0.3, observed=observed, shape=len(observed))

            #Observed = pm.Laplace("Observed", mu=signal+1e-7, b=10*sigma,observed=observed, shape=len(observed))
            # Observed = pm.Normal("Observed", mu=signal+1e-7, sd=sigma,
            #                     observed=observed, shape=len(observed))

        llk = pm.Deterministic("proba", model.logpt)
        # llk = pm.Deterministic("signal", signal)

        best_init = pm.find_MAP()
        print(best_init)
        # print(signal)
        if best_only:
            if sigmoid:
                return [], [best_init["weight"],best_init["weights"],best_init["weightc"]]
            else:
                return [], best_init["weight"]

        trace = pm.sample(4*n, tune=n, cores=1, chains=1)
        print(np.argmax(trace["proba"]))

    return trace, trace["weight"][np.argmax(trace["proba"])]


if __name__ == "__main__":
    from repli1d.expeData import whole_genome, replication_data
    from repli1d.analyse_RFD import smooth
    import numpy as np
    import argparse
    import pickle
    import theano
    import theano.tensor.nnet as nnet
    from scipy import optimize
    import pymc3 as pm

    parser = argparse.ArgumentParser()
    # parser.add_argument('--start', type=int, default=5000)
    # parser.add_argument('--end', type=int, default=120000)
    parser.add_argument('--resolution', type=int, default=5)
    # parser.add_argument('--ndiff', type=int, default=60)
    parser.add_argument('--cell', type=str, default="K562")
    parser.add_argument('--wig', action="store_true")
    parser.add_argument('--redo', action="store_true")
    parser.add_argument('--sigmoid', action="store_true")

    parser.add_argument('--signal', type=str, default="detected.peak")
    parser.add_argument('--visu', action="store_true")
    parser.add_argument('--error', action="store_true")
    parser.add_argument('--DNase', action="store_true")
    parser.add_argument('--GC', action="store_true")
    parser.add_argument('--Faire', action="store_true")
    parser.add_argument('--NFR', action="store_true")
    parser.add_argument('--Meth', action="store_true")
    parser.add_argument('--Meth450', action="store_true")

    parser.add_argument('--norm', action="store_true")


    parser.add_argument('--name-visu', dest="name", type=str, default="tmp.html")
    parser.add_argument('--name-weight', dest="namew", type=str, default="tmp.weight")
    parser.add_argument('--smooth', type=int, default=None)
    parser.add_argument('--maxi', type=int, default=None)
    parser.add_argument('--exclude', nargs='+', dest="exclude", type=str, default=[])
    parser.add_argument('--extra', action="store_true")
    parser.add_argument("--percent",type=float,default=None)
    parser.add_argument('--constant', action="store_true")
    parser.add_argument('--fromlower', action="store_true")
    parser.add_argument('--nosmoothdata',dest="smoothdata", action="store_false")
    parser.add_argument('--ms', nargs='+', dest="ms", type=int, default=[])
    parser.add_argument('--nooverfit', action="store_true")







    Exp = ['DNaseI', 'ORC2', 'H2az', 'H3k27ac', 'H3k27me3', 'H3k36me3', 'H3k4me1',
           'H3k4me2', 'H3k4me3', 'H3k79me2', 'H3k9ac', 'H3k9me1', 'H3k9me3', 'H4k20me1']  # , "Meth"]

    Exp = Exp[2:]


    args = parser.parse_args()

    for ex in args.exclude:
        Exp.remove(ex)

    wig = args.wig
    strain = args.cell
    resolution = args.resolution
    redo = args.redo

    if wig:
        Exp = [e+"wig" for e in Exp ]#+ Exp[-1:]

    if args.DNase:
        Exp += ["DNaseI"]
    if args.GC:
        Exp += ["GC"]
    if args.Faire:
        Exp += ["Faire"]
    if args.NFR:
        Exp += ["NFR"]
    if args.Meth:
        Exp += ["Meth"]
    if args.Meth450:
        Exp += ["Meth450"]

    if args.constant:
        Exp += ["Constant"]


    Signals = {}
    M = {}
    S = {}
    add =[]
    if args.extra:
        if args.cell == "K562":
            dire = "K562whole"
        if args.cell == "Gm12878":
            dire = "GM12878"
        if args.cell in ["HeLaS3","Hela","Helas3"]:
            dire = "Hela"
        add=glob.glob("/mnt/data/data/K562whole/all_chip/conservative/*.bed")
        add=glob.glob("/mnt/data/data/%s/all_chip/whole_signal/*.bed" % dire)

        print(len(add))

        add = add#[:200]
        Exp+=add

    if args.norm:
        smark = whole_genome(strain=strain, experiment="CNV",
                             resolution=resolution, raw=False, oData=False,
                             bp=True, bpc=False, filename=None, redo=redo,root="/home/jarbona/repli1D/")
        smark = np.concatenate(smark, axis=0)
        smark[smark==0] = 4
        smark[np.isnan(smark)] = 4
        CNV=smark



    for markn in Exp:
        print(markn)
        Mark = []
        if args.ms != []:
            for sm in args.ms:
                Mark.append([markn+"_%i"%sm,sm])
        else:
            Mark=[[markn, args.smooth]]
        print(Mark)
        for mark,sm in Mark:
            smark = whole_genome(strain=strain, experiment=markn,
                                     resolution=resolution, raw=False, oData=False,
                                     bp=True, bpc=False, filename=None, redo=redo)

            # print(smark)
            if args.nooverfit:
                smark = np.concatenate(smark[1::2], axis=0)
            else:
                smark = np.concatenate(smark, axis=0)


            if args.norm:
                smark /= CNV

            if mark == "Constant":
                Signals[mark], M[mark], S[mark] = smark,1,1
            else:

                Signals[mark], M[mark], S[mark] = norm2(smark,cut=15)#, cutp=99.5)
            # print(Signals[mark].dtype)
            Signals[mark][np.isnan(Signals[mark])] = 0

            # Remove strange peak in the data

            #m = Signals[Exp[0]].argmax()
            #print(Signals[mark][m])

            #Signals[mark][m] = 0

            if sm:  # and mark != "H3k36me3wig":
                Signals[mark] = smooth(Signals[mark], sm)[:args.maxi]

            if args.percent != None:
                maxi = int(len(Signals[mark])*args.percent)
                Signals[mark] = Signals[mark][:maxi]



        #if mark == "H3k36me3":
    #        Signals[mark] = np.zeros_like(Signals[mark])

    print(Signals.keys())
    with open(args.signal, "rb") as f:
        data = pickle.load(f)

    if args.fromlower:
        smark = whole_genome(strain=strain, experiment=mark,
                                 resolution=resolution, raw=False, oData=False,
                                 bp=True, bpc=False, filename=None, redo=redo)
        Data = []
        for low,high in zip(data,smark):
            Data.append(mapboth(low,high,f=5,pad=True))
        data = Data

    if args.nooverfit:
        Data = np.concatenate(data[1::2], axis=0)
    else:
        Data = np.concatenate(data, axis=0)
    print(len(Data)*1000*resolution)
    Data, mean, std = norm2(Data)
    Data[np.isnan(Data)] = 0
    Data = Data[:args.maxi]

    if args.percent != None:
        maxi = int(len(Data)*args.percent)
        Data = Data[:maxi]

    Exp = list(Signals.keys())
    k0 = Exp[0]
    print("Sizes",len(Data),len(Signals[k0]))
    assert(len(Data) == len(Signals[k0]))

    if args.smoothdata and args.smooth:
        Data = smooth(Data, args.smooth)
    """
    indices = np.random.permutation(len(Data))
    delta = len(indices)//5
    Bestw = []
    for sp in range(0, len(indices), delta):
        if sp+delta> len(indices):
            continue
        print(sp,sp+delta,len(indices))
        training_idx = indices[sp:sp+delta]
        trace, bestw = fit_observed_value(observed=Data[training_idx], Exp=Exp, signals={name:d[training_idx] for name,d in Signals.items()}, best_only=True,sigmoid=args.sigmoid)
        Bestw.append(bestw)
        with open(args.namew+"_bs", "wb") as f:
            pickle.dump(Bestw, f)
    """
    trace, bestw = fit_observed_value(observed=Data, Exp=Exp, signals=Signals, best_only=True,sigmoid=args.sigmoid)

    for exp, w in zip(Exp, bestw):
        print(exp, w)

    with open(args.namew, "wb") as f:
        pickle.dump([M, S, bestw, Exp,args.norm], f)

    x, H3k36w = replication_data(strain, "H3k36me3wig",
                                 chromosome=1, start=0,
                                 end=50000,
                                 resolution=resolution)
    x, H3k36 = replication_data(strain, "H3k36me3",
                                chromosome=1, start=0,
                                end=50000,
                                resolution=resolution)
    if args.visu:
        maxi = 10000
        x = np.arange(maxi)*resolution
        if not args.sigmoid:
            comp = np.array([bestw[i] * Signals[iexp] for i, iexp in enumerate(Exp)])
        else:
            comp = [bestw[0][i]*(-1+2/(1+np.exp(-bestw[1][i]*(Signals[iexp]-bestw[2][i])))) for i, iexp in enumerate(Exp)]

        plotly_blocks(x, [[[Data[:maxi],"Detected peaks"], [np.sum(comp, axis=0),"Fitted signal"]],
                          [[H3k36, "H3k36me3"]],
                          [[H3k36w]],
                          [[replication_data(
                              strain, args.namew, chromosome=1, resolution=resolution, start=0, end=10000*5)[1],"From file"]]], name=args.name, default="lines")
    if args.error:

        if not args.sigmoid:
            comp = np.array([bestw[i] * Signals[iexp] for i, iexp in enumerate(Exp)])
        else:
            comp = [bestw[0][i]*(-1+2/(1+np.exp(-bestw[1][i]*(Signals[iexp]-bestw[2][i])))) for i, iexp in enumerate(Exp)]
        comp1 = np.sum(comp, axis=0)
        comp1[comp1 < 0] = 0
        comp = Data - comp1
        sub = 10
        print(len(comp))
        plotly_blocks(x, [[[smooth(comp, sub)[::sub]]],
                          [[smooth(comp1, sub)[::sub]], [smooth(Data, sub)[::sub], "Peaks"]]
                          ], name="error.html", default="lines")

        plotly_blocks(x, [[[smooth(bestw[i] * Signals[iexp], sub)[::sub], iexp]] for i, iexp in enumerate(Exp)],
                      name="error_whole.html", default="lines")
