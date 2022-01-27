from __future__ import division, print_function
from repli1d.expeData import replication_data
import pandas as pd
import numpy as np

from scipy import stats


# %load ./../functions/detect_peaks.py
"""Detect peaks in data based on their amplitude and other features."""


import numpy as np

__author__ = "Marcos Duarte, https://github.com/demotu/BMC"
__version__ = "1.0.5"
__license__ = "MIT"


def detect_peaks_base(x, mph=None, mpd=1, threshold=0, edge='rising',
                      kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height (if parameter
        `valley` is False) or peaks that are smaller than maximum peak height
         (if parameter `valley` is True).
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=-1.2, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)

    Version history
    ---------------
    '1.0.5':
        The sign of `mph` is inverted if parameter `valley` is True

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
            if mph is not None:
                mph = -mph
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()


def cut_larger_than(mask, size=2):
    deltas = np.logical_xor(mask[1:], mask[:-1])
    chunks = np.where(deltas)[0] + 1
    chunks = chunks.tolist()
    # print(chunks)

    if chunks[-1] != len(mask):
        chunks.append(len(mask))
    boolv = [mask[0]]
    for v in chunks[1:]:
        boolv.append(~boolv[-1])

    chunks.insert(0, 0)
    chunks = np.array(chunks)
    sizes = chunks[1:] - chunks[:-1]

    start = 0
    end = 0

    # Merge smaller than size
    # First swich from false to True
    for i, (v, s) in enumerate(zip(boolv, sizes)):
        if s <= size and not v:
            boolv[i] = True

    #print(deltas)
    #print(chunks)
    #print(boolv)
    #print(sizes)

    start = 0
    end = 0
    segments = {"start": [], "end": [], "keep": []}

    for iseg,(v, s) in enumerate(zip(boolv, sizes)):
        #if s == 0:
        if s > size and not v:

            # Add previous segment
            if iseg != 0:

                segments["start"].append(start)
                segments["end"].append(end)
                segments["keep"].append(True)

            start = end
            end = start + s

            segments["start"].append(start)
            segments["end"].append(end)
            segments["keep"].append(False)

            start = end
            end = start
        else:
            end += s
    if v:
        segments["start"].append(start)
        segments["end"].append(end)
        segments["keep"].append(True)

    return segments

def propagate_false(a):
    wn = np.where(~a)[0]
    if len(wn) == 0:
        return a
    # print(wn)
    end = None
    if wn[-1] == len(a)-1:
        end = -1
    a[wn[:end]+1] = False
    start = 0
    if wn[0] == 0:
        start = 1
    a[wn[start:]-1] = False
    return a
    #a[wn+1] = False


def propagate_n_false(a, n):
    for i in range(n):
        a = propagate_false(a)
    return a


def compare(simu, signal, cell, res, ch, start, end, trim=0.05, return_exp=False, rescale=1, nanpolate=False,
            smoothf=None, trunc=False, pad=False, return_mask=False,masking=True,propagateNan=True):
    x, exp_signal = replication_data(cell, signal, chromosome=ch,
                                     start=start, end=end,
                                     resolution=res, raw=False, pad=pad)

    print(len(exp_signal),len(simu),cell)
    exp_signal *= rescale

    l = None
    if trunc and len(simu) != len(exp_signal):
        print("Truncating", len(simu), len(exp_signal))
        l = min(len(simu), len(exp_signal))
        simu = simu[:l]
        exp_signal = exp_signal[:l]

    mask_exp = np.array([not np.isnan(e) for e in exp_signal])
    if masking:
        maskl = masking  # kb

        if propagateNan:
            mask_exp = propagate_n_false(mask_exp, int(maskl/res))

        exclude = int(maskl/res)
        mask_exp[:exclude] = False
        mask_exp[-exclude:] = False


    #Due to masking
    mask_exp[np.isnan(simu)] = False

    if smoothf is not None:
        exp_signal = nan_polate(exp_signal)
        exp_signal = smooth(exp_signal, smoothf)
    if simu is not None:
        ret = [stats.pearsonr(simu[mask_exp], exp_signal[mask_exp]),
               np.mean((simu[mask_exp] - exp_signal[mask_exp])**2)**0.5]
    else:
        ret = [None,None]
    if return_exp:
        ret.append(exp_signal)
    if return_mask:
        ret.append([mask_exp, l])
    return ret


def nan_polate(A):
    ok = ~np.isnan(A)
    xp = ok.ravel().nonzero()[0]
    fp = A[~np.isnan(A)]
    x = np.isnan(A).ravel().nonzero()[0]

    A[np.isnan(A)] = np.interp(x, xp, fp)
    return A


def mapboth(low, high, f, pad=False):
    cp = np.zeros_like(high)
    if not pad:
        if len(low) * f < len(high):
            print("%i must be >= %i" % (len(low) * f, len(high)))
            print("You can use the pad option")
            raise

    else:
        np.add(cp, np.nan, out=cp, casting="unsafe")
        #cp += np.nan
    for i in range(f):
        w = len(cp[i::f])
        cp[i::f][:len(low)] = low[:w]
    return cp


def smooth(ser, sc):
    return np.array(pd.Series(ser).rolling(sc, min_periods=1, center=True).mean())


def get_codire(signal, cell, ch, start, end, resolution, min_expre=0):
    try:
        Xg, Yg, xmg, ymg, direction = get_expression(
            cell, ch, start, end, resolution, min_expre=min_expre)
    except:
        return np.nan
    fil = (direction != 0)

    s = min(len(signal), len(direction))
    return np.nanmean(signal[:s][fil[:s]]*direction[:s][fil[:s]])


def get_expression(cell, ch, start, end, resolution, min_expre=0):
    re = replication_data(cell, "ExpGenes", chromosome=ch,
                          start=start, end=end, resolution=resolution, raw=True)

    X = []
    Y = []
    D = []
    xm = []
    ym = []
    Ym = []
    Xm = []
    #std = np.nanstd(re["signalValue"])
    for istart, iend, v, strand in zip(re["chromStart"], re["chromEnd"], re["signalValue"], re["strand"]):

        # print(istart*5,strand)
        if strand == "+":
            X.extend([istart, iend, iend + 1])
            Y.extend([v, v, np.nan])
        else:
            Xm.extend([istart, iend, iend + 1])
            Ym.extend([v, v, np.nan])
        xm.append(istart / 2 + iend / 2)
        ym.append(v)
        D.append(strand)

    mean = np.nanmean(Y)
    stdv = np.nanstd(Y)
    print(mean, stdv)
    Y = np.array(Y)
    X = np.array(X)
    Ym = np.array(Ym)
    Xm = np.array(Xm)
    D = np.array(D)
    xm = np.array(xm)
    ym = np.array(ym)

    directionp = np.arange(start, end, resolution)*0
    for istart, iend, v in zip(X[::3], X[1::3], Y[::3]):
        if v > min_expre:
            # print(start,istart,iend,int(round((istart-start)/resolution)),int(round((iend-start)/resolution)))
            directionp[int(round(istart-start/resolution)):int(round(iend-start/resolution))] = 1
    directionm = np.arange(start, end, resolution)*0
    for istart, iend, v in zip(Xm[::3], Xm[1::3], Ym[::3]):
        if v > min_expre:
            directionm[int(round(istart-start/resolution)):int(round(iend-start/resolution))] = 1

    return X*resolution, Y, Xm*resolution, Ym, directionp-directionm


def sm(ser, sc): return np.array(pd.Series(ser).rolling(
    sc, min_periods=sc, center=True).mean())


def detect_peaks(start, end, ch, resolution_polarity=5, exp_factor=6, percentile=85, cell="K562",
                 cellMRT=None, cellRFD=None, nanpolate=False, fsmooth=None, gsmooth=5,
                 recomp=False, dec=None, fich_name=None, sim=True,expRFD="OKSeq",
                 rfd_only=False,exp4=False,oli=False,peak_mrt=False):

    rpol = resolution_polarity
    if exp4:
        exp_factor=6
    if fich_name is None:
        print("Loading here")

        if cellMRT is None:
            cellMRT = cell
        if cellRFD is None:
            cellRFD = cell
        print(start, end, cellRFD, ch, rpol)


        if "Yeast" in cellMRT:
            resolution = 1
        elif cell in ["K562","Hela","GM","HeLa","HeLaS3","Gm12878"]:

            resolution = 10
        else:
            resolution=resolution_polarity

        #print(cell)
        if (not rfd_only) or exp4:
            x_mrt, mrt_exp = replication_data(cellMRT, "MRT", chromosome=ch,
                                              start=start, end=end, resolution=resolution, raw=False)



        # Loading RFD

        if not peak_mrt:
            x_pol, pol_exp = replication_data(cellRFD, expRFD, chromosome=ch,
                                              start=start, end=end, resolution=rpol, raw=False, pad=True)
            if nanpolate:
                pol_exp = nan_polate(pol_exp)
            #print(pol_exp[:10])
            if fsmooth != None:
                print("Smoothing")
                pol_exp = smooth(pol_exp, fsmooth)
                #mrt_exp = np.array(pd.Series(np.cumsum(pol_expc)).rolling(10000, min_periods=1, center=True).apply(lambda x: np.mean(x<x[len(x)//2])))[::2]


        else:
            if resolution == rpol:
                smrt = smooth(mrt_exp,5)
                pol_exp = np.concatenate([[0],smrt[1:]-smrt[:-1]])
                x_pol=x_mrt
        Smpol = np.copy(pol_exp)
        #exit()
        #print(pol_exp[:10])
        #exit()
        ratio_res = resolution // rpol




        #print(mrt_exp.shape[0]*2, pol_exp.shape, ratio_res,)
        if not rfd_only:
            nmrt = mapboth(mrt_exp, pol_exp, ratio_res, pad=True)
    else:
        print("Here datafile")
        strain = pd.read_csv(fich_name, sep=",")
        #resolution = 5
        x_pol = strain.chromStart
        if sim:
            pol_exp = strain.RFDs
            mrt_exp = strain.MRTs
        else:
            pol_exp = strain.RFDe
            mrt_exp = strain.MRTe
        nmrt = mrt_exp
        if fsmooth != None:
            #print("smothing")
            pol_exp = smooth(pol_exp, fsmooth)
        Smpol = np.copy(pol_exp)
        ratio_res = 1
        #exit()
    #print(fich_name)
    #exit()

    if not rfd_only:
        """
        for delta in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8][::-1]:

            c1 = nmrt > delta
            Smpol[c1] = np.array(sm(Smpol, gsmooth))[c1]"""

        Smpol = sm(Smpol, 5)

    else:
        Smpol = sm(Smpol, 10)

    delta = Smpol[1:] - Smpol[:-1]
    delta -= np.nanmin(delta)
    print(delta[:10])
    percentile = np.percentile(delta[~np.isnan(delta)], percentile)
    print("Threshold value", percentile)
    delta[delta < percentile] = 0.0

    if recomp:
        pol_exp = smooth(pol_exp, 2)
        deltap = pol_exp[1:] - pol_exp[:-1]
        deltap -= np.nanmin(delta)
        deltap[delta <= 0] = 0
        #deltap[deltap < percentile] = 0
        delta = deltap
        delta[delta < 0] = 0

    if dec != None:
        if dec != 2:
            raise
        else:
            for i, (ok0, ok1, ok2) in enumerate(zip(pol_exp, pol_exp[1:], pol_exp[2:])):

                if ok0 + 0.05 > ok2:
                    delta[i] = 0  # shifted from one on purpose
                    delta[i+1] = 0
    if (not rfd_only) or exp4:
        #
        if oli:
            #delta = -np.log(1-delta/2)/ mapboth(mrt_exp, delta, ratio_res, pad=True)
            delta = delta/ (mapboth(mrt_exp, delta, ratio_res, pad=True)+0.05)

        else:
            delta *= mapboth(np.exp(-exp_factor * mrt_exp), delta, ratio_res, pad=True)
        print(exp_factor,mrt_exp[:15])
        print(len(delta),len(mrt_exp))
        print("here dela")


    delta[np.isnan(delta)] = 0

    return x_pol, np.concatenate(([0], delta))
