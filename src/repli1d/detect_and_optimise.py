# detect peak


from repli1d.analyse_RFD import detect_peaks, compare, smooth
from repli1d.fast_sim import get_fast_MRT_RFDs
from repli1d.expeData import replication_data
from repli1d.single_mol_analysis import compute_info,compute_real_inter_ori
from repli1d.pso import PSO
import sys
import argparse
from repli1d.visu_browser import plotly_blocks
import pylab
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=5000)
parser.add_argument('--end', type=int, default=120000)
parser.add_argument('--ch', type=int, default=1)
parser.add_argument('--resolution', type=int, default=5)
parser.add_argument('--ndiff', type=int, default=60)
parser.add_argument('--percentile', type=int, default=82)
parser.add_argument('--cell', type=str, default="K562")
parser.add_argument('--visu', action="store_true")
parser.add_argument('--name', type=str, default="tmp.html")
parser.add_argument('--nsim', type=int, default=500)
parser.add_argument('--signal', type=str, default="peak")
parser.add_argument('--input', action="store_true")
parser.add_argument('--correct', action="store_true")
parser.add_argument('--continuous', action="store_true")
parser.add_argument('--noise', type=float, default=.1)
parser.add_argument('--fspeed', type=float, default=.3)
parser.add_argument('--RFDo', action="store_true")



args = parser.parse_args()

start = args.start
end = args.end
ch = args.ch
cell = args.cell
resolution_polarity = args.resolution
resolution = args.resolution
exp_factor = 4
percentile = args.percentile
fork_speed = args.fspeed
kon = 0.005
ndiff = args.ndiff
nsim = args.nsim

if args.signal == "peak":
    x, d3p = detect_peaks(start, end, ch,
                          resolution_polarity=resolution_polarity,
                          exp_factor=exp_factor,
                          percentile=percentile, cell=cell,nanpolate=True)

    if args.correct:
        x, DNaseI = replication_data(cell, "DNaseI", chromosome=ch,
                                     start=start, end=end,
                                     resolution=resolution, raw=False)
        x, CNV = replication_data(cell, "CNV", chromosome=ch,
                                  start=start, end=end,
                                  resolution=resolution, raw=False)
        CNV[CNV == 0] = 2
        DNaseI[np.isnan(DNaseI)] = 0
        DNaseI /= CNV

        DNaseIsm = smooth(DNaseI, 100)
        DNaseIsm /= np.mean(DNaseIsm)

        d3p *= DNaseIsm

        d3p[np.isnan(d3p)] = 0
        d3p[np.isinf(d3p)] = 0
        print(np.sum(np.isnan(d3p)))
        # pylab.plot(d3p)
        # pylab.show()

else:
    x, d3p = replication_data(cell, args.signal, chromosome=ch,
                              start=start, end=end,
                              resolution=resolution, raw=False)

    if args.correct:

        x, CNV = replication_data(cell, "CNV", chromosome=ch,
                                  start=start, end=end,
                                  resolution=resolution, raw=False)
        CNV[CNV == 0] = 2
        d3p /= CNV
    d3p[np.isnan(d3p)] = 0

    # simulate


def run(profile):
        MRTp, MRTs, RFDs, Rept_time, single_mol_exp, pos_time_activated_ori, It = get_fast_MRT_RFDs(
            nsim, profile, ndiff, kon=kon,
            fork_speed=fork_speed, dori=20*5/resolution,single_mol_exp=False,continuous=args.continuous)
        #print("check", np.sum(d3p), np.sum(np.ones_like(d3p)*np.sum(d3p)*0.1/len(d3p)))


        # Compare to exp data
        """
        MRTpearson, MRTstd, MRT = compare(
            MRTp[::10//resolution], "MRT", cell, res=10, ch=ch, start=start, end=end, return_exp=True)
        RFDpearson, RFDstd, RFD = compare(RFDs, "OKSeq", cell, res=resolution, ch=ch,
                                          start=start, end=end, return_exp=True, rescale=1/resolution)

        """
        if cell != "Cerevisae":
            MRTpearson, MRTstd, MRT = compare(
                MRTp[::10//resolution], "MRT", cell, res=10, ch=ch, start=start, end=end, return_exp=True)
            RFDpearson, RFDstd, RFD = compare(RFDs, "OKSeq", cell, res=resolution, ch=ch,
                                              start=start, end=end, return_exp=True, rescale=1/resolution)
        else:
            MRTpearson, MRTstd, MRT = compare(
                MRTp, "MRT", cell, res=1, ch=ch, start=start, end=end, return_exp=True)
            RFDpearson, RFDstd, RFD = compare(RFDs, "OKSeq", cell, res=1, ch=ch,
                                              start=start, end=end, return_exp=True, rescale=1,nanpolate=True,smoothf=2)

        if args.RFDo:
            #print(RFDstd)
            return RFDstd,MRTp,MRT,RFD,RFDs
        else:
            return MRTstd+RFDstd,MRTp,MRT,RFD,RFDs

def evaluate(profile):
    return run(profile)[0]

x0 = d3p#+np.ones_like(d3p)*np.sum(d3p)*args.noise/len(d3p)

x0 /= np.sum(x0)
init = x0.copy()

initerror,MRTpi,MRTi,RFDi,RFDsi = run(x0)
import pandas as pd
for i in range(20):
    vs = 2*np.median(x0[x0 != 0])
    #print(100/len(x0),)
    opti = PSO(evaluate,bounds=[0,1],x0=x0,num_particles=15,maxiter=20,velocity_scale=vs,normed=True)

    end_error,MRTp,MRT,RFD,RFDs = run(opti.best)
    print("End error",end_error)
    if cell != "Cerevisae":
        ToSee = [[[1-MRTpi, "MRTi"],[1-MRTp, "MRT"], [x[::10//resolution], 1 - MRT[:len(x[::10//resolution])], "MRTexp"]],
                [[RFD, "RFDExp"], [RFDsi, "RFDsimi_init"],[RFDs, "RFDsim_end"]]]
    else:
        ToSee = [[[1-MRTpi, "MRTi"],[1-MRTp, "MRT"], [x, 1 - MRT, "MRTexp"]],
                [[RFD, "RFDExp"], [RFDsi, "RFDsimi_init"],[RFDs, "RFDsim_end"]]]
    ToSee += [[[init,"init"],[opti.best,"Optimised"]]]

    x0 = opti.best
    print("Tot",np.sum(x0))
    #x0 /= np.sum(x0)



    plotly_blocks(x,ToSee, name=args.name+"_%i" % i, default="lines")
    pd.DataFrame({"signalValue":x0,"chromStart":x*1000,"chromEnd":x*1000,"chrom":["chr%i"%ch]*len(x)}).to_csv(args.name+"optimised_signal.csv",index=False,sep="\t")

print(initerror,end_error)
