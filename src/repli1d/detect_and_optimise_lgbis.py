# detect peak


from repli1d.analyse_RFD import detect_peaks, compare, smooth
from repli1d.fast_sim import get_fast_MRT_RFDs
from repli1d.expeData import replication_data
from repli1d.single_mol_analysis import compute_info,compute_real_inter_ori
import pandas as pd
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
parser.add_argument('--nsim', type=int, default=600)
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


rstart = int(start/resolution)

def run(profile,tndiff,tstart,tend,actualProfile=None):



        res = args.resolution
        if actualProfile is not None:

            tmpProfile = actualProfile.copy()
            #print(tmpProfile.shape,np.array(profile).shape,start,tstart,tend,tmpProfile[tstart-start:tend-start].shape,tstart-start,tend-start)
            weight_t = np.sum(tmpProfile[tstart-start:tend-start])
            tmpProfile[tstart-start:tend-start] = weight_t * np.array(profile) / np.sum(profile)
        else:
            tmpProfile = profile
        MRTp, MRTs, RFDs, Rept_time, single_mol_exp, pos_time_activated_ori, It,wRFD = get_fast_MRT_RFDs(
            nsim, tmpProfile, tndiff, kon=kon,
            fork_speed=fork_speed, dori=20*5/resolution,single_mol_exp=False,continuous=args.continuous,wholeRFD=True)
        #print("check", np.sum(d3p), np.sum(np.ones_like(d3p)*np.sum(d3p)*0.1/len(d3p)))

        #print("Start, end",tstart,tend,tndiff)
        # Compare to exp data
        """
        MRTpearson, MRTstd, MRT = compare(
            MRTp[::10//resolution], "MRT", cell, res=10, ch=ch, start=start, end=end, return_exp=True)
        RFDpearson, RFDstd, RFD = compare(RFDs, "OKSeq", cell, res=resolution, ch=ch,
                                          start=start, end=end, return_exp=True, rescale=1/resolution)

        """



        if cell != "Cerevisae":
            if actualProfile is not None:
                MRTpearsonl, MRTstdl, MRTl = compare(
                    MRTp[int((tstart-start)/res):int((tend-start)/res)][::10//resolution], "MRT", cell, res=10, ch=ch, start=tstart, end=tend, return_exp=True)
                RFDpearsonl, RFDstdl, RFDl = compare(RFDs[int((tstart-start)/res):int((tend-start)/res)], "OKSeq", cell, res=resolution, ch=ch,
                                                  start=tstart, end=tend, return_exp=True, rescale=1/resolution)
            else:
                RFDstdl = 0

            MRTpearson, MRTstd, MRT = compare(
                MRTp[::10//resolution], "MRT", cell, res=10, ch=ch, start=start, end=end, return_exp=True)
            RFDpearson, RFDstd, RFD = compare(RFDs, "OKSeq", cell, res=resolution, ch=ch,
                                              start=start, end=end, return_exp=True, rescale=1/resolution)
        else:
            if actualProfile is not None:
                MRTpearsonl, MRTstdl, MRTl = compare(
                    MRTp[int((tstart-start)/res):int((tend-start)/res)], "MRT", cell, res=1, ch=ch, start=tstart, end=tend, return_exp=True)
                RFDpearsonl, RFDstdl, RFDl = compare(RFDs[int((tstart-start)/res):int((tend-start)/res)], "OKSeq", cell, res=1, ch=ch,
                                                  start=tstart, end=tend, return_exp=True, rescale=1,nanpolate=True,smoothf=2)
            else:
                RFDstdl = 0

            MRTpearson, MRTstd, MRT = compare(
                MRTp, "MRT", cell, res=1, ch=ch, start=start, end=end, return_exp=True)
            RFDpearson, RFDstd, RFD = compare(RFDs, "OKSeq", cell, res=1, ch=ch,
                                              start=start, end=end, return_exp=True, rescale=1,nanpolate=True,smoothf=2)

            RFDs = smooth(RFDs,2)

        if args.RFDo:
            print(RFDstd,RFDstdl)
            return RFDstd+RFDstdl,MRTp,MRT,RFD,RFDs,wRFD
        else:
            return MRTstd+RFDstd,MRTp,MRT,RFD,RFDs,wRFD



x0 = d3p#+np.ones_like(d3p)*np.sum(d3p)*args.noise/len(d3p)

x0 /= np.sum(x0)
init = x0.copy()

initerrorg,gMRTpi,gMRTi,gRFDi,gRFDsi,wRFDi = run(x0,ndiff,start,end)
initerrorg,gMRTp,gMRT,gRFD,gRFDs,wRFD = run(x0,ndiff,start,end)

print("Initial Global error",initerrorg)

for i in range(100):
    # How to select ndiff

    print("Local update")




    tndiff = 6
    fraction = (end-start) * tndiff / ndiff / resolution
    fraction = min(fraction,end-start)

    exclude = 200

    delta = smooth(np.abs(gRFDs-gRFD)[exclude:-exclude],int(fraction))

    middle = np.argmax(delta) + exclude
    tstart = max(0,middle-fraction/2)
    tstart = min(end-start-fraction,tstart)

    #tstart = int(start + (end-start - (end-start) * tndiff / ndiff ) * np.random.rand())
    #tstart = max(0,tstart)
    tend = int(tstart + fraction)
    tstart = int(10*round(tstart/10))
    tend = int(10*round(tend/10))

    #EStimate correction:
    def correct(wRFD,left=True):
        c = wRFD.copy()
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                if left:
                    if c[i,j] == -1:
                        c[i,j:]=0
                        break
                else:
                    if c[i,-j-1] == 1:
                        c[i,:-j-1]=0
                        break
        if not left:
            c[c==1] = 0
        return c

    res = 1
    left = correct(wRFD[::,int((tstart-start)/res):int((tend-start)/res)])
    right = correct(wRFD[::,int((tstart-start)/res):int((tend-start)/res)],left=False)




    def evaluate(profile,score=True):
        res = run(profile,ndiff,tstart,tend,actualProfile=x0)
        if score:
            return res[0]
        else:
            return res
    x0t = x0[tstart-start:tend-start].copy()
    x0t /= np.sum(x0t)
    initt = x0t.copy()

    print(start,tstart,tend,end)
    initerror,MRTpi,MRTi,RFDi,RFDsi,wRFDt = evaluate(x0t,score=False)

    print("Initial local error",initerror)
    vs =10*np.median(x0[x0 != 0])
    vs = 100/len(x0t)
    opti = PSO(evaluate,bounds=[0,1],x0=x0t,num_particles=15,maxiter=20,velocity_scale=vs)

    #end_error,MRTp,MRT,RFD,RFDs,wRFDt = evaluate(opti.best,score=False)

    #print("End error",end_error)
    """
    if cell != "Cerevisae":
        ToSee = [[[1-MRTpi, "MRTi"],[1-MRTp, "MRT"], [x[::10//resolution], 1 - MRT[:len(x[::10//resolution])], "MRTexp"]],
                [[RFD, "RFDExp"], [RFDsi, "RFDsimi_init"],[RFDs, "RFDsim_end"]]]
    else:
        ToSee = [[[1-MRTpi, "MRTi"],[1-MRTp, "MRT"], [x[tstart:tend], 1 - MRT, "MRTexp"]],
                [[RFD, "RFDExp"], [RFDsi, "RFDsimi_init"],[RFDs, "RFDsim_end"],[RFDs+np.mean(left+right,axis=0),"Corrected2"],[gRFDsi[int((tstart-start)/res):int((tend-start)/res)]]]]
    ToSee += [[[initt,"init"],[opti.best,"Optimised"]],[[np.mean(left,axis=0)],[np.mean(right,axis=0)]]]
    plotly_blocks(x[tstart:tend],ToSee, name=args.name+"local_%i" % i, default="lines")
    """
    #Print reconneect:


    weight_t  = np.sum(x0[tstart-start:tend-start])
    x0[tstart-start:tend-start] = weight_t * np.array(opti.best) / np.sum(opti.best)
    end_error,gMRTp,gMRT,gRFD,gRFDs,wRFD = run(x0,ndiff,start,end)
    print("New Global error",initerror)


    if cell != "Cerevisae":
        ToSee = [[[1-gMRTpi, "MRTi"],[1-gMRTp, "MRT"], [x[::10//resolution], 1 - gMRT[:len(x[::10//resolution])], "MRTexp"]],
                [[gRFD, "RFDExp"], [gRFDsi, "RFDsimi_init"],[gRFDs, "RFDsim_end"]]]
    else:
        ToSee = [[[1-gMRTpi, "MRTi"],[1-gMRTp, "MRT"], [x, 1 - gMRT, "MRTexp"]],
                [[gRFD, "RFDExp"], [gRFDsi, "RFDsimi_init"],[gRFDs, "RFDsim_end"]]]
    ToSee += [[[init,"init"],[x0,"Optimised"]],[[smooth(np.abs(gRFD-gRFDs),250),"dsm"]]]
    plotly_blocks(x,ToSee, name=args.name+"global_%i" % i, default="lines")

    pd.DataFrame({"signalValue":x0,"chromStart":x*1000,"chromEnd":x*1000,"chrom":["chr%i"%ch]*len(x)}).to_csv(args.name+"optimised_signal.csv",index=False,sep="\t")
print(initerrorg,end_error)
