# detect peak


from repli1d.analyse_RFD import compare
from repli1d.fast_sim import get_fast_MRT_RFDs
from repli1d.expeData import replication_data
import argparse
import collections
import numpy as np
import os
import pickle
import pandas as pd
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=5000)
parser.add_argument('--end', type=int, default=120000)
parser.add_argument('--ch', type=int, default=1)
parser.add_argument('--dori', type=int, default=30)
parser.add_argument('--cell', type=str, default="K562")
parser.add_argument('--root', type=str, default="./")
parser.add_argument('--redo', action="store_true")
parser.add_argument('--name', type=str, default="tmp.html")
parser.add_argument('--nsim', type=int, default=200)
parser.add_argument('--signal', type=str, default="peak")
parser.add_argument('--correct', action="store_true")
parser.add_argument('--wig', action="store_true")
parser.add_argument('--sigmoid', action="store_true")
parser.add_argument('--noise', type=float, default=.1)
parser.add_argument('--compMRT', type=str, default=None)
parser.add_argument('--compRFD', type=str, default=None)
parser.add_argument('--ndiff', type=int, default=60)




args = parser.parse_args()

start = args.start
end = args.end
ch = args.ch
cell = args.cell
resolution_polarity = 5
exp_factor = 4
percentile = 82
fork_speed = 0.3
kon = 0.005
nsim = args.nsim
root = args.root
ndiff = args.ndiff
noise = args.noise
dori = args.dori

compMRT = cell
compRFD = cell
if args.compMRT is not None:
    compMRT = args.compMRT
if args.compRFD is not None:
    compRFD = args.compRFD

if not os.path.exists(root):
    os.makedirs(root)


def fl(name):
    def format_value(val):
        # print(type(val))
        if type(val) in [float, np.float64]:
            return "%.2e" % val
        else:
            val = str(val)
            val = val.replace("/","")
            return val
    if type(name) == collections.OrderedDict:
        return "".join(["%s-%s" % (p, format_value(fl(value))) for p, value in name.items()])
    else:
        return name


M = []
D = []
R = []
Do = []
P = []
SM = []
MRT_pearson = []
RFD_std = []

RFD_pearson = []
MRT_std = []
RepT = []

percentile = [90,80,70,60,50]

for p in percentile:
    signal = "%s/%s_p%i.peak" % (root, cell, p)
    command = "python src/repli1d/detect.py --resolution 5 --cell %s --percentile %i --name %s --cellMRT %s --cellRFD %s" % (cell,p,signal,compMRT,compRFD)
    print(command)
    if not os.path.exists(signal) or args.redo:
        return_code = subprocess.call(command, shell=True)


    for smooth in [1,5,10,15,20,30,40,50,60,80,100,120]:
        for wig in [True,False]:

            weightname = "%s/%s_p%i_sm%i_wig_%s.weight" % (root, cell, p, smooth, str(wig))
            if wig:
                extra = " --wig"
            else:
                extra = ""
            if args.sigmoid:
                extra += " --sigmoid"

            command = "python src/repli1d/retrieve_marks.py --cell %s --smooth %i  --exclude H3k9me1 H4k20me1 --signal %s --name-weight %s %s" %(cell,
                                                                                                                                                smooth,
                                                                                                                                                signal,
                                                                                                                                                weightname,extra)

            print(command)
            if not os.path.exists(weightname) or args.redo:
                return_code = subprocess.call(command, shell=True)

            mark = weightname
            random_activation = noise

            od = collections.OrderedDict([["mark", mark], ["ndiff", ndiff],
                                        ["random_activation", random_activation],
                                        [ "dori", dori]])
            filename = os.path.join(args.root, fl(od))
            filename += ".pick"
            print(filename)
            if os.path.exists(filename) and not args.redo:

                print("Found")
                with open(filename, "rb") as f:
                    sim = pickle.load(f)
            else:


                x, d3p = replication_data(cell, mark, chromosome=ch,
                                          start=start, end=end,
                                          resolution=5, raw=False, oData=False,
                                          bp=True, bpc=False)

                if args.correct:

                    x, CNV = replication_data(cell, "CNV", chromosome=ch,
                                              start=start, end=end,
                                              resolution=5, raw=False)
                    CNV[CNV == 0] = 2
                    d3p /= CNV
                d3p[np.isnan(d3p)] = 0

                print("Simu")
                sim = get_fast_MRT_RFDs(
                    nsim, d3p+np.ones_like(d3p)*np.sum(d3p)*random_activation/len(d3p), ndiff, kon=kon,
                    fork_speed=fork_speed, dori=dori)
                with open(filename, "wb") as f:
                    pickle.dump(sim, f)

            MRTp, MRTs, RFDs, Rep_Time, single_mol_exp, pos_time_activated_ori, It = sim
            # Compare to exp data
            MRTpearson, MRTstd, MRT = compare(
                MRTp[::2], "MRT", compMRT, res=10, ch=ch, start=start, end=end, return_exp=True)
            RFDpearson, RFDstd, RFD = compare(RFDs, "OKSeq", compRFD, res=5, ch=ch,
                                              start=start, end=end, return_exp=True, rescale=1/5)

            print(MRTpearson, MRTstd, RFDpearson, RFDstd)

            M.append(mark)
            D.append(ndiff)
            R.append(random_activation)
            Do.append(dori)
            P.append(p)
            SM.append(smooth)
            MRT_std.append(MRTstd)
            RFD_std.append(RFDstd)
            MRT_pearson.append(MRTpearson)
            RFD_pearson.append(RFDpearson)
            RepT.append(np.median(Rep_Time))

            Data = pd.DataFrame({"marks": M, "Diff": D, "Random": R, "Dori": Do,
                                 "MRTp": MRT_pearson, "RFDp": RFD_pearson,
                                 "MRTstd": MRT_std, "RFDstd": RFD_std, "RepTime": RepT,"percentile":P,"Smooth":SM})
            Data.to_csv(os.path.join(args.root, "result.csv"))
