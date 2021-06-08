# detect peak


from collections import OrderedDict
from repli1d.analyse_RFD import compare
from repli1d.fast_sim import get_fast_MRT_RFDs
from repli1d.expeData import replication_data
import argparse
import numpy as np
import os
import pickle
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=5000)
parser.add_argument('--end', type=int, default=120000)
parser.add_argument('--ch', type=int, default=1)
parser.add_argument('--ndiff', type=int, default=60)
parser.add_argument('--cell', type=str, default="K562")
parser.add_argument('--root', type=str, default="./")
parser.add_argument('--redo', action="store_true")
parser.add_argument('--name', type=str, default="tmp.html")
parser.add_argument('--nsim', type=int, default=200)
parser.add_argument('--signal', type=str, default="peak")
parser.add_argument('--correct', action="store_true")
parser.add_argument('--wig', action="store_true")
parser.add_argument('--array', type=int, default=None)
parser.add_argument('--noise', type=float, default=.1)
parser.add_argument('--compMRT', type=str, default=None)
parser.add_argument('--compRFD', type=str, default=None)
parser.add_argument('--reverse_profile',  action="store_true")


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
compMRT = cell
compRFD = cell
if args.compMRT is not None:
    compMRT = args.compMRT
if args.compRFD is not None:
    compRFD = args.compRFD


if cell =="GM12878":
    marks = ["results//nn_GM_from_None.csv"]
else:
    marks = ["results//nn_%s_from_None.csv" %(cell)]

marks += ["RNA_seq"]

marks += ["Bubble"]
marks += ['DNaseI', 'ORC2',  'H2az', 'H3k27ac', 'H3k27me3', 'H3k36me3', 'H3k4me1',
         'H3k4me2', 'H3k4me3', 'H3k79me2', 'H3k9ac',  'H3k9me3', 'H4k20me1', 'H3k9me1']

marks += ["SNS"]

#marks = ["DNaseI"]


if args.wig:
    marks = marks[:2]+[e+"wig" for e in marks[2:]]

if args.array is not None:
    marks = marks[args.array-1:args.array]
print(marks)


def fl(name):
    def format_value(val):
        # print(type(val))
        if type(val) in [float, np.float64]:
            return "%.2e" % val
        else:
            return str(val)
    if type(name) in [dict, OrderedDict]:
        return "".join(["%s-%s" % (p, format_value(fl(value))) for p, value in name.items()])
    else:
        return name


M = []
D = []
R = []
Do = []
MRT_pearson = []
RFD_std = []

RFD_pearson = []
MRT_std = []
RepT = []
Kon = []


def score(repo, rfd=False):
    score = pd.read_csv("%s//global_corre.csv" % repo)
    # print(score["MRTp"][0])
    MRTp = float(score["MRTp"][0].split(",")[0][1:])
    MRTstd = score["MRTstd"][0]
    RFDp = float(score["RFDp"][0].split(",")[0][1:])
    RFDstd = score["RFDstd"][0]
    RepTime = score["RepTime"][0]
    #scorev = 2-c1-c2

    return MRTp, MRTstd, RFDp, RFDstd, RepTime


for mark in marks:

    x, d = replication_data(cell, mark, chromosome=ch,
                            start=start, end=end,
                            resolution=5, raw=False, oData=False,
                            bp=True, bpc=False)
    print(mark, d)
    if d == []:
        print("Skipping %s" % mark)
        continue
    for kon in [5e-7]:
        for ndiff in [30, 45, 60, 75, 90, 105, 120]:
            #ndiff = 60
            for random_activation in [0, 0.05, 0.1, 0.2]:
                for dori in [1,5, 15, 30]:
                    if "/" in mark:
                        mark0 = "rfd2init"
                    else:
                        mark0 = mark
                    # simulate
                    filename = os.path.join(args.root, fl(OrderedDict([["mark", mark0],
                                                                       ["ndiff", ndiff],
                                                                       ["random_activation",
                                                                           random_activation],
                                                                       ["dori", dori],
                                                                       ["kon", kon]])))
                    if args.reverse_profile:
                        add = " reverse_profile"
                    else:
                        add = ""
                    filename += add
                    #filename += ".pick"

                    if not os.path.exists(filename + "/global_corre.csv") or args.redo:
                        print(filename)


                        if args.reverse_profile:
                            add =" --reverse_profile "
                        else:
                            add = ""
                        bgcmd = ("python src/repli1d/detect_and_simulate.py --input --visu " 
                                    "--signal %s --ndiff %.3f --dori %i --ch 1 "
                                    "--name %s/ --resolution 5 --resolutionpol 5"
                                    " --nsim 200  --wholecell --kon 1e-5 --save --cutholes 1500"
                                    " --experimental --n_jobs 8 --noise %.2f --only_one " % (mark, ndiff/110, dori, filename, random_activation))
                        bgcmd += add
                        if cell in ["HeLaS3","Hela","K562","GM"]:
                            csa=cell
                            if cell in ["HeLaS3","Hela","Helas3"]:
                                csa = "Hela"


                            commands = [ bgcmd + "--cell %s"%csa]
                        if ("GM" in cell) or ("Gm" in cell):

                            commands = [ bgcmd + "--cell Gm12878 --comp GM12878 --cellseq GM06990" ]

                        for command in commands:
                            print(command)
                            #exit()
                            os.system(command)

                    MRTpearson, MRTstd, RFDpearson, RFDstd, Rep_Time = score(filename)
                    print(MRTpearson, MRTstd, RFDpearson, RFDstd)

                    M.append(mark)
                    D.append(ndiff)
                    R.append(random_activation)
                    Do.append(dori)
                    Kon.append(kon)
                    MRT_std.append(MRTstd)
                    RFD_std.append(RFDstd)
                    MRT_pearson.append(MRTpearson)
                    RFD_pearson.append(RFDpearson)
                    RepT.append(np.median(Rep_Time))

                    Data = pd.DataFrame({"marks": M, "Diff": D, "Random": R, "Dori": Do,
                                         "MRTp": MRT_pearson, "RFDp": RFD_pearson, "kon": Kon,
                                         "MRTstd": MRT_std, "RFDstd": RFD_std, "RepTime": RepT})
                    if args.array is None:
                        Data.to_csv(os.path.join(args.root, "result.csv"))
                    else:
                        Data.to_csv(os.path.join(args.root, "%i_result.csv" % args.array))
