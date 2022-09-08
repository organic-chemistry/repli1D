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
parser.add_argument('--n_jobs', type=int, default=8)
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
parser.add_argument('--OE2IE',action="store_true")
parser.add_argument('--around_opti',action="store_true")



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
n_jobs = args.n_jobs
print(n_jobs)
if args.compMRT is not None:
    compMRT = args.compMRT
if args.compRFD is not None:
    compRFD = args.compRFD


#mark,correct,save,single,check_if_exist,rename_root

if cell =="GM12878":

    marks = [["results//nn_GM_from_None.csv",False,True,True,False,"rfd2init"]]
else:
    marks = [["results//nn_%s_from_None.csv" %(cell),False,True,True,False,"rfd2init"]]

marks += [["peak",False,False,False,False,""],
          ["exp4",False,False,False,False,""],
          ["oli",False,False,False,False,""],
          ["flat",True,False,False,False,""], #5
          ["ORC2",False,False,False,True,""],
          ["SNS",True,False,False,True,""],  # 7
          ["SNS",False,False,False,True,""],
          ["MCMp",False,False,False,True,""],
          ["Bubble",False,False,False,True,""],
          ["Bubble",True,False,False,True,""], #11
          ["MCM",False,False,False,True,""],
          ["MCMo",False,False,False,True,""]]
"""
marks += ["RNA_seq"]
marks += ['DNaseI',  'H2az', 'H3k27ac', 'H3k27me3', 'H3k36me3', 'H3k4me1',
         'H3k4me2', 'H3k4me3', 'H3k79me2', 'H3k9ac',  'H3k9me3', 'H4k20me1', 'H3k9me1']
#marks = ["MCM","MCMo","MCMp"]
marks = ["SNS"]
marks += ["Bubble"]

marks=["exp4","oli","flat"]
marks=["Mcm3","Mcm7","Orc2","Orc3"]

marks+=["exp4","oli","flat"]
marks += ["results/Raji_nn_global_profiles.csv"]
"""
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
    score = pd.read_csv("%s/global_corre.csv" % repo)
    # print(score["MRTp"][0])
    MRTp = float(score["MRTp"][0].split(",")[0][1:])
    MRTstd = score["MRTstd"][0]
    RFDp = float(score["RFDp"][0].split(",")[0][1:])
    RFDstd = score["RFDstd"][0]
    RepTime = score["RepTime"][0]
    #scorev = 2-c1-c2

    return MRTp, MRTstd, RFDp, RFDstd, RepTime


for mark,correct,save,single,check_if_exist,rename_root in marks:


    if check_if_exist:
        x, d = replication_data(cell, mark, chromosome=ch,
                                start=start, end=end,
                                resolution=5, raw=False, oData=False,
                                bp=True, bpc=False)
        if d ==[]:
            print("Skipping %s" % mark)
            continue
    if not args.around_opti:
        lp = []
        for kon in [1e-6]:
            for ndiff in np.arange(30,121,15)/110:
                #ndiff = 60
                for random_activation in [0, 0.05, 0.1, 0.2]:
                    for dori in [15]:
                        for fork_speed in [1.5]:
                            lp.append([kon,ndiff,random_activation,dori,fork_speed])

    else:
        lp=[]

        if "K562" in mark:
            lp0=np.array([3e-6,0.52,0,20,1.5])
            params=[[1,np.arange(0.25,2.5,0.1)],
                    [2,np.arange(0,0.3,0.05)],
                    [3,[1,5,10,15,20,30,40,50,75,100]],
                    [4,np.arange(0.5,5.1,0.5)]]
        if "Hela" in mark:
            lp0=np.array([3e-6,1.03,0,20,1.5])
            params=[[1,np.arange(0.32,2.5,0.1)],
                    [2,np.arange(0,0.3,0.05)],
                    [3,[1,5,10,15,20,30,40,50,75,100]],
                    [4,np.arange(0.5,5.1,0.5)]]
        if "GM" in mark:
            lp0=np.array([3e-6,0.56,0,20,1.5])
            params=[[1,np.arange(0.25,2.5,0.1)],
                    [2,np.arange(0,0.3,0.05)],
                    [3,[1,5,10,15,20,30,40,50,75,100]],
                    [4,np.arange(0.5,5.1,0.5)]]



        for p,rangep in params:
            for new_value in rangep:
                lp.append(lp0.copy())
                lp[-1][p]=new_value
        print(len(lp))

    for kon,ndiff,random_activation,dori,fork_speed in lp:
        if rename_root != "":
            mark0 = rename_root
        else:
            mark0 = mark
        if correct:
            mark0+="_corrected"

        # simulate
        filename = os.path.join(args.root, fl(OrderedDict([["mark", mark0],
                                                           ["ndiff", ndiff],
                                                           ["random_activation",
                                                               random_activation],
                                                           ["dori", dori],
                                                           ["kon", kon],
                                                           ["fork_speed",fork_speed]])))
        if args.reverse_profile:
            add = " reverse_profile"
        else:
            add = ""
        filename += add
        #filename += ".pick"
        score_file = filename + "/global_corre.csv"
        if os.path.exists(score_file) and not args.redo:
            MRTpearson, MRTstd, RFDpearson, RFDstd, Rep_Time = score(filename)
        else:
            print(filename)

            add=""
            if args.reverse_profile:
                add =" --reverse_profile "
            else:
                add = ""
            if correct:
                add += " --OE2IE "
            if save:
                add += " --save "
            if single:
                add += " --single "
            print(n_jobs)
            bgcmd = ("python src/repli1d/detect_and_simulate.py --input --visu "
                        "--signal %s --ndiff %.3f --dori %i --ch 1 "
                        "--name %s/ --resolution 5 --resolutionpol 5"
                        " --nsim 200  --wholecell --kon %.3e  --cutholes 1500"
                        " --experimental --n_jobs %i --noise %.2f --only_one --introduction_time 60 "
                        " --fspeed %.1f " % (mark, ndiff, dori,filename, kon,n_jobs,random_activation,fork_speed))
            bgcmd += add
            if cell in ["HeLaS3","Hela","K562","GM","Raji"]:
                csa=cell
                if cell in ["HeLaS3","Hela","Helas3"]:
                    csa = "Hela"


                commands = [ bgcmd + " --cell %s"%csa]
            if cell in ["K562norandom"]:
                commands = [ bgcmd + " --cell K562"]

            if ("GM" in cell) or ("Gm" in cell):

                commands = [ bgcmd + " --cell Gm12878 --comp GM12878 --cellseq GM06990" ]

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
