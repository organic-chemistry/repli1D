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
import json

parser = argparse.ArgumentParser()
parser.add_argument('--cmd', type=str, default="")
parser.add_argument('--root', type=str, default="")
parser.add_argument('--redo', action="store_true")
parser.add_argument('--pearson', action="store_true")
parser.add_argument('--rfd_opti_only', action="store_true")

parser.add_argument('--maxT', type=float,default=None)
parser.add_argument('--ndiff', nargs='+', type=int, default=[45,60, 75, 90, 105, 120,140])
parser.add_argument('--size_segment', type=float, default=110,
                    help="Size of the segment (in MB to simulate)")




args = parser.parse_args()


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
    MRTp = round(float(score["MRTp"][0].split(",")[0][1:]),2)
    MRTstd = score["MRTstd"][0]
    RFDp = round(float(score["RFDp"][0].split(",")[0][1:]),2)
    RFDstd = score["RFDstd"][0]
    RepTime = score["RepTime"][0]
    #scorev = 2-c1-c2

    return MRTp, MRTstd, RFDp, RFDstd, RepTime

pearson = args.pearson
pearson=False
maxi=0
mini=10000
params = {}
for ndiff in args.ndiff:
    #ndiff = 60
    for random_activation in [0,0.02, 0.05,0.1]:


        # simulate
        filename = os.path.join(args.root, fl(OrderedDict([["ndiff", ndiff],
                                                           ["random_activation",
                                                               random_activation]])))
        filename += "/"
        #filename += ".pick"

        if not os.path.exists(filename + "/global_corre.csv") or args.redo:
            print(filename)

            bgcmd = f"python src/repli1d/detect_and_simulate.py {args.cmd} --ndiff {ndiff} --noise {random_activation} --name {filename}"
            commands = [ bgcmd ]

            for command in commands:
                print(command)
                #exit()
                os.system(command)

        MRTpearson, MRTstd, RFDpearson, RFDstd, Rep_Time = score(filename)
        if pearson:
            if args.rfd_opti_only:
                new = RFDpearson
            else:
                new = MRTpearson+RFDpearson

        else:
            if args.rfd_opti_only:
                new = RFDstd
            else:
                new = RFDstd+MRTstd
        if (pearson and (new >= maxi)) or (not pearson and new<=mini):
            if args.maxT is  None or Rep_Time < args.maxT:
                if pearson:
                    maxi=new
                else:
                    mini=new
                print("New",maxi,ndiff,random_activation)
                params={"ndiff":ndiff/args.size_segment,"noise":random_activation}
        print(MRTpearson, MRTstd, RFDpearson, RFDstd,)

        D.append(ndiff)
        R.append(random_activation)
        MRT_std.append(MRTstd)
        RFD_std.append(RFDstd)
        MRT_pearson.append(MRTpearson)
        RFD_pearson.append(RFDpearson)
        RepT.append(np.median(Rep_Time))

        Data = pd.DataFrame({"Diff": D, "Random": R,
                             "MRTp": MRT_pearson, "RFDp": RFD_pearson, "MRTstd": MRT_std, "RFDstd": RFD_std, "RepTime": RepT})
        Data.to_csv(os.path.join(args.root, "result.csv"))

        with open(args.root+"/params.json","w") as f:
            json.dump(params,f)
if params == {}:
    print("No simulation are satisfying the time constraint")
    raise
