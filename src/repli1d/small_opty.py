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
    MRTp = float(score["MRTp"][0].split(",")[0][1:])
    MRTstd = score["MRTstd"][0]
    RFDp = float(score["RFDp"][0].split(",")[0][1:])
    RFDstd = score["RFDstd"][0]
    RepTime = score["RepTime"][0]
    #scorev = 2-c1-c2

    return MRTp, MRTstd, RFDp, RFDstd, RepTime


maxi=0
params = {}
for ndiff in [30, 45, 60, 75, 90, 105, 120,140]:
    #ndiff = 60
    for random_activation in [0, 0.05]:


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
        new = MRTpearson+RFDpearson
        if new > maxi:
            maxi=new
            params={"ndiff":ndiff/110,"noise":random_activation}
        print(MRTpearson, MRTstd, RFDpearson, RFDstd)

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