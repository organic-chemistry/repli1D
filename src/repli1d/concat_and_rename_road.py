#Concat from roadmap
import glob
import pandas as pd
import argparse
from repli1d.expeData import replication_data
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default="")
parser.add_argument('--output', type=str, default="")
args = parser.parse_args()


data = []
for ifile,file in enumerate(glob.glob(args.root+"*.csv")):
    if "input_road" in file:
        continue
    if "H2A.Z" in file:
        name = "H2az"
    else:
        name = file.split("/")[-1].split(".")[0].split("_")[1]
        name = name.replace("K", "k")[:-2]

    if ifile == 0:
        data = pd.read_csv(file,sep="\t")
        data[name] = data["signalValue"]
        print(data.columns)
        data.drop(["signalValue"],axis=1)
    else:
        tmp = pd.read_csv(file,sep="\t")
        data[name] = tmp["signalValue"]

data.to_csv(args.output,sep="\t",index=False)
