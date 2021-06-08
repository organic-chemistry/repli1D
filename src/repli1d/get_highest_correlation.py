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
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default="")

args = parser.parse_args()

simus = glob.glob(args.root+"/*_RFD_to_init_wholecell*/summary.csv")

maxi = 0
selected_fich = ""
for fich in simus:
    data = pd.read_csv(fich)
    m = np.array((data["MRTp"] + data["RFDp"]))[0]
    print(fich,m)

    if m > maxi:
        maxi = m
        selected_fich = fich

selected_fich = selected_fich.replace("summary","nn_global_profiles")
selected_fich = selected_fich.replace("wholecell","nn")
print(selected_fich)
with open(args.root+"/highest_correlation.csv","w") as f:
    f.write(selected_fich)


