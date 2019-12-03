import pandas as pd

import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--fn', type=str, default="")
parser.add_argument('--root', type=str, default="data/training/")



args = parser.parse_args()

Data = pd.read_csv(args.fn)

chs = set(Data["chrom"])

os.makedirs(args.root,exist_ok=True)
for ch in chs:
    sub = Data["chrom"] == ch
    for export in ["RFDe","RFDs","MRTe","MRTs","signal"]:
        sd = Data[sub]
        name = args.root+"/" + ch +"_" + export
        with open(name,"w") as f:
            f.write("\n".join([str(v) for v in sd[export]]))