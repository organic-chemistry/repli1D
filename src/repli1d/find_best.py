import numpy as np
import os
import pandas as pd
import glob

import argparse
from repli1d.hopt import score


parser = argparse.ArgumentParser()


parser.add_argument('--root', type=str, default="tmp")
parser.add_argument('--RFD', action="store_true")
parser.add_argument('--onlyOne', action="store_true")


args = parser.parse_args()

best = 2000
bfolder = ""
l = []
for folder in glob.glob(args.root+"/*"):
    if args.onlyOne:
        f =folder.split("/")[-1]

        #print(f,f.count("_"))
        if f.count("_")>1:
            continue
    if os.path.exists("%s/bestglobal_scores.csv"%folder):
        sc = score(folder,rfd=args.RFD)
        if args.RFD:
            sc = -sc
        if sc <= best:
            best = sc
            bfolder = folder
        l.append([sc,folder])

l.sort()
for v,f in l[:40]:
    print("%.3f %s" %(v,f))

if args.RFD:
    print(bfolder,-best)
else:
    print(bfolder,best)
