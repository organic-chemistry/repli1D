import os
import argparse
from repli1d.expeData import replication_data
import numpy as np
import pandas as pd
from repli1d.analyse_RFD import nan_polate

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default="")
parser.add_argument('--output', type=str, default="")

args = parser.parse_args()

to1 = True
if not to1:
    raise

if to1:
    resolution = 1


chroms = [248956422, 242193529, 198295559, 190214555, 181538259,
               170805979, 159345973, 145138636, 138394717,
               133797422, 135086622, 133275309, 114364328, 107043718,
               101991189, 90338345, 83257441,
               80373285, 58617616, 64444167, 46709983, 50818468]

data = []
X = []
for ch in range(1,len(chroms)+1):

    if type(chroms) == list:
        end = chroms[ch-1]
        end=int(end / 1000)
    else:
        end = None
    print(ch,end)
    x, y = replication_data("hela", args.file, filename=args.file,
                            chromosome=ch, start=0, end=end, resolution=resolution)

    if to1:
        y = nan_polate(y)
    data.append(y)

X = [["chr%i" % i] * len(d) for i, d in enumerate(data, 1)]
Pos = [range(0, len(d) * resolution * 1000, resolution * 1000)  for i, d in enumerate(data, 1)]
X = np.concatenate(X).tolist()
Pos = np.concatenate(Pos).tolist()

data = np.concatenate(data, axis=0)

pd.DataFrame({"chrom":X, "chromStart":np.array(Pos),"chromEnd":np.array(Pos) ,"signalValue":data}).to_csv(args.output,sep="\t",index=False)
