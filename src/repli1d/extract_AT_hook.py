import argparse
import os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--filename', type=str, default=None)
parser.add_argument('--output', type=str, default=None)
parser.add_argument('--constant', nargs='+', type=int, default=[5,10,20,30])
parser.add_argument('--resolution', type=int, default=5)


args = parser.parse_args()

def get_percent(line):
    if len(line)<25 or line.count("N")> 15:
        return np.nan
    else:
        line=line.upper()
        G = line.count("G")
        C = line.count("C")
        A = line.count("A")
        T = line.count("T")
        if G+C==0:
            return 1-0.5
        else:
            return (A+T)/(G+C+A+T) - 0.5

with open(args.filename,"r") as f:
    line = f.readline()
    data={c:[] for c in args.constant}
    chrs=[]
    while line:
        if line.startswith(">chr"):
            chrs.append(line[1:-1])
            for c in args.constant:
                data[c].append([])
        else:
            line = line.strip("\n")
            for sline in [line[:25],line[25:]]:
                p = get_percent(line)
                #print(p)
                for c in args.constant:
                    data[c][-1].append(np.exp(c*p))
        line = f.readline()

from itertools import zip_longest  # for Python 3.x
def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return list(zip_longest(*[iter(iterable)] * n, fillvalue=padvalue))

for c in args.constant:
    for i_sub,sub in enumerate(data[c]):
        data[c][i_sub] = [float("%.4f"%np.nanmean(g)) for g in grouper((args.resolution*1000)//25,sub,np.nan)]


c0=args.constant[0]

X = [[chrs[i]] * len(d) for i, d in enumerate(data[c0])]
Pos = [range(0, len(d) * args.resolution * 1000, args.resolution * 1000)  for i, d in enumerate(data[c0])]
X = np.concatenate(X).tolist()
Pos = np.concatenate(Pos).tolist()

for c in args.constant:
    s_data=data[c]
    s_data = np.concatenate(s_data, axis=0)

    pd.DataFrame({"chrom":X, "chromStart":np.array(Pos),"chromEnd":np.array(Pos) ,"signalValue":s_data}).to_csv(args.output+"_%i.csv"%c,sep="\t",index=False)
