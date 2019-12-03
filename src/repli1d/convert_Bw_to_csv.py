import os
import argparse
from repli1d.expeData import replication_data
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default="")
parser.add_argument('--root', type=str, default="")
parser.add_argument('--output', type=str, default="")
parser.add_argument('--globalonly', action="store_true")
parser.add_argument('--resolution', type=int,default=1)
parser.add_argument('--indexfile', type=str,default=None)




args = parser.parse_args()


if args.indexfile is None:
    try:
        import pyBigWig

        cell = pyBigWig.open(args.file)
        chroms = cell.chroms()
        print(chroms)
    except:
        chroms = [248956422, 242193529, 198295559, 190214555, 181538259,
                       170805979, 159345973, 145138636, 138394717,
                       133797422, 135086622, 133275309, 114364328, 107043718,
                       101991189, 90338345, 83257441,
                       80373285, 58617616, 64444167, 46709983, 50818468]
else:
    chromsf = pd.read_csv(args.indexfile,sep="\t")
    lch = []
    for ch in chromsf.chrom:
        if ch not in lch:
            lch.append(ch)

    chroms = [sum(chromsf.chrom==ch) * args.resolution * 1000 for ch in lch] # The size is already at the resolution



#os.makedirs(args.root, exist_ok=True)

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
                            chromosome=ch, start=0, end=end, resolution=args.resolution)
    data.append(y)


X = [["chr%i" % i] * len(d) for i, d in enumerate(data, 1)]
Pos = [range(0, len(d) * args.resolution * 1000, args.resolution * 1000)  for i, d in enumerate(data, 1)]
X = np.concatenate(X).tolist()
Pos = np.concatenate(Pos).tolist()

data = np.concatenate(data, axis=0)

if not args.globalonly:
    pd.DataFrame({"chrom":X, "chromStart":Pos,"chromEnd":Pos}).to_csv(args.output+"index",sep="\t",index=False)

    pd.DataFrame({"signalValue":data}).to_csv(args.output,sep="\t",index=False)

pd.DataFrame({"chrom":X, "chromStart":np.array(Pos),"chromEnd":np.array(Pos) ,"signalValue":data}).to_csv(args.output[:-4]+"wh.csv",sep="\t",index=False)
