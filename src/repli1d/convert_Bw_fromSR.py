import os
import argparse
from repli1d.expeData import replication_data
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--files',  nargs='+',type=str, default=[])
parser.add_argument('--output', type=str, default="")
parser.add_argument('--remove', type=float, default=None)



args = parser.parse_args()

chromlength = [248956422, 242193529, 198295559, 190214555, 181538259,
               170805979, 159345973, 145138636, 138394717,
               133797422, 135086622, 133275309, 114364328, 107043718,
               101991189, 90338345, 83257441,
               80373285, 58617616, 64444167, 46709983, 50818468]

#os.makedirs(args.root, exist_ok=True)


data = []
for ch, l in enumerate(chromlength, 1):
    Y = []
    for file in args.files:
        x, y = replication_data("hela", file, filename=file,
                                chromosome=ch, start=0, end=None, resolution=5)
        if args.remove is not None:
            print(file,"removing %i points" %np.sum(y>args.remove))
            y[y>args.remove] = np.nan
        Y.append(y)

    #if len(args.files) == 1:
    #    data.append(Y)
    #else:

    #
    data.append(np.nanmean(Y,axis=0))



X = [["chr%i" % i] * len(d) for i, d in enumerate(data, 1)]
Pos = [range(0, len(d) * 5000, 5000)  for i, d in enumerate(data, 1)]
X = np.concatenate(X).tolist()
Pos = np.concatenate(Pos).tolist()

data = np.concatenate(data, axis=0)

pd.DataFrame({"chrom":X, "chromStart":Pos,"chromEnd":Pos,"signalValue":data}).to_csv(args.output,sep="\t",index=False)


