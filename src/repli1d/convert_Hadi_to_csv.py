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

res=1000
X = [["chr%i" % i] * int(d/res) for i, d in enumerate(chromlength, 1)]
Pos = [range(0, int(d/res) * res, res)  for i, d in enumerate(chromlength, 1)]
X = np.concatenate(X).tolist()
Pos = np.concatenate(Pos).tolist()
Data= np.zeros_like(Pos,dtype=np.int)
Dataf = pd.DataFrame({"chrom":X, "chromStart":Pos})
view=None
for f in args.files:
    with open(f,"r") as file:
        file.readline() #skip the first
        for line in file.readlines():
            #print(line)
            if line.startswith("fixedStep"):
                if view is not None:
                    Data[Dataf["chrom"] == ch] += view
                ch = line.split()[1].split("=")[1]
                view = Data[Dataf["chrom"]==ch]
                print(ch,len(view))
                index=0
                continue
            else:
                if index <len(view):
                    view[index] += int(line)
                    #print(int(line))
                index += 1
                """
                if index> 100:
                    print(view[:100])
                    exit()
                """


pd.DataFrame({"chrom":X, "chromStart":Pos,"chromEnd":Pos,"signalValue":Data}).to_csv(args.output,sep="\t",index=False)


