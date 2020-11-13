import os
import argparse
from repli1d.expeData import replication_data
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--file',type=str, default="")
parser.add_argument('--output', type=str, default="")
parser.add_argument('--percentile',type=int,default=20)
parser.add_argument('--from_sim',action="store_true")
parser.add_argument('--rfd',type=float,default=0.3)



args = parser.parse_args()

if not args.from_sim:
    data = pd.read_csv(args.file,sep="\t")

else:
    data = pd.read_csv(args.file)
    data["signalValue"] = data["signal"]
    data["chromStart"] = data["chromStart"] * 1000
    data["chromEnd"] = data["chromStart"] + 5000
    data["chrom"] = ["chr%s"%d[5:] for d in data["chrom"]]
    #print("RFD",np.sum(data["RFDe"]>args.rfd) / len(data["RFDe"]))
    #data["signalValue"] = data["RFDs"]>args.rfd

p = np.percentile(data.signalValue, [args.percentile])[0]
print(p)
data.signalValue[data.signalValue < p] = 0


p_start = None
p_end = None
current_ch = data.chrom[0]
if data.signalValue[0] != 0:
    p_start = data.chromStart[0]
    c_s = data.signalValue[0]
final = []
for ch,start,end,signal in zip(data["chrom"],data["chromStart"],data["chromEnd"],data["signalValue"]):

    if  p_start != None and (signal == 0 or current_ch != ch):
        final.append([current_ch,p_start,p_end,c_s,len(final)])
        p_start = None
        c_s = 0


    if p_start is None and signal != 0:
        p_start = start
        c_s = 0
        current_ch = ch

    if p_start != None:
        c_s += signal
        p_end=end

pd.DataFrame(final,columns=["chrom","chromStart","chromEnd","signalValue","index_ori"]).to_csv(args.output,index=False,sep="\t")