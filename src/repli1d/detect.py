# detect peak


from repli1d.analyse_RFD import detect_peaks
import argparse
import pickle
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=5000)
parser.add_argument('--end', type=int, default=120000)
parser.add_argument('--ch', type=int, default=1)
parser.add_argument('--resolution', type=int, default=1)
parser.add_argument('--cell', type=str, default="K562")
parser.add_argument('--cellMRT', type=str, default=None)
parser.add_argument('--cellRFD', type=str, default=None)
parser.add_argument('--percentile', type=int, default=82)
parser.add_argument('--dec', type=int, default=None)
#parser.add_argument('--visu', action="store_true")
parser.add_argument('--name', type=str, default="detected.peak")
parser.add_argument('--correct', action="store_true")
parser.add_argument('--recomp', action="store_true")
parser.add_argument('--smoothpeak', type=int, default=5)


args = parser.parse_args()

start = args.start
end = args.end
ch = args.ch
cell = args.cell
resolution_polarity = args.resolution
exp_factor = 4
percentile = args.percentile


chromlength = [248956422, 242193529, 198295559, 190214555, 181538259,
               170805979, 159345973, 145138636, 138394717,
               133797422, 135086622, 133275309, 114364328, 107043718,
               101991189, 90338345, 83257441,
               80373285, 58617616, 64444167, 46709983, 50818468]
data = []

for chrom, length in enumerate(chromlength, 1):

    data.append(detect_peaks(0, length//1000, chrom,
                             resolution_polarity=resolution_polarity,
                             exp_factor=exp_factor,
                             fsmooth=args.smoothpeak,
                             percentile=percentile, cell=cell,cellMRT=args.cellMRT,cellRFD=args.cellRFD,recomp=args.recomp,dec=args.dec)[1])

data = np.concatenate(data)
data[data==0] = np.nan
data[~np.isnan(data)] = 1
pd.DataFrame({"signalValue":data}).to_csv("ARS.csv",index=False)

with open(args.name, "wb") as f:
    pickle.dump(data, f)
