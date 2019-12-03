import os
import argparse
from repli1d.expeData import replication_data

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default="")
parser.add_argument('--root', type=str, default="")
parser.add_argument('--cell', type=str, default="")


args = parser.parse_args()

chromlength = [248956422, 242193529, 198295559, 190214555, 181538259,
               170805979, 159345973, 145138636, 138394717,
               133797422, 135086622, 133275309, 114364328, 107043718,
               101991189, 90338345, 83257441,
               80373285, 58617616, 64444167, 46709983, 50818468]

os.makedirs(args.root, exist_ok=True)
for ch, l in enumerate(chromlength, 1):
    x, y = replication_data("hela", args.file, filename=args.file,
                            chromosome=ch, start=0, end=None, resolution=1)
    namef = args.root+"/%s.w1kb.chr%i.R.txt" % (args.cell, ch)
    print(namef)
    with open(namef, "w") as f:
        f.writelines("\n".join([str(iy) for iy in y]))
