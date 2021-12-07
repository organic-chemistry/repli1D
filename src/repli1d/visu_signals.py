# detect peak


from repli1d.analyse_RFD import detect_peaks, compare, smooth, get_expression
from repli1d.fast_sim import get_fast_MRT_RFDs
from repli1d.expeData import replication_data
from repli1d.single_mol_analysis import compute_info, compute_real_inter_ori
import sys
import argparse
from repli1d.visu_browser import plotly_blocks
import pylab
import numpy as np
import ast
import os

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=5000)
parser.add_argument('--end', type=int, default=120000)
parser.add_argument('--ch', type=str, default="1")
parser.add_argument('--resolution', type=float, default=5)
parser.add_argument('--ndiff', type=int, default=60)
parser.add_argument('--percentile', type=int, default=82)
parser.add_argument('--cell', type=str, default="K562")
parser.add_argument('--visu', action="store_true")
parser.add_argument('--name', type=str, default="tmp.html")
parser.add_argument('--nsim', type=int, default=200)
parser.add_argument('--input', action="store_true")
parser.add_argument('--correct', action="store_true")
parser.add_argument('--continuous', action="store_true")
parser.add_argument('--nan0', action="store_true")
parser.add_argument('--noise', type=float, default=.1)
parser.add_argument('--fspeed', type=float, default=.3)
parser.add_argument('--comp', type=str, default=None)
parser.add_argument('--signal', nargs='+', type=str, default=[])
parser.add_argument('--filename', nargs='+', type=str, default=[])


# Cell is use to get signal
# comp is use to compare

args = parser.parse_args()

start = args.start
end = args.end
ch = args.ch
cell = args.cell
resolution_polarity = args.resolution
resolution = args.resolution
exp_factor = 4
percentile = args.percentile
fork_speed = args.fspeed
kon = 0.005
ndiff = args.ndiff
nsim = args.nsim
comp = args.comp
if comp is None:
    comp = cell

ToSee = []
sup_sig=None
if args.signal != []:
    blocs = []
    tmp = []
    for s in args.signal:
        if s != ",":
            tmp.append(s)
        else:
            blocs.append(tmp)
            tmp = []
    if tmp != []:
        blocs.append(tmp)
    for bloc in blocs:
        btosee = []
        for signal in bloc:
            print(signal)

            def mini(resolution,signal):
                if signal in ["MRT","MRTstd"] and not ("Yeast" in cell or "Cerevisae" in cell):
                    if resolution < 10:
                        print("Warning MRT res will be dispalyed at 10 kb")
                    return max(resolution,10)
                else:
                    return resolution



            if signal == "Exp":
                Xg, Yg, xmg, ymg, direction = get_expression(
                    cell, ch, start, end, resolution, min_expre=1)
                d3p = direction
                x = Xg
                d3p = Yg
                #ymg[ymg<1] = np.nan
                sup_sig=[xmg,-ymg,"neg"]
                #d3p[np.abs(d3p)<1]=np.nan
            elif "[" not in signal and "--" not in signal:
                print("H")
                x, d3p = replication_data(cell, signal, chromosome=ch,
                                          start=start, end=end,
                                          resolution=mini(resolution,signal), raw=False, filename=None)
                #print(d3p)
            elif "--" in signal or ":" in signal:
                weights_list = []
                if "--" in signal:
                    signal, sigv = signal.split("--")


                    if ":" in sigv:
                        sigv, *weights_list  = sigv.split(":")
                    x, d3p = replication_data(cell, signal, chromosome=ch,
                                              start=start, end=end,
                                              resolution=mini(resolution,signal), raw=False, filename=None, signame=sigv)
                    raw = d3p
                    print(np.nansum(d3p))

                    if sigv.count(":") == 1:
                        weights_list = [weights_list]
                    print(weights_list)
                    signal += "-" + sigv
                else:
                    signal, *weights_list = signal.split(":")
                    x, raw = replication_data(cell, signal, chromosome=ch,
                                              start=start, end=end,
                                              resolution=mini(resolution,signal), raw=False, filename=None)

                    if signal.count(":") == 1:
                        weights_list = [weights_list]
                for weights in weights_list:

                #if len(weights)== 0
                    print(weights)
                    #print(ast.literal_eval(weights))
                    try:
                        weights = ast.literal_eval(weights)
                    except:
                        if weights not in ["submed"]:
                            print("Warning unrecon option %s"%weights)
                        pass
                    if type(weights) == list:

                        d3p = np.zeros_like(raw)
                    #print(weights)
                        for smoothv, weightv in weights:
                            d3p += smooth(raw, int(smoothv)) * weightv

                        raw = d3p
                    else:
                        #print(type(weights),np.nanmean(raw))
                        if weights == "submed":
                            raw -= np.nanmedian(raw)
                        d3p = raw
                #print(weights, "here")

            #print(d3p[:50])
            if args.nan0 and "Exp" not in signal:

                d3p[np.isnan(d3p)] = 0
            if len(signal) > 20:

                signal0 = os.path.split(signal)[1]
                if signal0 != "":
                    signal = signal0
            btosee.append([x, d3p, signal])
            if sup_sig != None:
                btosee.append(sup_sig)
                sup_sig=None


        ToSee.append(btosee)
"""
for signal in args.filename:
    print(signal)
    x, d3p = replication_data(cell, "", chromosome=ch,
                              start=start, end=end,
                              resolution=resolution, raw=False,filename=signal)

    ToSee += [[[x,d3p, signal]]]
"""
plotly_blocks(x, ToSee, name=args.name, default="lines")
