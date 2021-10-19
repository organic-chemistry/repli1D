import argparse
import subprocess
import os
import json
import pandas as pd
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, default="datafile.csv")
parser.add_argument('--no-optiMRT',dest="optiMRT", action="store_false")
parser.add_argument('--no-optiRFD',dest="optiRFD", action="store_false")
parser.add_argument('--resolution', type=int,default=5,
                    help="Resolution in kb")
parser.add_argument('--speed', type=float,default=2.5,
                    help="speed in kb/min")
parser.add_argument('--nsim', type=int,default=200,
                        help="number of simulation")
parser.add_argument('--repTime', type=float,default=800,
                    help="replication time in minute")
parser.add_argument('--repTime_max_factor', type=float,default=1.2,
                    help="replication time factor in repTime unit")
parser.add_argument('--max_epoch', type=int,default=150,
                    help="maximum number of epoch for Neural network training")
parser.add_argument('--window', type=int,default=401,
                    help="window size (in resolution unit) for the neural network (must be impair number)")
parser.add_argument('--nfilters', type=int,default=15,
                    help="conv filters")
parser.add_argument('--chr_sub', type=int,default=2,
                    help="chromosome on which perform the small optimisation")
parser.add_argument('--introduction_time', type=float,default=60,
                    help="caracteristic time of the exponential increase of firing factors ")
parser.add_argument('--cutholes', type=int,default=1500,
                    help="remove parts of the genome where there is no information on length given by the argument (in resolution size)")
parser.add_argument('--root', type=str, default="./",
                    help="where to store the results")
parser.add_argument('--update', action="store_true")
parser.add_argument('--no-safety', dest="safety",action="store_false")
parser.add_argument('--RFDonly', dest="RFDonly",action="store_true")


args = parser.parse_args()

#Get genome length
data=pd.read_csv(args.data)
if args.optiMRT and "MRT" not in data.columns:
    print(f"Missing MRT data in {args.data}" )
    raise
if args.optiRFD and "OKSeq" not in data.columns:
    print(f"Missing OKSeq data in {args.data}" )
    raise


l_ch = list(set(data.chr))
if args.chr_sub not in l_ch:
    print(f"Chromosome for optimisation (--chr_sub) must be choosen inside this list {l_ch}")
    raise
if args.window % 2  != 1:
    print("Argument window must be impair")
    raise

cellcmd=f" --datafile {args.data} "


print("Estimating Ndiff (not taking into account holes)")
ndiff = len(data)/(2*args.repTime*args.speed)
print(f"If always actif, the total number of firing factor is {ndiff}")
ndiff /= len(data)/1000
print(f"USing density {ndiff} in firing factor per Mb")


small_sub = f" --n_jobs 6 --fspeed {args.speed/args.resolution} --single --visu --experimental --input --resolution {args.resolution} --resolutionpol {args.resolution} "
if args.cutholes != 0:
    small_sub+= f"--cutholes {args.cutholes} "
if args.introduction_time > 0.20 * args.repTime:
    print("Introduction time larger than 20% of the replication time")
    print("If it is not on purpose please set --no-safety")
    if args.safety:
        raise
small_sub += f"--introduction_time {args.introduction_time} "

kon = 1e-5 * 2875000 / (len(data) * args.resolution )
print(f"Using kon = {kon}")

standard_parameters = small_sub + f" --exclude_noise_save --wholecell --kon {kon} --noise 0.0 --nsim {args.nsim}  --dori 5 "

maxT = args.repTime * args.repTime_max_factor
#maxT *= 100
standard_parameters += "--ndiff %.2f %s"%(ndiff,cellcmd)

cmd=[]
for loop in range(5):

    if loop != 0:
        directory_nn = args.root+f"/RFD_to_init_nn_{loop}/"
        cmd += [[f"python src/repli1d/nn.py --reduce_lr --nfilters {args.nfilters} --listfile {directory}/global_profiles.csv --datafile --marks RFDs MRTs --root {directory_nn} --sm 10  --noenrichment --window {args.window} --max_epoch {args.max_epoch}",
                 ""]]
    #print(sum(data.chr==args.chr_sub)*args.resolution/1000)
    ndiff0 = max(int(sum(data.chr==args.chr_sub)*ndiff*args.resolution/1000),1)
    ndiffs = " ".join([str(int(ndiff0*f)) for f in np.arange(1.5,8,0.5)])
    #ndiffs = " ".join([str(int(ndiff0*f)) for f in np.arange(1,2,0.5)])


    directory_opti = args.root+f"/_RFD_to_init_small_opti_{loop}/"
    end = int(sum(data.chr==args.chr_sub) * args.resolution) # in kb
    cmd_opti = f"python src/repli1d/small_opty.py --size_segment {end/1000} --ndiff {ndiffs} --root {directory_opti}  --cmd '{small_sub} --start 0 --end {end} --ch {args.chr_sub} --nsim 200 --dori 10 {cellcmd} "

    if loop == 0:
        if args.RFDonly:
            cmd_opti += "--signal peakRFDonly' "
        else:
            cmd_opti += "--signal peak' "

    else:
        cmd_opti += f"--signal {directory_nn}/nn_global_profiles.csv' --maxT {maxT} "

    cmd += [[cmd_opti,""]]

    directory = args.root+f"/wholecell_{loop}/"

    cmd_wholecell=f"python src/repli1d/detect_and_simulate.py {standard_parameters} --name {directory} --extra_param {directory_opti}/params.json "

    if loop == 0:
        if args.RFDonly:
            cmd_wholecell += "--signal peakRFDonly "
        else:
            cmd_wholecell += "--signal peak"
    else:
        cmd_wholecell += f"--signal {directory_nn}/nn_global_profiles.csv"

    cmd += [ [cmd_wholecell,
             directory+"/global_profiles.csv"]]




redo = not args.update
exe= False
script = []
for cm in cmd:
    print(cm)
    if type(cm) == list:
        sup=None
        if "global_profile" in cm[1]:
            sup = "python src/repli1d/average_expe.py --dir %s" % "/".join(cm[1].split("/")[:-1])
        if not redo and os.path.exists(cm[1]):
            print(sup)
            script.append(sup)
            if exe:
                process = subprocess.Popen(sup, shell=True, stdout=subprocess.PIPE)
                process.wait()
            continue

        else:
            cm = cm[0]




        script.append(cm)
        if exe:
            process = subprocess.Popen(cm, shell=True, stdout=subprocess.PIPE)
            process.wait()
        if sup is not None:
            print(sup)
            script.append(sup)
            if exe:
                process = subprocess.Popen(sup, shell=True, stdout=subprocess.PIPE)
                process.wait()

with open("script.sh","w") as f:
    f.writelines("\n".join(script))
