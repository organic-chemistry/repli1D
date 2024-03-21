import argparse
import subprocess
import os
import json
import pandas as pd
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, default="datafile.csv")
parser.add_argument('--name_script', type=str, default="script.sh")

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
parser.add_argument('--repTime_max_factor', type=float,default=3,
                    help="replication time factor in repTime unit")
parser.add_argument('--dori', type=float,default=5,
                    help="average distance between origins (in kb)")
parser.add_argument('--max_epoch', type=int,default=150,
                    help="maximum number of epoch for Neural network training")
parser.add_argument('--window', type=int,default=401,
                    help="window size (in resolution unit) for the neural network (must be impair number)")
parser.add_argument('--nfilters', type=int,default=50,
                    help="conv filters")
parser.add_argument('--percentile', type=int,default=82,
                    help="percentile of Delta RFD for I0")
parser.add_argument('--chr_sub', type=str,default="chr2",
                    help="chromosome on which perform the small optimisation")
parser.add_argument('--introduction_time', type=float,default=60,
                    help="caracteristic time of the exponential increase of firing factors ")
parser.add_argument('--cut_holes', type=int,default=1500,
                    help="remove parts of the genome where there is no information on length given by the argument (in kb)")
parser.add_argument('--masking', type=int,default=200,
                    help="remove parts of the genome for computing correlation of masking size around holes(in kb)")
parser.add_argument('--root', type=str, default="./",
                    help="where to store the results")
parser.add_argument('--threads', type=int,default=8,help="number of threads for the DNA simulations")
parser.add_argument('--update', action="store_true")
parser.add_argument('--no-safety', dest="safety",action="store_false")
parser.add_argument('--RFDonly', dest="RFDonly",action="store_true")
parser.add_argument('--grid_rfd_opti_only', dest="grid_rfd_opti_only",action="store_true",help="For the small opti")
parser.add_argument('--exclude_noise', dest="exclude_noise",action="store_true")
parser.add_argument('--show', action="store_true")
parser.add_argument('--test', action="store_true")
parser.add_argument('--reduce_lr', action="store_true")
parser.add_argument('--single', action="store_true",help="Compute single molecule information (only for wholecell)")
parser.add_argument('--save', action="store_true",help="Save individual MRTs and RFDs (only for wholecell)")
parser.add_argument('--pearson', action="store_true",help="Perform the grid search optimisation on sum of pearson correlation (otherwise on absolute error)")
parser.add_argument('--max_factor_reptime', type=float,default=1.41)
parser.add_argument('--on_input_signal', type=str,default=None)
parser.add_argument('--sm', type=int,default=10,help="Smoothing of the signal before predicting using nn")
parser.add_argument('--no_randval',dest="randval", action="store_false")
parser.add_argument('--logr', dest="logr",action="store_true")


args = parser.parse_args()

#Get genome length
data=pd.read_csv(args.data)
if args.optiMRT and "MRT" not in data.columns:
    print(f"Missing MRT data in {args.data}" )
    raise
if args.optiRFD and "OKSeq" not in data.columns:
    print(f"Missing OKSeq data in {args.data}" )
    raise

if args.test:
    max_epoch=10
    nsim=50
    fexplo = np.arange(.8,1.21,0.2)
else:
    max_epoch=args.max_epoch
    nsim=args.nsim
    delta=(args.max_factor_reptime-0.61)/8
    fexplo = np.arange(.6,args.max_factor_reptime,delta)


l_ch = list(set(data.chrom))
if args.chr_sub not in l_ch:
    print(f"Chromosome for optimisation (--chr_sub) must be choosen inside this list {l_ch}")
    raise
window=args.window
if args.window % 2  != 1:
    window = args.window+1


cellcmd=f" --datafile {args.data} "


print("Estimating Ndiff (not taking into account holes)")
ndiff = len(data)*args.resolution/(2*args.repTime*args.speed)
print(f"If always actif, the total number of firing factor is {ndiff}")
ndiff /= len(data)*args.resolution/1000
print(f"USing density {ndiff} in firing factor per Mb")


small_sub = f" --percentile {args.percentile} --mrt_res {args.resolution} --n_jobs {args.threads} --fspeed {args.speed} --visu --experimental --input --resolution {args.resolution} --resolutionpol {args.resolution} "
if args.cut_holes != 0:
    small_sub+= f"--cutholes {args.cut_holes} "
if args.masking != 0:
    small_sub+= f"--masking {args.masking} "
if args.logr:
    small_sub+= f"--logr "
if args.introduction_time > 0.20 * args.repTime:
    print("Introduction time larger than 20% of the replication time")
    print("If it is not on purpose please set --no-safety")
    if args.safety:
        raise
small_sub += f"--introduction_time {args.introduction_time} --dori {args.dori}"



kon = 8.625 / (len(data) * args.resolution ) / 3  * 3
print(f"Using kon = {kon}")

standard_parameters = small_sub + f"  --wholecell --kon {kon} --noise 0.0 --nsim {nsim} "

if args.exclude_noise:
    #print("Laaaaaaaaaaaaa")
    standard_parameters += " --exclude_noise_save "

maxT = args.repTime * args.repTime_max_factor
#maxT *= 100
standard_parameters += "--ndiff %.2f %s"%(ndiff,cellcmd)

cmd = [f"mkdir -p {args.root}"]
cmd += [f"cp {args.name_script} {args.root}/"]

nloop=5
if args.on_input_signal != None:
    nloop=1

for loop in range(nloop):

    if loop != 0:
        extra_nn = ""
        if args.reduce_lr:
            extra_nn = " --reduce_lr "
        
        directory_nn = args.root+f"/RFD_to_init_nn_{loop}/"
        cmd += [[f"python src/repli1d/nn.py --generator {extra_nn} --max_epoch {max_epoch} --add_noise --nfilters {args.nfilters} --listfile {directory}/global_profiles.csv --datafile --marks RFDs MRTs --root {directory_nn} --sm {args.sm}  --noenrichment --window {window}",
                 ""]]
    #print(sum(data.chr==args.chr_sub)*args.resolution/1000)
    megabase_sub= sum(data.chrom==args.chr_sub)*args.resolution/1000
    #print(megabase_sub)
    ndiff0 = max(int( megabase_sub* ndiff),1)
    ndiffs = " ".join([str(int(ndiff0*f)) for f in fexplo])
    #ndiffs = " ".join([str(int(ndiff0*f)) for f in np.arange(1,2,0.5)])


    directory_opti = args.root+f"/_RFD_to_init_small_opti_{loop}/"
    end = int(sum(data.chrom==args.chr_sub) * args.resolution) # in kb
    print(end)
    extra_small = ""
    if args.pearson:
        extra_small=" --pearson "
    if args.grid_rfd_opti_only:
        extra_small+=" --rfd_opti_only "

    cmd_opti = f"python src/repli1d/small_opty.py {extra_small} --size_segment {end/1000} --ndiff {ndiffs} --root {directory_opti}  --cmd '--kon {8.625/end} {small_sub} --start 0 --end {end} --ch {args.chr_sub} --nsim {nsim}  {cellcmd} "

    if loop == 0:
        if args.on_input_signal == None:
            if args.RFDonly:
                cmd_opti += "--signal peakRFDonly --dec 2 ' "
            else:
                cmd_opti += "--signal peak --dec 2' "
        else:
            cmd_opti += f"--signal {args.on_input_signal}' "


    else:
        cmd_opti += f"--signal {directory_nn}/nn_global_profiles.csv' --maxT {maxT} "

    cmd += [[cmd_opti,""]]

    directory = args.root+f"/wholecell_{loop}/"

    extra_params=""
    if args.save:
        extra_params += " --save "
    if args.single:
        extra_params += " --single "
    cmd_wholecell=f"python src/repli1d/detect_and_simulate.py {extra_params} {standard_parameters} --name {directory} --extra_param {directory_opti}/params.json "

    if loop == 0:
        if args.on_input_signal == None:
            if args.RFDonly:
                cmd_wholecell += "--signal peakRFDonly --dec 2 "
            else:
                cmd_wholecell += "--signal peak"
        else:
            cmd_wholecell += f"--signal {args.on_input_signal} "
    else:
        cmd_wholecell += f"--signal {directory_nn}/nn_global_profiles.csv"

    cmd += [ [cmd_wholecell,
             directory+"/global_profiles.csv"]]




redo = not args.update
exe= False
script = []
for cm in cmd:
    if args.show:
        print(cm)
    if type(cm) == list:
        sup=None
        if "global_profile" in cm[1]:
            sup = "python src/repli1d/average_expe.py --dir %s" % "/".join(cm[1].split("/")[:-1])
        if not redo and os.path.exists(cm[1]):
            if args.show:
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
            if args.show:
                print(sup)
            script.append(sup)
            if exe:
                process = subprocess.Popen(sup, shell=True, stdout=subprocess.PIPE)
                process.wait()
    else:
        script.append(cm)
        if exe:
            process = subprocess.Popen(cm, shell=True, stdout=subprocess.PIPE)
            process.wait()
with open(args.name_script,"w") as f:
    f.writelines("\n".join(script))
