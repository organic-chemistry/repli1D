import argparse
import subprocess
import os

parser = argparse.ArgumentParser()

parser.add_argument('--cell', type=str, default="K562")
parser.add_argument('--root', type=str, default="./")
parser.add_argument('--update', action="store_true")

args = parser.parse_args()


standard_parameters = "--cutholes 1500 --single --visu --experimental --wholecell --input  --nsim 200 --dori 5  --noise 0.05 --kon 5e-7 "
standard_parameters0 = "--cutholes 1500 --single --visu --experimental --wholecell --input  --nsim 200 --dori 5  --noise 0.05 --kon 1e-5 "

cell = args.cell

if cell == "K562":
    ndiff = 0.4
    cellcmd = "--cell K562"
elif cell  == "GM":
    ndiff = 0.6
    cellcmd = "--cell Gm12878 --comp GM12878 --cellseq GM06990"
elif cell == "Hela":
    ndiff = 1.1
    cellcmd = "--cell Helas3"

standard_parameters = standard_parameters0 + "--ndiff %.2f %s"%(ndiff,cellcmd)

#run simulation with peak
directory = args.root+"/%s_peak_wholecell/" % cell

directory = args.root+"/%s_peak_DNase_wholecell/" % cell
cmd = [ ["python src/repli1d/detect_and_simulate.py %s --name %s --signal DNaseI --dec 2 "%(standard_parameters,directory),
         directory+"/global_profiles.csv"]]
#run simulation with peak and random large
directory = args.root+"/%s_DNaseIRandom_wholecell/" % cell
cmd += [ ["python src/repli1d/detect_and_simulate.py %s --name %s --signal DNaseIRandom --dec 2  "%(standard_parameters,directory),
         directory+"/global_profiles.csv"]]

directory = args.root+"/%s_DNaseIFlat_wholecell/" % cell
cmd += [ ["python src/repli1d/detect_and_simulate.py %s --name %s --signal DNaseIFlat --dec 2  "%(standard_parameters,directory),
         directory+"/global_profiles.csv"]]
"""

directory_nn = args.root+"/%s_RFD_to_init_nn/" % cell
cmd += [["python src/repli1d/nn.py --listfile %s/global_profiles.csv --marks RFDs MRTs --root %s --sm 10  --enrichment 0.1" % (directory,directory_nn),
         directory_nn+"/nn_global_profiles.csv"]]
"""

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

with open("script_%s.sh" % cell,"w") as f:
    f.writelines("\n".join(script))