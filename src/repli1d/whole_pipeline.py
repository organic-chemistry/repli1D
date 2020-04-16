import argparse
import subprocess
import os

parser = argparse.ArgumentParser()

parser.add_argument('--cell', type=str, default="K562")
parser.add_argument('--root', type=str, default="./")
parser.add_argument('--update', action="store_true")

args = parser.parse_args()


standard_parameters = "--cutholes 1500 --single --visu --experimental --wholecell --input  --nsim 200 --dori 5  --noise 0.05 --kon 5e-7 "
standard_parameters = "--cutholes 1500 --single --visu --experimental --wholecell --input  --nsim 200 --dori 5  --noise 0.05 --kon 1e-5 "

cell = args.cell

if cell == "K562":
    ndiff = 0.6
    cellcmd = "--cell K562"
elif cell  == "GM":
    ndiff = 0.6
    cellcmd = "--cell Gm12878 --comp GM12878 --cellseq GM06990"
elif cell == "Hela":
    ndiff = 1.1
    cellcmd = "--cell Helas3"

standard_parameters += "--ndiff %.2f %s"%(ndiff,cellcmd)

#run simulation with peak
directory = args.root+"/%s_peak_wholecell/" % cell
#cmd = [ ["python src/repli1d/detect_and_simulate.py %s --name %s --signal peak --dec 2 "%(standard_parameters,directory),
#         directory+"/global_profiles.csv"]]
cmd = []
"""
#run simulation with peak and random large
directory = args.root+"/%s_peak_random_large_wholecell/" % cell
cmd += [ ["python src/repli1d/detect_and_simulate.py %s --name %s --signal peak --dec 2 --randomlarge "%(standard_parameters,directory),
         directory+"/global_profiles.csv"]]

#run nn.py on peak to predict peak from RFD MRT  # now only on the cell, should als do that on original simu
directory_nn = args.root+"/%s_RFD_to_Init_nn/" % cell
cmd += [["python src/repli1d/nn.py --listfile %s/global_profiles.csv --marks RFDs MRTs --root %s --sm 10  --enrichment 0.1" % (directory,directory_nn),
         directory_nn+"/nn_global_profiles.csv"]]

#run simulation with predicted peak from nn
directory = args.root+"/%s_RFD_to_Init_wholecell/" % cell
cmd += [["python src/repli1d/detect_and_simulate.py %s --name %s --signal %s/nn_global_profiles.csv  "%(standard_parameters,directory,directory_nn),
         directory+"/global_profiles.csv"]]
"""
directory_ml = "/home/jarbona/repli1D/data/mlformat_whole_pipe_%s.csv" % cell
# Generate file to do nn (must contain initiation and epi)

#cmd += [["python src/repli1d/nn_create_input.py --peak %s/nn_global_profiles.csv  --cell %s --outfile %s "%(directory_nn,cell,directory_ml),
#        directory_ml]]

#run nn.py on peaks to predict from epigenetic marks
extra = "/home/jarbona/repli1D/data/mlformat_whole_pipe_GM.csv /home/jarbona/repli1D/data/mlformat_whole_pipe_Hela.csv"
directory_nn = args.root+"/%s_Epi_nn/" % cell
file = "/nn_%s_from_%s.csv" % (cell,"None") # to check for output
cmd += [["python src/repli1d/nn.py   --targets initiation --root %s --listfile %s  --window 51 --wig 1 --predict_files %s %s" % (directory_nn,directory_ml,directory_ml,extra),
        directory+file]]

directory = args.root+"/%s_Epi_from_%s_wholecell/" % (cell,cell)
#run simulation on epigenetic marks (and for other cell lines)
cmd += [["python src/repli1d/detect_and_simulate.py %s --name %s --signal %s/%s   "%(standard_parameters,directory,directory_nn,file),
        directory +"/global_profiles.csv"]]


#run nn.py on peaks to predict from epigenetic marks (Bigger net)
directory_ml = "/home/jarbona/repli1D/data/mlformat_whole_pipe_%s.csv" % cell
directory_nn_bigger = args.root+"/%s_Epi_nn_bigger/" % cell
file = "/nn_%s_from_%s.csv" % (cell,"None") # to check for output
cmd += [["python src/repli1d/nn.py   --targets initiation --root %s --listfile %s  --window 101 --kernel_length 20 --nfilters 30 --enrichment 0.1 --wig 1 --predict_files %s %s" % (directory_nn_bigger,directory_ml,directory_ml,extra),
        directory+file]]

directory = args.root+"/%s_Epi_bigger_from_%s_wholecell/" % (cell,cell)
#run simulation on epigenetic marks (and for other cell lines)
cmd += [["python src/repli1d/detect_and_simulate.py %s --name %s --signal %s/%s   "%(standard_parameters,directory,directory_nn_bigger,file),
        directory +"/global_profiles.csv"]]




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