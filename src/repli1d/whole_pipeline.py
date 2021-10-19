import argparse
import subprocess
import os
import json

parser = argparse.ArgumentParser()

parser.add_argument('--cell', type=str, default="K562")
parser.add_argument('--root', type=str, default="./")
parser.add_argument('--update', action="store_true")
parser.add_argument('--start', default="rfdpeak",choices=["rfdpeak","random","DNaseI"])
parser.add_argument('--maxT', action="store_true")
parser.add_argument('--savenoise', action="store_true")
parser.add_argument('--add_noise', action="store_true")
parser.add_argument('--filter_anomaly', action="store_true")
parser.add_argument('--rfd_opti_only', action="store_true")

parser.add_argument('--n_jobs',type=int,default=8)



args = parser.parse_args()


#standard_parameters = " --cutholes 1500 --single --visu --experimental --wholecell --input  --nsim 200 --dori 5  --noise 0.0 --kon 5e-7 "
small_sub= f"--cutholes 1500 --single --visu --experimental --input --introduction_time 60 --n_jobs {args.n_jobs}"
standard_parameters = small_sub + " --wholecell --kon 1e-5 --noise 0.0 --nsim 200  --dori 5 "
if not args.savenoise:
    standard_parameters +=  " --exclude_noise_save "

cell = args.cell

if cell == "K562":
    ndiff = 0.6
    cellcmd = "--cell K562"
    maxT = 12 * 60 * 1.1
elif cell  == "GM":
    ndiff = 0.6
    cellcmd = "--cell Gm12878 --comp GM12878 --cellseq GM06990"
    maxT = 12 * 60 * 1.1
elif cell == "Raji":
    ndiff = 0.6
    cellcmd = "--cell Raji"
elif cell == "Hela":
    ndiff = 1.1
    cellcmd = "--cell Helas3"
    maxT = 8 * 60 * 1.1

#maxT *= 100
standard_parameters += "--ndiff %.2f %s"%(ndiff,cellcmd)

#run simulation with peak
"""
directory = args.root+"/%s_peak_wholecell/" % cell
cmd = [ ["python src/repli1d/detect_and_simulate.py %s --name %s --signal peak --dec 2 "%(standard_parameters,directory),
         directory+"/global_profiles.csv"]]

#run simulation with peak and random large
"""

"""
directory = args.root+"/%s_peakRFD_no_random_large_wholecell/" % cell
cmd += [ ["python src/repli1d/detect_and_simulate.py %s --name %s --signal peakRFDonly --dec 2  --noise 0.05 --extra_param %s/params.json "%(standard_parameters,
                                                                                                    directory,directory_opti),
         directory+"/global_profiles.csv"]]
"""
if args.add_noise:
    extra_nn=" --add_noise "
else:
    extra_nn=" "

if args.filter_anomaly:
    extra_nn+=" --filter_anomaly "

if args.rfd_opti_only:
    sub_cmd=" --rfd_opti_only "
else:
    sub_cmd="  "


if args.start=="random":
    directory_opti = args.root+"/%s_RFD_to_init_small_opti_0/" % cell
    cmd = [[f"python src/repli1d/small_opty.py --root {directory_opti} {sub_cmd} --cmd ' --randomprofile 1 {small_sub} --ch 2 --nsim 200 --dori 10 {cellcmd} --signal peakRFDonly' ",
             directory_opti+"/global_profiles.csv"]]
    directory = args.root+"/%s_peakRFD_random_wholecell/" % cell
    cmd += [ ["python src/repli1d/detect_and_simulate.py --randomprofile 1 %s --name %s --signal peakRFDonly --dec 2  --noise 0.05 --extra_param %s/params.json "%(standard_parameters,
                                                                                                        directory,directory_opti),
             directory+"/global_profiles.csv"]]
if args.start=="rfdpeak":
    directory_opti = args.root+"/%s_RFD_to_init_small_opti_0/" % cell
    cmd = [[f"python src/repli1d/small_opty.py --root {directory_opti} {sub_cmd} --cmd '  {small_sub} --ch 2 --nsim 200 --dori 10 {cellcmd} --signal peakRFDonly' ",
             directory_opti+"/global_profiles.csv"]]
    directory = args.root+"/%s_peakRFD_random_wholecell/" % cell
    cmd += [ ["python src/repli1d/detect_and_simulate.py  %s --name %s --signal peakRFDonly --dec 2  --noise 0.05 --extra_param %s/params.json "%(standard_parameters,
                                                                                                        directory,directory_opti),
             directory+"/global_profiles.csv"]]
if args.start=="DNaseI":
    directory_opti = args.root+"/%s_RFD_to_init_small_opti_0/" % cell
    cmd = [[f"python src/repli1d/small_opty.py --root {directory_opti}  {sub_cmd} --cmd '  {small_sub} --ch 2 --nsim 200 --dori 10 {cellcmd} --signal DNaseI' ",
             directory_opti+"/global_profiles.csv"]]
    directory = args.root+"/%s_peakRFD_random_wholecell/" % cell
    cmd += [ ["python src/repli1d/detect_and_simulate.py  %s --name %s --signal DNaseI --dec 2  --noise 0.05 --extra_param %s/params.json "%(standard_parameters,
                                                                                                        directory,directory_opti),
             directory+"/global_profiles.csv"]]

if args.cell == "Hela":
    rp=1.2
else:
    rp=0.8
# Create simulation with random profile
"""
directory = args.root+"/%s_peak_random_profile_wholecell/" % cell
cmd = [ ["python src/repli1d/detect_and_simulate.py %s --name %s --signal peak --dec 2 --randomprofile %.1f --noise 0.01 "%(standard_parameters,
                                                                                                                            directory,rp),
         directory+"/global_profiles.csv"]]

"""
if args.maxT:
    extra= f" --maxT {maxT} "
else:
    extra=""

# run nn.py on random profile to predict peak from RFD MRT  # now only on the cell, should als do that on original simu
directory_nn = args.root+"/%s_RFD_to_init_nn/" % cell
cmd += [[f"python src/repli1d/nn.py --listfile {directory}/global_profiles.csv --marks RFDs MRTs --root {directory_nn} --sm 10  --noenrichment --window 401 {extra_nn}",
         directory_nn+"/nn_global_profiles.csv"]]

#small opti
directory_opti = args.root+"/%s_RFD_to_init_small_opti_1/" % cell
cmd += [[f"python src/repli1d/small_opty.py {extra} --root {directory_opti} {sub_cmd} --cmd '{small_sub} --ch 2 --nsim 200 --dori 10 {cellcmd} --signal {directory_nn}/nn_global_profiles.csv' ",
         directory+"/global_profiles.csv"]]
# load parameters

# run simulation with predicted peak from nn
directory = args.root+"/%s_RFD_to_init_wholecell/" % cell
cmd += [["python src/repli1d/detect_and_simulate.py %s --name %s --signal %s/nn_global_profiles.csv --noise 0 --extra_param %s/params.json "%(standard_parameters,
                                                                                                                                directory,
                                                                                                                                directory_nn,directory_opti),
         directory+"/global_profiles.csv"]]


# run nn.py on simulation from nn to improove to predict peak from RFD MRT  # now only on the cell, should als do that on original simu
directory_nn = args.root+"/%s_RFD_to_init_nn2/" % cell
directory_nn2 = "" + directory_nn
cmd += [[f"python src/repli1d/nn.py --listfile {directory}/global_profiles.csv --marks RFDs MRTs --root {directory_nn} --sm 10  --noenrichment --window 401 {extra_nn}",
         directory_nn+"/nn_global_profiles.csv"]]

directory_opti = args.root+"/%s_RFD_to_init_small_opti_2/" % cell
cmd += [[f"python src/repli1d/small_opty.py {extra} --root {directory_opti}  {sub_cmd} --cmd '{small_sub} --ch 2 --nsim 200 --dori 10 {cellcmd} --signal {directory_nn}/nn_global_profiles.csv' ",
         directory+"/global_profiles.csv"]]

# run simulation with predicted peak from nn2
directory = args.root+"/%s_RFD_to_init_wholecell2/" % cell
cmd += [["python src/repli1d/detect_and_simulate.py %s --name %s --signal %s/nn_global_profiles.csv --noise 0 --extra_param %s/params.json --save "%(standard_parameters,
                                                                                                                                               directory,
                                                                                                                                               directory_nn,
                                                                                                                                               directory_opti),
         directory+"/global_profiles.csv"]]

directory_nn = args.root+"/%s_RFD_to_init_nn3/" % cell
extra_ml_pred = "/home/jarbona/repli1D/data/mlformat_whole_mrtrfd_GM.csv /home/jarbona/repli1D/data/mlformat_whole_mrtrfd_Hela.csv /home/jarbona/repli1D/data/mlformat_whole_mrtrfd_K562.csv"

cmd += [[f"python src/repli1d/nn.py --listfile {directory}/global_profiles.csv --marks RFDs MRTs --root {directory_nn} --sm 10  --noenrichment --window 401 {extra_nn} --predict_file {extra_ml_pred}",
         directory_nn+"/nn_global_profiles.csv"]]

# run simulation with predicted peak from nn2
directory_opti = args.root+"/%s_RFD_to_init_small_opti_3/" % cell
cmd += [[f"python src/repli1d/small_opty.py {extra} --root {directory_opti} {sub_cmd} --cmd '{small_sub} --ch 2 --nsim 200 --dori 10 {cellcmd} --signal {directory_nn}/nn_{cell}_from_None.csv' ",
         directory+"/global_profiles.csv"]]

directory = args.root+"/%s_RFD_to_init_wholecell3/" % cell
cmd += [["python src/repli1d/detect_and_simulate.py %s --name %s --signal %s/nn_%s_from_None.csv --noise 0 --save --extra_param %s/params.json"%(standard_parameters,
                                                                                                                                              directory,
                                                                                                                                              directory_nn,
                                                                                                                                              cell,
                                                                                                                                              directory_opti),
         directory+"/global_profiles.csv"]]
if args.savenoise:
    directory = args.root+"/%s_RFD_to_init_wholecell3_no_noise/" % cell
    cmd += [["python src/repli1d/detect_and_simulate.py --no_noise %s --name %s --signal %s/nn_%s_from_None.csv --noise 0 --save --extra_param %s/params.json"%(standard_parameters,
                                                                                                                                                  directory,
                                                                                                                                                  directory_nn,
                                                                                                                                                  cell,
                                                                                                                                                  directory_opti),
             directory+"/global_profiles.csv"]]

directory_ml = "/home/jarbona/repli1D/data/mlformat_whole_pipe_%s.csv" % cell


# Select the highest correlation
cmd += [[f"python src/repli1d/get_highest_correlation.py --root {args.root}",
        directory_ml]]


# Generate file to do nn (must contain initiation and epi)
#cmd += [["python src/repli1d/nn_create_input.py --peak %s/highest_correlation.csv  --cell %s --outfile %s "%(args.root,cell,directory_ml),#
#        directory_ml]]
"""
marks = ['H2az', 'H3k27ac', 'H3k79me2', 'H3k27me3', 'H3k9ac',
                 'H3k4me2', 'H3k4me3', 'H3k9me3', 'H3k4me1', 'H3k36me3', "H4k20me1"]
marks += ["RNA_seq","DNaseI"]
marks = " ".join(marks)
add = "_plus_AT/"
add="_RNA_seq_AT/"
add="_all/"
#run nn.py on peaks to predict from epigenetic marks
directory_nn = args.root+"/%s_Epi_nn" % cell
directory_nn += add
file = "/nn_%s_from_%s.csv" % (cell,"None") # to check for output
cmd += [["python src/repli1d/nn.py  --noenrichment --targets initiation --root %s --listfile %s  --window 51 --wig 1 --predict_files %s %s --marks %s" % (directory_nn,
                                                                                                                                                          directory_ml,
                                                                                                                                                          directory_ml,extra,marks),
        directory+file]]


directory_opti = args.root+"/%s_RFD_to_init_small_opti_4/" % cell
cmd += [[f"python src/repli1d/small_opty.py --root {directory_opti} --cmd '{small_sub} --ch 2 --nsim 200 --dori 10 {cellcmd} --signal {directory_nn}/{file}' ",
         directory+"/global_profiles.csv"]]

directory = args.root+"/%s_Epi_from_%s_wholecell" % (cell,cell)
directory += add
#run simulation on epigenetic marks (and for other cell lines)
cmd += [["python src/repli1d/detect_and_simulate.py %s --name %s --signal %s/%s --save --extra_param %s/params.json  "%(standard_parameters,
                                                                                                                  directory,
                                                                                                                  directory_nn,
                                                                                                                  file,directory_opti),
        directory +"/global_profiles.csv"]]


#run nn.py on peaks to predict from epigenetic marks (Bigger net)
directory_nn_bigger = args.root+"/%s_Epi_smaller_dropout_enrichment" % cell
directory_nn_bigger += add
file = "/nn_%s_from_%s.csv" % (cell,"None") # to check for output
#Before double filter and kernel
cmd += [["python src/repli1d/nn.py   --targets initiation --root %s --listfile %s   --wig 1 --predict_files %s %s --marks %s" % (directory_nn_bigger,directory_ml,directory_ml,extra,marks),
        directory+file]]


directory_opti = args.root+"/%s_RFD_to_init_small_opti_12/" % cell
cmd += [[f"python src/repli1d/small_opty.py --root {directory_opti} --maxT {maxT} --cmd '{small_sub} --ch 2 --nsim 200 --dori 10 {cellcmd} --signal {directory_nn_bigger}/{file}' ",
         directory+"/global_profiles.csv"]]



directory = args.root+"/%s_Epi_smaller_dropout_enrichment_from_%s_wholecell" % (cell,cell)
directory += add
#run simulation on epigenetic marks (and for other cell lines)
cmd += [["python src/repli1d/detect_and_simulate.py %s --name %s --signal %s/%s  --extra_param %s/params.json "%(standard_parameters,
                                                                                                                 directory,
                                                                                                                 directory_nn_bigger,
                                                                                                                 file,directory_opti),
        directory +"/global_profiles.csv"]]

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
