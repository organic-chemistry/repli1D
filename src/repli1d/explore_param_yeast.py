import subprocess
import numpy as np
import pandas as pd
root ="results/yeast-opti-fork/"
for ndiff in range(5,20):
    for noise in np.arange(0,0.3,0.025):
        command = "python src/repli1d/detect_and_simulate.py --visu   --dori 0.1   --cell Yeast-MCM  --signal ARSpeak   --start 0  --ch 4   --nsim 200 --resolution 1 --end 1200 --input  --experimental  --resolutionpol 1 --exp_factor 8 --fspeed 1.5 --kon 0.1 --percentile 55 --wholecell   --n_job 6 --nMRT 10 --smoothpeak 3"
        name = "%syeast_%i_%.3f/" %(root,ndiff,noise)
        command += " --forkseq --ndiff %i --noise %.3f --name %s " %(ndiff,noise,name)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)

        process.wait()

        scored = pd.read_csv(name+"global_corre.csv")
        print(ndiff,noise)
        print(scored)