import os
import subprocess
import numpy as np

cell = {"K562":"E123",
        "GM12878" :  "E116" ,
        "Hela"  :   "E115" ,
        "IMR90" :   "E017"}

marks = [ "H3K4me1", "H3K4me3", "H3K27me3", "H3K36me3", "H3K9me3"]

root = "/mnt/data/data/optiyeast/"
import time

local_root = root
os.makedirs(local_root,exist_ok=True)
for mark in marks:
    # E123-H3K36me3.tagAlign
    dori = 5
    for ndiff in [10]:
        for random in  np.arange(0.,0.41,0.05):#[0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.20]:
            dir  = local_root +"/nd_%.2f_r_%.2f_%.2f/res" %(ndiff,random,dori)
            if os.path.exists(dir+"global_scores.csv"):
                continue
            cmd = ["python src/repli1d/detect_and_simulate.py --visu   --dori 1  --ndiff %.2f --noise %.2f  --cell Yeast-MCM --name %s  --signal ARSpeak   --start 0  --ch 4   --nsim 200 --resolution 1 --end 1200 --input  --experimental  --resolutionpol 1 --exp_factor 5 --fspeed 1.5 --kon 0.1 --percentile 55 --wholecell  --forkseq --n_job 1"%(ndiff,random,dir)]
    #cmd +=
            # ["rm %s"%compressed[:-3]]
    #print(cmd)
            for cm in cmd:
                print(cm)

                process = subprocess.Popen(cm, shell=True, stdout=subprocess.PIPE)
                process.wait()
        #os.popen(cm)
    #break
    #time.sleep(10*60)

#break
    #c

print("End")
