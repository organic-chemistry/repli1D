import numpy as np
import os
from hyperopt import fmin, tpe, hp,Trials
import pandas as pd
from hyperopt.mongoexp import MongoTrials


import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--lsim', type=str, default=None)
    parser.add_argument('--root', type=str, default="tmp")


    args = parser.parse_args()
    extra = ""
    if "Hela" in args.lsim:
        cell = "Helas3"
        name0= "Hela"
    if "GM" in args.lsim:
        cell = "Gm12878"
        extra = "--comp GM12878 --cellseq GM06990"
        name0= "GM"
    if "K562" in args.lsim:
        cell = "K562"
        name0 = "K562"

    lsim = pd.read_csv(args.lsim)

    name1 ="%s/%s_whole" % (args.root,name0)

    for mark,ndiff,noise,dori in zip(lsim.marks,lsim.Diff,lsim.Random,lsim.Dori):
        name = "%s/%s/" % (name1,mark)

        if mark == "combi":
            continue
        if mark == "combiMS":
            mark = "results/best/comb_8_14_%s/%s_ms.weight" % (name0,name0)
        ndiff = ndiff / (120-5)
        commands = ["mkdir -p %s"%name,
                    "python src/repli1d/detect_and_simulate.py --input --visu --signal %s --ndiff %.2f --dori %i --ch 1 --name %s"
                    " --resolution 5  --nsim 200 --dec 2 --cell %s --noise %.2f %s --wholecell" %(mark,
                                                                                                  ndiff,
                                                                                                dori,
                                                                                                name+"/whole",
                                                                                                cell,
                                                                                                noise,
                                                                                                extra)
                    ]
        for command in commands:
            print(command)
            #os.system(command)
