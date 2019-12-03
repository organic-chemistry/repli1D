import numpy as np
import os
from hyperopt import fmin, tpe, hp,Trials
import pandas as pd
from hyperopt.mongoexp import MongoTrials


import argparse

def score(repo,rfd=False):
    score = pd.read_csv("%s/bestglobal_scores.csv"%repo)
    #print(score["MRTp"][0])
    if not rfd:
        c1 = float(score["MRTp"][0].split(",")[0][1:])
        c2 = float(score["RFDp"][0].split(",")[0][1:])
        scorev = 2-c1-c2
    else:
        scorev = float(score["RFDp"][0].split(",")[0][1:])
    return scorev

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--mongo', action="store_true")
    parser.add_argument('--maxeval', type=int, default=100)
    parser.add_argument('--root', type=str, default="tmp")



    args = parser.parse_args()


    marks = ['H2az', 'H3k27ac', 'H3k79me2', 'H3k27me3', 'H3k9ac', 'H3k4me2', 'H3k4me3', 'H3k9me3', 'H3k4me1', 'H3k36me3'] + ["H4k20me1"]

    space = {m:hp.choice(m,[True,False]) for m in marks}


    #print(score("results/best/comb_1_5_30_test_K562"))
    def fun(space):
        print(space)
        repo = args.root+"/Hela"
        ms = ""
        listv = [m for m in marks if space[m]]
        print(space[marks[0]])


        ms = " ".join(listv)
        repo += "_".join(listv)
        if ms == "":
            return 2

        print(ms)
        repo += "/"

        if os.path.exists("%s/bestglobal_scores.csv"%repo):
            return score(repo)


        commands = ["mkdir -p %s"%repo,
                    "python src/repli1d/nn.py --rootnn %s --cell K562 --marks %s" % (repo,ms),
                    "python src/repli1d/detect_and_simulate.py --input --visu --signal %s/nn_K562_from_K562.csv --ndiff 45 --dori 1 --ch 1 --name %sbest --resolution 5  --nsim 400 --dec 2 --cell K562 --noise 0.1" %(repo,repo)
                    ]
        commands = ["mkdir -p %s"%repo,
                    "python src/repli1d/nn.py --rootnn %s --cell Hela --marks %s" % (repo,ms),
                    "python src/repli1d/detect_and_simulate.py --input --visu --signal %s/nn_K562_from_K562.csv --ndiff 60 --dori 1 --ch 1 --name %sbest --resolution 5  --nsim 400 --dec 2 --cell Helas3 --noise 0.1" %(repo,repo)
                    ]
        for command in commands:
            print(command)
            os.system(command)

        return score(repo)
    if args.mongo:
        trials = MongoTrials('mongo://localhost:1234/foo_db/jobs', exp_key='exp%i' % args.nsmooth)
    else:
        trials = Trials()
    best = fmin(fn=fun,
        space=space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=args.maxeval)
    pd.DataFrame(best,index=[0]).to_csv("best%i.csv"% args.nsmooth,index=False)

    print(best)
