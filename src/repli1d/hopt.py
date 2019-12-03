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

    parser.add_argument('--nsmooth', type=int, default=1)
    parser.add_argument('--mongo', action="store_true")
    parser.add_argument('--maxeval', type=int, default=100)
    parser.add_argument('--root', type=str, default="tmp")



    args = parser.parse_args()


    if args.nsmooth == 1:
        space = {"sm0":hp.choice("sm0",[None,hp.quniform('v0', 1, 15,1)]),

                }

    if args.nsmooth == 2:
        space = {"sm0":hp.choice("sm0",[None,hp.quniform('v0', 1, 45,1)]),
                 "sm1":hp.choice("sm1",[None,hp.quniform('v1', 5, 50,1)])
                }
    if args.nsmooth == 3:
        space = {"sm0":hp.choice("sm0",[None,hp.quniform('v0', 1, 25,1)]),
                 "sm1":hp.choice("sm1",[None,hp.quniform('v1', 5, 35,1)]),
                 "sm2":hp.choice("sm2",[None,hp.quniform('v2', 15, 50,1)])
                }
    if args.nsmooth == 4:
        space = {"sm0":hp.choice("sm0",[None,hp.quniform('v0', 1, 15,1)]),
                 "sm1":hp.choice("sm1",[None,hp.quniform('v1', 5, 25,1)]),
                 "sm2":hp.choice("sm2",[None,hp.quniform('v2', 15, 35,1)]),
                 "sm3":hp.choice("sm3",[None,hp.quniform('v3', 25, 50,1)])
                }
    if args.nsmooth == 5:
        space = {"sm0":hp.choice("sm0",[None,hp.quniform('v0', 1, 15,1)]),
                 "sm1":hp.choice("sm1",[None,hp.quniform('v1', 5, 25,1)]),
                 "sm2":hp.choice("sm2",[None,hp.quniform('v2', 15, 35,1)]),
                 "sm3":hp.choice("sm3",[None,hp.quniform('v3', 25, 45,1)]),
                 "sm4":hp.choice("sm4",[None,hp.quniform('v4', 35, 50,1)])
                }





    #print(score("results/best/comb_1_5_30_test_K562"))
    def fun(space):
        print(space)
        exclude = "H3k9me1 H4k20me1"
        repo = args.root+"/Hela"
        ms = ""
        listv = [space.get(v,None) for v in ["sm0","sm1","sm2","sm3","sm4"]]
        listv = [v for v in listv if v is not None]
        listv.sort()

        for v in listv:
            ms += " %i" % v
            repo += "_%i" % v
        if ms == "":
            return 2

        print(ms)
        repo += "/"
        
        if os.path.exists("%s/bestglobal_scores.csv"%repo):
            return score(repo)


        commands = ["mkdir -p %s"%repo,
                    "python src/repli1d/detect.py --dec 2 --resolution 5 --name %sHeladec2.peak --percentile 85 --cell Hela --smoothpeak 5"%(repo),
                    "python src/repli1d/retrieve_marks.py --signal %sHeladec2.peak --wig --name-weight %sHela_ms.weight  --ms %s --exclude  %s --cell Helas3  --constant" %(repo,repo,ms,exclude),
                    "python src/repli1d/detect_and_simulate.py --input --visu --signal %sHela_ms.weight --ndiff 75 --dori 1 --ch 1 --name %sbest --resolution 5  --nsim 400 --dec 2 --cell Helas3 --noise 0.2" %(repo,repo)
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
