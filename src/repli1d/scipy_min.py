from scipy.optimize import minimize
import numpy as np
import argparse
import pandas as pd
import subprocess
import os
import math

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--init', type=str, default="K562")
    parser.add_argument('--root', type=str, default="./results/scipy_opti/")
    parser.add_argument('--n', type=int, default=100)

    parser.add_argument('--command',type=str)
    parser.add_argument('--init_cmd',type=str)


    args = parser.parse_args()

    root = args.root
    os.makedirs(root,exist_ok=True)

    #Run initial simulation to generate parameter list
    name_init = root + "/init"
    process = subprocess.Popen(args.command +" "+ args.init_cmd + " --name %s" % name_init, shell=True,
                               stdout=subprocess.PIPE)

    process.wait()

    whole_info = pd.read_csv(name_init + "global_profiles.csv")
    x0 = np.array(whole_info.signal)
    init_x0 = x0.copy()

    x0[np.isnan(x0)] = 0
    where = x0> 1e-3

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def isigmoid(x):
        return np.log(x/(1-x))


    x0 = x0[where]
    x0 /= np.sum(x0)
    x0 = isigmoid(x0)
    command = args.command

    iter = 0
    gscore = 0
    Np = np.sum(where)
    print("Np",Np)

    def fun(x):
        global iter
        global gscore
        signal = init_x0
        signal[where] = sigmoid(x)
        #if np.sum(x<0)>0:
        #    return 2
        filen =  args.root +"/tmp.csv"
        d = pd.DataFrame({"chrom":whole_info.chrom,
                         "chromStart": whole_info.chromStart,
                         "chromEnd":whole_info.chromStart,
                          "signalValue":signal})
        d.to_csv(filen,index=False)
        process = subprocess.Popen(command + " --signal %s --name %s" %(filen,root+"/tmp"), shell=True, stdout=subprocess.PIPE)

        process.wait()

        score = pd.read_csv(root+"/tmpglobal_corre.csv")
        c1 = float(score["MRTp"][0].split(",")[0][1:])
        c1 = 1
        c2 = float(score["RFDp"][0].split(",")[0][1:])
        #c2 = 1-score["RFDstd"][0]


        if iter % 10 == 0:
            print("every10",c1,c2)

        score = 2-c1-c2 # + 0.01 * (np.sum(x)-1)**2

        if iter == 0:
            print("Initial value",gscore)
            gscore = score

        if score < gscore:
            print("New minimum %.3f , old %.3f",score,gscore)
            print(c1,c2)
            print(pd.read_csv(root+"/tmpglobal_corre.csv"))
            d.to_csv(root+"_%i.csv" % iter,index=False)
            gscore = score


        iter += 1


        return score

    ret = minimize(fun,x0=x0,method="SLSQP",options={"maxiter":200})#bounds=[(0,1) for n in list(range(Np)) ])  # SLSQP worked well for ch 13

    print(ret)