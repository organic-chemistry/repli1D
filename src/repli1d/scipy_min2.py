from scipy.optimize import minimize
import numpy as np
import argparse
import pandas as pd
import subprocess
import os
from repli1d.analyse_RFD import smooth


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--init', type=str, default="K562")
    parser.add_argument('--alpha', type=float,default=0.1)

    parser.add_argument('--root', type=str, default="./results/scipy_opti/")
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--extension', type=int, default=5)

    parser.add_argument('--command',type=str)


    args = parser.parse_args()

    root = args.root
    os.makedirs(root,exist_ok=True)

    whole_info = pd.read_csv(args.init)
    x0 = np.array(whole_info.signal)
    init_x0 = x0.copy()

    x0[np.isnan(x0)] = 0
    where = np.where(x0 != 0)

    x0 = x0[where]
    x0 /= np.sum(x0)
    command = args.command

    iter = 0
    gscore = 0


    def fun(x, alpha):
        global iter
        global gscore
        signal = init_x0
        signal[where] = x
        if np.sum(x < 0) > 0:
            return 2
        filen = root + "/tmp.csv"
        d = pd.DataFrame({"chrom": whole_info.chrom,
                          "chromStart": whole_info.chromStart,
                          "chromEnd": whole_info.chromStart,
                          "signalValue": signal})
        d.to_csv(filen, index=False)
        process = subprocess.Popen(command + " --signal %s --name %s" % (filen, root + "/tmp"), shell=True,
                                   stdout=subprocess.PIPE)

        process.wait()

        scored = pd.read_csv(root + "/tmpglobal_corre.csv")
        c1 = float(scored["MRTp"][0].split(",")[0][1:])
        c1 = 0
        c2 = float(scored["RFDp"][0].split(",")[0][1:])

        print(scored)

        if iter % 10 == 0:
            print("every10", c1, c2)

        score = 2 - c1 - c2  # + 0.01 * (np.sum(x)-1)**2

        if iter == 0:
            print("Initial value", gscore)
            gscore = score

        if score < gscore:
            print("New minimum %.3f , old %.3f", score, gscore)
            print(c1, c2)
            d.to_csv(root + "_%i.csv" % iter, index=False)
            gscore = score

        iter += 1

        scored = pd.read_csv(root + "/tmpglobal_profiles.csv")

        def delta(s):
            return np.array(s)[1:] - np.array(s)[:-1]

        deltas = smooth(delta(scored["RFDs"]) - delta(scored["RFDe"]), args.extension)

        direction = deltas[where]
        direction /= np.mean(np.abs(direction))

        x -= alpha * direction * x
        x[x < 0] = 0

        return score, x

    #ret = minimize(fun,x0=x0,method='Nelder-Mead',options={"maxiter":200})

    x = x0
    for i in range(args.n):
        score, x = fun(x, alpha=args.alpha)
        print(i, score)
