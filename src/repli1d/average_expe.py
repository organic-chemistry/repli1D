import numpy as np
import os
import pandas as pd
import glob

import argparse


def score(repo, rfd=False):
    score = pd.read_csv("%s/bestglobal_scores.csv" % repo)
    # print(score["MRTp"][0])
    if not rfd:
        c1 = float(score["MRTp"][0].split(",")[0][1:])
        c2 = float(score["RFDp"][0].split(",")[0][1:])
        scorev = 2-c1-c2
    else:
        scorev = float(score["RFDp"][0].split(",")[0][1:])
    return scorev


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--file', type=str, default=None)
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--extra', type=str, default="")
    parser.add_argument('--dirs', type=str, default=None)

    args = parser.parse_args()

    D = []
    if args.dirs is not None:
        dirs = glob.glob(args.dirs + "/*")
    else:
        dirs = [args.dir]
    print(dirs)
    for dir in dirs:
        extra = args.extra
        fich = dir + "/%sglobal_scores.csv" % extra
        print("HCH",fich)
        if os.path.exists(fich):
            data = pd.read_csv(fich)
            data["MRTp"] = [float(d.split(",")[0][1:]) for d in data["MRTp"]]
            data["RFDp"] = [float(d.split(",")[0][1:]) for d in data["RFDp"]]
            data["Cumulstd"] = data["MRTstd"] + data["RFDstd"]
            data["Cumulp"] = data["MRTp"] + data["RFDp"]

        else:
            continue
        sd = {}
        # print(data.columns)
        # , 'Dori', 'MRTkeep', 'MRTp', 'MRTstd', 'RFDkeep', 'RFDp',
        # 'RFDstd', 'Random', 'RepTime', 'ch', , 'marks'

        for k in ['Diff', 'lenMRT', 'lenRFD', 'MRTkeep', 'RFDkeep']:
            sd[k] = data.sum()[k]

        for k in ["marks", 'Random', "Dori","NactiOri","Realinterdist"]:
            if k in data.columns:

                sd[k] = data[k][0]
        SingleFiber = ["Fiber200_percent_0_Forks","Fiber200_percent_1_Forks",
                       "Fiber200_percent_2_Forks","Fiber200_percent_3_Forks",
                       "Fiber200_percent_4_Forks ",	"Nforkperfibrwithfork","codire"]
        for k in ["MRTp", "RFDp", "RepTime", "MRTstd", "RFDstd", "Cumulstd","Cumulp"] + SingleFiber:
            if k in data.columns:
                sd[k] = data.mean()[k]

        if "csv" in sd["marks"]:
            sd["marks"] = sd["marks"].split("/")[-1][:-4]
            print("Changing name")
        if "weight" in sd["marks"]:
            sd["marks"] = sd["marks"].split("/")[-1]
            print("Changing name")
        if sd["marks"] == "nn_hela_fk.csv":
            sd["marks"] = "nn_Hela_from_K562.csv"
            print("Changing name")
        if sd["marks"] == "nn_gm_fk.csv":
            sd["marks"] = "nn_GM_from_K562"
            print("Changing name")
        if sd["marks"] == "nn_fk.csv":
            sd["marks"] = "nn_K562_from_K562"
            print("Changing name")

        # Compute deltas:
        strain = pd.read_csv(dir + "/%sglobal_profiles.csv"%extra, sep=",")

        Exp = strain.RFDe
        Sim = strain.RFDs
        #hist(Exp-Sim, bins=50, histtype="step")
        th = 1
        sd["delta_th_1"] = (np.sum(Exp-Sim > th)+np.sum(Exp-Sim < -th))
        th = 0.5
        sd["delta_th_0.5"] = (np.sum(Exp-Sim > th)+np.sum(Exp-Sim < -th))
        D.append(sd)

    D = pd.DataFrame(D)
    D = D.sort_values("Cumulp", ascending=False)
    write = args.dirs
    if args.dir is not None:
        write = args.dir
    D.to_csv(write + "/summary.csv", index=False)
    #cell = args.dir.split("/")[1].split("_")[0]
    # print(cell)
    #D.to_csv(args.dir + "/%ssummary.csv" % cell, index=False)

    print(D)
