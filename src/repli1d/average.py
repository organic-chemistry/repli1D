import numpy as np
import os
import pandas as pd
import glob

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

    parser.add_argument('--file', type=str, default=None)
    parser.add_argument('--dir',type=str, default=None)

    args = parser.parse_args()

    if args.dir != None:
        D = []
        for dir in glob.glob(args.dir + "/*"):
            fich = dir +"/wholeglobal_scores.csv"
            if os.path.exists(fich):
                data = pd.read_csv(fich)
                data["MRTp"] = [float(d.split(",")[0][1:]) for d in data["MRTp"]]
                data["RFDp"] = [float(d.split(",")[0][1:]) for d in data["RFDp"]]
                data["Cumulstd"] = data["MRTstd"] + data["RFDstd"]

            else:
                continue
            sd = {}
            #print(data.columns)
            #, 'Dori', 'MRTkeep', 'MRTp', 'MRTstd', 'RFDkeep', 'RFDp',
            #'RFDstd', 'Random', 'RepTime', 'ch', , 'marks'

            for k in ['Diff','lenMRT', 'lenRFD','MRTkeep', 'RFDkeep']:
                sd[k] = data.sum()[k]

            for k in ["marks",'Random',"Dori"]:
                sd[k] = data[k][0]
            for k in ["MRTp","RFDp","RepTime","MRTstd","RFDstd","Cumulstd"]:
                sd[k] = data.mean()[k]

            if "csv" in sd["marks"]:
                sd["marks"] = sd["marks"].split("/")[-1]
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






            D.append(sd)



        D = pd.DataFrame(D)
        D = D.sort_values("Cumulstd",ascending=True)


        D.to_csv(args.dir + "/summary.csv",index=False)
        cell = args.dir.split("/")[1].split("_")[0]
        print(cell)
        D.to_csv(args.dir + "/%ssummary.csv"%cell,index=False)


        print(D)

    else:

        data = pd.read_csv(args.file)
        print(data.head())
        data["MRTp"] = [float(d.split(",")[0][1:]) for d in data["MRTp"]]
        data["RFDp"] = [float(d.split(",")[0][1:]) for d in data["RFDp"]]

        print(data[["MRTp","RFDp"]].mean())
