import pickle
#from .retrieve_marks import norm2
import os
import _pickle as cPickle
#import cache
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
import glob
import pprint
from scipy.signal import find_peaks

try:
    import pyBigWig
    import gffutils
except:
    print("You may need to install pyBigWig")
pp = pprint.PrettyPrinter(indent=2)


def smooth(ser, sc):
    return np.array(pd.Series(ser).rolling(sc, min_periods=1, center=True).mean())
# string,string -> bool,[],res
# select a strain and an experimental value and return if available, the files and the resolution
# def is_available(strain, experiment):
#    return True,[],1


ROOT = "../DNaseI/data/"


def nan_polate_c(A, kind="linear"):
    ok = ~np.isnan(A)
    x = np.arange(len(A))
    f2 = interp1d(x[ok], A[ok], kind=kind, bounds_error=False)
    # print(ok)
    return f2(x)


def is_available_alias(strain, experiment):
    alias = {"Hela": ["Hela", "HeLaS3", "Helas3"],
             "Helas3": ["Helas3", "HeLaS3", "Hela"],
             "GM12878": ["GM12878", "Gm12878"],
             "Gm12878": ["GM12878", "Gm12878"]
             }
    # alias={"Helas3":["HeLaS3","Hela","Helas3"]}
    if strain not in alias.keys():
        avail, files, res = is_available(strain, experiment)
    else:
        for strain1 in alias[strain]:
            avail, files, res = is_available(strain1, experiment)
            if files != []:
                if strain1 != strain:
                    print("Using alias %s" % strain1)
                return avail, files, res
    return avail, files, res


def is_available(strain, experiment):

    avail_exp = ["MRT", "OKSeq", "OKSeqo", "DNaseI", "ORC2", "ExpGenes", "Faire", "Meth", "Meth450",
                 "Constant", "OKSeqF", "OKSeqR", "OKSeqS", "CNV", "NFR",
                 "MCM", "HMM", "GC", "Bubble","G4","G4p","G4m","Ini","ORC1","AT_20","AT_5","AT_30","RHMM","MRTstd",
                 "RNA_seq","MCMo","MCMp","MCM-beda","Mcm3","Mcm7","Orc2","Orc3"]
    marks = ['H2az', 'H3k27ac', 'H3k27me3', 'H3k36me3', 'H3k4me1',
             'H3k4me2', 'H3k4me3', 'H3k79me2', 'H3k9ac', 'H3k9me1',
             'H3k9me3', 'H4k20me1', "SNS"]
    marks_bw = [m + "wig" for m in marks]
    Prot = ["Rad21","ORC2"]

    #print("La")

    if strain in ["Yeast-MCM"]:
        lroot = ROOT+"/external/"+strain + "/"
        resolutions = glob.glob(lroot + "/*")
        #print(lroot + "/*")
        resolutions = [r.split("/")[-1] for r in resolutions if "kb" in r]
        #print(resolutions)
        if len(resolutions) != 0:
            exps = glob.glob(lroot + resolutions[0]+"/*")
            files = []+exps
            exps =  [exp.split("/")[-1][:-4] for exp in exps if "csv" in exp]

            for iexp,exp in enumerate(exps):
                if exp == experiment:
                    return True,[files[iexp]],int(resolutions[0].replace("kb",""))
    if strain in ["Cerevisae"] and experiment =="MCM-beda":
        lroot = ROOT+"/external/Yeast-MCM-bedalov/"

        return True,glob.glob(lroot+"/*"),0.001

    if experiment not in avail_exp + marks + Prot + marks_bw:
        print("Exp %s not available" % experiment)
        print("Available experiments", avail_exp + marks + Prot)
        return False, [], None

    if experiment == "Constant":
        return True, [], 1
    if experiment == "MRT":

        if strain == "Cerevisae":
            return True, ["/home/jarbona/ifromprof/notebooks/exploratory/Yeast_wt_alvino.csv"], 1
        elif strain == "Raji":
            files = glob.glob(ROOT + "/external/timing_final//*_Nina_Raji_logE2Lratio_w100kbp_dw10kbp.dat" )
            return True, files, 10
        else:
            root = ROOT + "/Data/UCSC/hsap_hg19/downloads/ENCODE/wgEncodeUwRepliSeq_V2/compute_profiles/timing_final/"
            root = ROOT + "/external/timing_final/"
            extract = glob.glob(root + "/*Rep1_chr10.dat")
            cells = [e.split("_")[-3] for e in extract]
            if strain in cells:
                files = glob.glob(root + "/timing_final_W100kb_dx10kb_%s*" % strain)
                return True, files, 10

    if experiment == "MRTstd":

        root = ROOT + "/external/Sfrac/"

        extract = glob.glob(root + "/*Rep1_chr10.dat")
        cells = [e.split("_")[-3] for e in extract]
        if strain in cells:
            files = glob.glob(root + "/Sfrac_HansenNormDenoised_W100kb_dx10kb_%s*" % strain)
            return True, files, 10

    if experiment == "ExpGenes":
        root = ROOT + "/external/ExpressedGenes/"
        extract = glob.glob(root + "/*ExpressedGenes_zero.txt")
        # print(extract)
        cells = [e.split("/")[-1].replace("ExpressedGenes_zero.txt", "") for e in extract]
        print(cells)
        if strain in cells:
            files = glob.glob(root + "/%sExpressedGenes_zero.txt" % strain)

            return True, files, 10

    if experiment == "RNA_seq":
        root = ROOT + "external//RNA_seq//"
        # root = ROOT + "/external/1kb_profiles//"
        extract = glob.glob(root + "*Tot")
        # print(extract)
        cells = [e.split("/")[-1].split("_")[0] for e in extract]
        cells.sort()
        if strain in cells:
            files = glob.glob(root + strain + "_Tot/*")
            files.sort()
            return True, files, 1

    if experiment == "NFR":
        root = ROOT + "/external/NFR/"
        # root = ROOT + "/external/1kb_profiles//"
        extract = glob.glob(root + "*.bed")
        return True, extract, 1

    if experiment == "Bubble":
        root = ROOT + "/external/Bubble/"
        # root = ROOT + "/external/1kb_profiles//"
        extract = glob.glob(root + "*.bedgraph")
        # print(extract)
        cells = [e.split("/")[-1].split(".bedgraph")[0] for e in extract]
        cells.sort()
        if strain in cells:
            files = glob.glob(root + strain + ".bedgraph")
            files.sort()
            return True, files, 1
    #print("IRCRRRRRRRRRRRRRRRRRRRR")
    if experiment == "ORC1":
        #print("LA")
        root = ROOT + "/external/ORC1/"
        # root = ROOT + "/external/1kb_profiles//"
        extract = glob.glob(root + "*.bed")
        #print(extract)
        cells = [e.split("/")[-1].split(".bed")[0] for e in extract]
        cells.sort()
        if strain in cells:
            files = glob.glob(root + strain + ".bed")
            files.sort()
            return True, files,

    if (experiment in ["Mcm3","Mcm7","Orc2","Orc3"]) and strain =="Raji":
        return True,glob.glob(ROOT+"/external/nina_kirstein/*_"+experiment+"_G1_1kbMEAN.txt") ,1

    if experiment in ["MCM","MCMp"]:
            #print("LA")
        if strain != "Hela":
            return False,[],1
        root = ROOT + "/external/MCM2-bed/R1/"
        # root = ROOT + "/external/1kb_profiles//"
        extract = glob.glob(root + "*.txt")
        #print(extract)
        return True, extract, 1

    if experiment == "Ini":
        root = ROOT + "/external/ini/"
        # root = ROOT + "/external/1kb_profiles//"
        extract = glob.glob(root + "*.csv")
        # print(extract)
        cells = [e.split("/")[-1].split(".csv")[0] for e in extract]
        cells.sort()
        if strain in cells:
            files = glob.glob(root + strain + ".csv")
            files.sort()
            return True, files, 1

    if experiment == "GC":
        root = ROOT + "/external//1ColProfiles/*1kbp*"  # chr1_gc_native_w1kbp.dat
        extract = glob.glob(root)
        return True, extract, 1
    if "AT" in experiment:
        root = ROOT + "/external//AT_hooks/c__%s.csv"%experiment.split("_")[1]  # chr1_gc_native_w1kbp.dat
        extract = glob.glob(root)
        return True, extract, 5

    if experiment == "SNS":
        root = ROOT + "/external/SNS/"
        # root = ROOT + "/external/1kb_profiles//"
        extract = []
        if strain in ["K562"]:
            extract = glob.glob(root + "*.bed")
        elif strain in ["HeLaS3","Hela","HeLa"]:
            extract=glob.glob(root + "*.csv")
        #print("Strain",strain)
        #print(extract, root)
        if strain not in ["K562","HeLaS3"]:
            print("Wrong strain")
            print("Only K562")
            return False, [], 1
        return True, extract, 1

    if experiment == "MCMo":
        if strain not in ["HeLa", "HeLaS3","Hela"]:
            print("Wrong strain")
            print("Only", "HeLa", "HeLaS3")
            raise
        root = ROOT + "/external/MCM/"
        # root = ROOT + "/external/1kb_profiles//"
        extract = glob.glob(root + "*.bed")
        print(extract, root)
        return True, extract, 1

    if experiment == "MCMbw":
        if strain not in ["HeLa", "HeLaS3"]:
            print("Wrong strain")
            print("Only", "HeLa", "HeLaS3")
            raise
        """
        root = ROOT + "/external/SNS/"
        # root = ROOT + "/external/1kb_profiles//"
        extract = glob.glob(root + "*.bed")
        print(extract, root)
        return True, extract, 1"""

    if  "G4" in experiment:
        root = ROOT + "/external/G4/"
        # root = ROOT + "/external/1kb_profiles//"
        extract = glob.glob(root + "*.bed")
        print(extract, root)
        return True, extract, 1

    if experiment == "CNV":
        root = ROOT + "/external/CNV/"
        # root = ROOT + "/external/1kb_profiles//"
        extract = glob.glob(root + "*.txt")
        # print(extract)
        cells = [e.split("/")[-1].split(".txt")[0] for e in extract]
        cells.sort()
        if strain in cells:
            files = glob.glob(root + strain + ".txt")
            files.sort()
            #print(files)
            return True, files, 10

    if experiment == "HMM":
        root = ROOT + "/external/HMM/"
        # root = ROOT + "/external/1kb_profiles//"
        extract = glob.glob(root + "*.bed")
        # print(extract)
        cells = [e.split("/")[-1].replace("wgEncodeBroadHmm", "").replace("HMM.bed", "")
                 for e in extract]
        cells.sort()
        if strain in cells:
            files = glob.glob(root + "wgEncodeBroadHmm%sHMM.bed" % strain)
            files.sort()
            # print(files)
            return True, files, 10

    if experiment == "RHMM":
        root = ROOT + "/external/RHMM/"
        # root = ROOT + "/external/1kb_profiles//"
        extract = glob.glob(root + "*.bed")
        #print(extract)
        cells = [e.split("/")[-1].replace("RHMM.bed", "")
                 for e in extract]
        cells.sort()
        if strain in cells:
            files = glob.glob(root + "%sRHMM.bed" % strain)
            files.sort()
            # print(files)
            return True, files, 1

    if experiment.startswith("OKSeq"):
        root = ROOT + "/Data/UCSC/hsap_hg19//local/Okazaki_Hyrien/1kb_profiles/"
        root = ROOT + "/external/1kb_profiles//"
        extract = glob.glob(root + "*")
        cells = [e.split("/")[-1] for e in extract]
        cells.sort()
        # print(cells)
        if strain in cells:
            if experiment == "OKSeqo":
                files = glob.glob(root + strain + "/*pol*")
            if experiment == "OKSeqF":
                files = glob.glob(root + strain + "/*F*")
            if experiment == "OKSeqR":
                files = glob.glob(root + strain + "/*R*")
            if experiment in ["OKSeqS", "OKSeq"]:
                # print("La")
                files = glob.glob(root + strain + "/*R*")
                files += glob.glob(root + strain + "/*F*")

            files.sort()
            return True, files, 1
    if experiment == "DNaseI":
        root = ROOT + "/external/DNaseI//"
        print(root)
        if strain == "Cerevisae":
            return True, [root + "/yeast.dnaseI.tagCounts.bed"], 0.001
        else:
            extract = glob.glob(root + "/*.narrowPeak")

            cells = [e.split("/")[-1].replace("wgEncodeAwgDnaseUwduke",
                                              "").replace("UniPk.narrowPeak", "") for e in extract]
            extract2 = glob.glob(root + "../DNaseIK562/*.narrowPeak")
            cells2 = [e.split("/")[-1].replace("wgEncodeOpenChromDnase",
                                               "").replace("Pk.narrowPeak", "") for e in extract2]

            extract3 = glob.glob(root + "../DNaseIK562/*.bigWig")
            cells3 = [e.split("/")[-1].replace("wgEncodeUwDnase",
                                               "").replace("Rep1.bigWig", "") for e in extract3]
            # print(extract2, cells2)
            extract += extract2
            cells += cells2

            extract += extract3
            cells += cells3

            if strain in cells:
                files = [extract[cells.index(strain)]]
                return True, files, 0.001

    if experiment == "Meth":
        root = ROOT + "/external/methylation//"

        extract = glob.glob(root + "*.bed")
        cells = [e.split("/")[-1].replace(".bed", "") for e in extract]

        if strain in cells:
            files = [extract[cells.index(strain)]]
            return True, files, 1
    if experiment == "Meth450":
        root = ROOT + "/external/methylation450//"

        extract = glob.glob(root + "*.bed")
        cells = [e.split("/")[-1].replace(".bed", "") for e in extract]

        if strain in cells:
            files = [extract[cells.index(strain)]]
            return True, files, 1

    if experiment == "Faire":
        root = ROOT + "/external/Faire//"

        extract = glob.glob(root + "*.pk")
        cells = [e.split("/")[-1].replace("UncFAIREseq.pk", "") for e in extract]

        if strain in cells:
            files = [extract[cells.index(strain)]]
            return True, files, 1

    if experiment in Prot:
        root = ROOT + "/external/DNaseI//"

        extract = glob.glob(root + "/*.narrowPeak")

        cells = [e.split("/")[-1].replace(experiment + "narrowPeak", "") for e in extract]

        if strain in cells:
            files = [extract[cells.index(strain)]]
            return True, files, 1

        root = ROOT + "/external/proteins//"

        extract = glob.glob(root + "/*.csv")
        cells = [e.split("/")[-1].replace("_ORC2_miotto.csv", "") for e in extract]
        if strain in cells:
            files = glob.glob(root + "/%s_ORC2_miotto.csv" % strain)
            return True, files, 1

    if experiment in marks:
        root = ROOT + "/external/histones//"

        if  experiment == "H2az" and strain == "IMR90":
            experiment = "H2A.Z"
        extract = glob.glob(root + "/*%s*.broadPeak" % experiment)
        #print(extract)
        if strain not in ["IMR90"]:
            cells = [e.split("/")[-1].replace("wgEncodeBroadHistone",
                                              "").replace("Std", "").replace("%sPk.broadPeak" % experiment, "") for e in extract]
            # print(extract,cells)
            if strain in cells:
                files = glob.glob(root + "/wgEncodeBroadHistone%s%sStdPk.broadPeak" %
                                  (strain, experiment))
                files += glob.glob(root + "/wgEncodeBroadHistone%s%sPk.broadPeak" %
                                   (strain, experiment))

                return True, files, 1
        else:
            cells = [e.split("/")[-1].split("-")[0] for e in
                     extract]
            # print(extract,cells)



            print("Larr")

            if strain in cells:
                files = glob.glob(root + "/%s-%s.broadPeak" %
                                  (strain, experiment))


                return True, files, 1

    if experiment[:-3] in marks:
        root = ROOT + "/external/histones//"
        if strain not in ["IMR90"]:
            extract = glob.glob(root + "/*%s*.bigWig" % experiment[:-3])
            # print(extract)
            cells = []
            for c in extract:
                if "StdSig" in c:
                    cells.append(c.split("/")[-1].replace("wgEncodeBroadHistone",
                                                          "").replace("%sStdSig.bigWig" % experiment[:-3], ""))
                else:
                    cells.append(c.split("/")[-1].replace("wgEncodeBroadHistone",
                                                          "").replace("%sSig.bigWig" % experiment[:-3], ""))

            # print(extract, cells)
            if strain in cells:
                files = glob.glob(root + "/wgEncodeBroadHistone%s%sStdSig.bigWig" %
                                  (strain, experiment[:-3]))
                if files == []:
                    #print("Warning using Sig")
                    files = glob.glob(root + "/wgEncodeBroadHistone%s%sSig.bigWig" %
                                      (strain, experiment[:-3]))
                # print(files)
                return True, files, 1
        else:
            exp = experiment[:-3]
            exp = exp.replace("k","K") # from roadmap epi
            extract = glob.glob(root + "/IMR90_%s*wh.csv" % exp)
            print(extract)
            cells = []

            return True, extract, 1

    print("Available cells")
    pp.pprint(cells)
    return False, [], None


def re_sample(x, y, start, end, resolution=1000):

    resampled = np.zeros(int(end / resolution - start / resolution)) + np.nan
    # print(data)
    # print(resampled.shape)
    for p, v in zip(x, y):
        #print(v)
        if not np.isnan(v):
            posi = int((p - start) / resolution)
            if np.isnan(resampled[min(posi, len(resampled) - 1)]):
                resampled[min(posi, len(resampled) - 1)] = 0
            resampled[min(posi, len(resampled) - 1)] += v
            if int(posi) > len(resampled) + 1:
                print("resample", posi, len(resampled))
                # raise "Problem"
    return np.arange(len(resampled)) * resolution + start, resampled


def cut_path(start, end, res=1):

    initpos = 0 + start
    delta = end - start
    path = [0 + initpos]

    def cond(x): return x <= end

    while (initpos + delta) != int(initpos) and cond(initpos):
        ddelta = int(initpos) + res - initpos
        initpos += ddelta
        ddelta -= initpos
        path.append(initpos)
    path[-1] = end
    if len(path) >= 2 and path[-1] == path[-2]:
        path.pop(-1)

    return path


def overlap(start, end, res):
    r = cut_path(start / res, end / res)
    return [ri * res for ri in r]


def overlap_fraction(start, end, res):
    assert(start <= end)
    v = np.array(overlap(start, end, res))
    deltas = (v[1:] - v[:-1]) / res
    indexes = np.array(v[:-1] / res, dtype=np.int)
    return deltas, indexes



def create_index_human(strain,exp,resolution=10,root="./"):
    chromlength = [248956422, 242193529, 198295559, 190214555, 181538259,
                   170805979, 159345973, 145138636, 138394717,
                   133797422, 135086622, 133275309, 114364328, 107043718,
                   101991189, 90338345, 83257441,
                   80373285, 58617616, 64444167, 46709983, 50818468]
    #chromlength = [248956422]
    data = {iexp:[] for iexp in exp}
    for chrom, length in enumerate(chromlength, 1):
        for iexp in exp:
            data[iexp].append(replication_data(strain, iexp,
                                     chromosome=chrom, start=0,
                                     end=length // 1000,
                                     resolution=resolution)[1])
            if iexp == "OKSeq":
                data[iexp][-1] /= resolution

    ran = [np.arange(len(dat)) * 1000 * resolution for dat in data[exp[0]]]
    index = {"chrom": np.concatenate([["chr%i"%i]*len(xran) for i,xran in enumerate(ran,1)]),
             "chromStart":np.concatenate(ran),
             "chromEnd":np.concatenate(ran)}

    print(root)
    os.makedirs(root,exist_ok=True)
    pd.DataFrame(index).to_csv(root+"/index.csv",index=False)

    for iexp in exp:
        index.update({"signalValue":np.concatenate(data[iexp])})
        Df = pd.DataFrame(index)
        Df.to_csv(root + "/%s.csv" % iexp, index=False)







def whole_genome(**kwargs):
    chromlength = [248956422, 242193529, 198295559, 190214555, 181538259,
                   170805979, 159345973, 145138636, 138394717,
                   133797422, 135086622, 133275309, 114364328, 107043718,
                   101991189, 90338345, 83257441,
                   80373285, 58617616, 64444167, 46709983, 50818468]
    data = []

    def fl(name):
        def sanit(z):
            z = z.replace("/", "")
            return z
        if type(name) == dict:
            items = list(name.items())
            items.sort()
            return "".join(["%s-%s" % (p, sanit(str(fl(value)))) for p, value in items])
        else:
            return name

    redo = kwargs.pop("redo")
    root = kwargs.get("root", "./")

    # print("ic")
    if "root" in kwargs.keys():
        # print("la")
        kwargs.pop("root")

    name = root + "data/saved/"+fl(kwargs)
    if os.path.exists(name) and not redo:
        with open(name, "rb") as f:
            return cPickle.load(f)
    strain = kwargs.pop("strain")
    experiment = kwargs.pop("experiment")
    resolution = kwargs.pop("resolution")
    for chrom, length in enumerate(chromlength, 1):

        data.append(replication_data(strain, experiment,
                                     chromosome=chrom, start=0,
                                     end=length//1000,
                                     resolution=resolution, **kwargs)[1])
        if len(data[-1]) != int(length / 1000 / resolution - 0 / resolution):
            print(strain, experiment, len(data[-1]),
                  int(length / 1000 / resolution - 0 / resolution))
            raise
    with open(name, "wb") as f:
        cPickle.dump(data, f)
    return data


def replication_data(strain, experiment, chromosome,
                     start, end, resolution, raw=False,
                     oData=False, bp=True, bpc=False, filename=None,
                     pad=False, smoothf=None, signame="signalValue"):

    marks = ['H2az', 'H3k27ac', 'H3k27me3', 'H3k36me3', 'H3k4me1',
             'H3k4me2', 'H3k4me3', 'H3k79me2', 'H3k9ac', 'H3k9me1',
             'H3k9me3', 'H4k20me1']

    if experiment != "" and os.path.exists(experiment):
        filename = experiment


    if os.path.exists(strain) and strain.endswith("csv"):
        #print(strain)
        data=pd.read_csv(strain)
        #print(len(data))
        sub = data[data.chr==chromosome][experiment]
        y = np.array(sub[int(start/resolution):int(end/resolution)])
        print("Sizes",chromosome,len(sub),int(end/resolution))
        return np.arange(len(y))*resolution + start,y
        #chn = list(set(data.chr))


    if experiment.endswith("weight"):
        from repli1d.retrieve_marks import norm2
        with open(experiment, "rb") as f:
            w = pickle.load(f)
            if len(w) == 4:
                [M, S, bestw, Exp] = w
                normed = False
            else:
                [M, S, bestw, Exp, normed] = w

        if normed:
            smark = replication_data(chromosome=chromosome, start=start,
                                     end=end, strain=strain, experiment="CNV",
                                     resolution=resolution, raw=False, oData=False,
                                     bp=True, bpc=False, filename=None)[1]
            smark[smark == 0] = 4
            smark[np.isnan(smark)] = 4
            CNV = smark

        Signals = {}
        for mark in Exp:

            if "_" in mark:
                markn, smoothf = mark.split("_")
                smoothf = int(smoothf)
            else:
                markn = mark
            smark = replication_data(chromosome=chromosome, start=start,
                                     end=end, strain=strain, experiment=markn,
                                     resolution=resolution, raw=False, oData=False,
                                     bp=True, bpc=False, filename=None)[1]

            if normed:
                smark /= CNV
            if mark != "Constant":
                Signals[mark] = norm2(smark, mean=M[mark], std=S[mark], cut=15)[0]
            else:
                Signals[mark] = smark

            if smoothf is not None:
                Signals[mark] = smooth(Signals[mark], smoothf)

        # print(bestw)
        if type(bestw[0]) in [list, np.ndarray]:
            comp = [bestw[0][i]*(-1+2/(1+np.exp(-bestw[1][i]*(Signals[iexp]-bestw[2][i]))))
                    for i, iexp in enumerate(Exp)]
        else:
            comp = np.array([bestw[i] * Signals[iexp] for i, iexp in enumerate(Exp)])
        y = np.sum(comp, axis=0)
        y[y < 0] = 0

        x = np.arange(len(y))*resolution + start
        return x, y
        # print(smark)

    if filename is None:
        avail, files, resolution_experiment = is_available_alias(strain, experiment)
        if not avail:
            return [], []
    else:
        print("Filename", filename)
        avail = True
        files = [filename]
        resolution_experiment = 0.001

        if filename.endswith("bigWig") or filename.endswith("bw"):
            cell = pyBigWig.open(files[0])
            if "chrI" in cell.chroms().keys():
                print("Yeast")
                #print(cell.chroms())
                from repli1d.tools import int_to_roman
                #print(int(chromosome))
                chromosome = int_to_roman(int(chromosome))
            if end is None:
                end = int(cell.chroms()['chr%s' % str(chromosome)] / 1000)
            #print(start * 1000, end * 1000, int((end - start) / (resolution)))

            #Check the end:
            endp = end
            smaller =False
            if end > cell.chroms()["chr%s" % str(chromosome)]/1000:
                print("Warning index > end ch")
                endp = int(cell.chroms()["chr%s" % str(chromosome)] /1000)
                smaller = True

            v = [np.nan if s is None else s for s in cell.stats(
                "chr%s" % str(chromosome), start * 1000, endp * 1000, nBins=int((endp - start) / resolution))]
            if not smaller:
                return np.arange(start, end + 100, resolution)[: len(v)], np.array(v)
            else:
                x = np.arange(start, end + 0.1, resolution)
                y = np.zeros_like(x) + np.nan
                y[:len(v)] = np.array(v)
                return x[:end], y[:end]

        if filename.endswith("narrowPeak"):
            index = ["chrom", "chromStart", "chromEnd", "name", "score",
                     "strand", "signalValue", "pValue", "qValue", "peak"]
            chro = str(chromosome)
            strain = pd.read_csv(files[0], sep="\t", names=index)

            data = strain[(strain.chrom == "chr%s" % chro) & (
                strain.chromStart > 1000 * start) & (strain.chromStart < 1000 * end)]

            if oData:
                return data

            x = np.array(data.chromStart / 2 + data.chromEnd / 2) / 1000  # kb
            y = np.array(data.signalValue)
            return re_sample(x, y, start, end, resolution)

        if filename.endswith("bed"):
            index = ["chrom", "chromStart", "chromEnd", "name", "score",
                     "strand", "signalValue", "pValue", "qValue", "peak"]
            chro = str(chromosome)
            strain = pd.read_csv(files[0], sep="\t", names=index)

            if "chrI" in set(strain["chrom"]):
                print("Yeast")
                # print(cell.chroms())
                from repli1d.tools import int_to_roman
                # print(int(chromosome))
                chro = int_to_roman(int(chromosome))

            #print(strain)
            data = strain[(strain.chrom == "chr%s" % chro) & (
                strain.chromStart > 1000 * start) & (strain.chromStart < 1000 * end)]
            #print("La")
            #print(data)
            if oData:
                return data

            x = np.array(data.chromStart / 2 + data.chromEnd / 2) / 1000  # kb
            y = np.array(data.signalValue)
            y = np.ones_like(x)
            #print(y)
            return re_sample(x, y, start, end, resolution)

        if filename.endswith("tagAlign"):
            index = ["chrom", "chromStart", "chromEnd", "N", "signalValue","pm"]
            chro = str(chromosome)
            strain = pd.read_csv(files[0], sep="\t", names=index)

            data = strain[(strain.chrom == "chr%s" % chro) & (
                    strain.chromStart > 1000 * start) & (strain.chromStart < 1000 * end)]

            if oData:
                return data

            x = np.array(data.chromStart / 2 + data.chromEnd / 2) / 1000  # kb
            y = np.array(data.signalValue) / 1000 # because the value is 1000 ?
            return re_sample(x, y, start, end, resolution)

        if filename.endswith("csv"):
            #index = ["chrom", "chromStart", "chromEnd",  "signalValue"]
            chro = str(chromosome)
            # print(files[0])

            strain = pd.read_csv(files[0], sep="\t")
            #print(strain.mean())
            tmpl = "chr%s"
            f = 1000
            if "chrom" not in strain.columns:
                strain = pd.read_csv(files[0], sep=",")
                # print(strain)
                #tmpl = "chrom%s"
                f = 1
                # strain.chrom
            # sanitize chrom:
            def sanitize(ch):
                if type(ch) == int:
                    return "chr%s"%ch
                if type(ch) == str:
                    if "chrom" in ch:
                        return ch.replace("chrom","chr")
                    if (not "chr" in ch) and (not "chrom" in ch):
                        return "chr%s"%ch
                    return ch

            strain["chrom"] = [sanitize(ch) for ch in strain["chrom"]]


            #print(strain.describe())
            #print(strain.head())
            #print( tmpl % chro)
            #print("F",f)
            data = strain[(strain.chrom == tmpl % chro) & (
                strain.chromStart >= f * start) & (strain.chromStart < f * end)]
            #print("Warning coold shift one")
            #print("Data",len(data))

            if oData:
                return data

            x = np.array(data.chromStart / 2 + data.chromEnd / 2) / f  # kb
            if signame == "signalValue" and signame not in data.columns:
                if "signal" in data.columns:
                    signame = "signal"
                    print("Warning changing signalValue to signal")
            y = np.array(data[signame])
            #print(x[:10])
            #print(y[:10])
            #print(start,end)
            #print(chro,np.mean(y),len(y))
            return re_sample(x, y, start, end, resolution)

    # print(files)
    assert(type(files) == list)

    if strain in ["Yeast-MCM"]:
        #print(files[0])
        index = "/".join(files[0].split("/")[:-1]) + "/index.csv"
        index = pd.read_csv(index,sep="\t")
        strain = index
        exp = pd.read_csv(files[0])

        if len(index) != len(exp):
            raise ValueError("Wrong size of indexing %i %i"%(len(index) , len(exp)))

        strain["signal"] = exp



        if "Yeast" in strain:
            from repli1d.tools import int_to_roman
            chro = int_to_roman(int(chromosome))
        else:
            chro = chromosome

        data = strain[(strain.chrom == "chr%s" % chro) & (
            strain.chromStart > 1000*start) & (strain.chromStart < 1000*end)]

        #print(data)
        x = np.array(data.chromStart / 2 + data.chromEnd / 2)  / 1000  # kb
        y = np.array(data.signal)

        if raw:
            return x, y
        else:
            return re_sample(x, y, start, end, resolution)

    if experiment in ["MCM","MCMp"]:
        #print(chromosome)
        files = [f for f in files if "chr%i."%chromosome in f]
        #print(files)
        files += [f.replace("R1","R2") for f in files]
        #print(files)
        data = np.sum([np.array(pd.read_csv(f))[:,0] for f in files],axis=0)

        x = np.arange(len(data))  # kb
        sub = (x> start) & (x < end)

        x=x[sub]
        y = np.array(data[sub],dtype=np.float)

        x,y = re_sample(x, y, start, end, resolution)
        if experiment == "MCMp":
            print(np.nanpercentile(y,50))
            peaks, _ = find_peaks(y / np.nanpercentile(y,50),width=1,prominence=1.)

            peaksa = np.zeros_like(y,dtype=np.bool)
            for p in peaks:
                peaksa[p]=True

            print(len(y),len(peaks),"Peaks")
            y[~peaksa]=0

        #raise "NT"

        return  x,y


    if experiment == "DNaseI":
        if strain == "Cerevisae":
            index = ["chrom", "chromStart", "chromEnd", "name", "signalValue"]
            print(files[0])
            strain = pd.read_csv(files[0], sep="\t", names=index)

            chro = str(chromosome)

            if oData:
                return strain

            data = strain[(strain.chrom == "chr%s" % chro) & (
                strain.chromStart > 1000 * start) & (strain.chromStart < 1000 * end)]

            x = np.array(data.chromStart / 2 + data.chromEnd / 2) / 1000  # kb
            y = np.array(data.signalValue)

        else:
            index = ["chrom", "chromStart", "chromEnd", "name", "score",
                     "strand", "signalValue", "pValue", "qValue", "peak"]
            chro = str(chromosome)
            if files[0].endswith("narrowPeak"):
                strain = pd.read_csv(files[0], sep="\t", names=index)

                data = strain[(strain.chrom == "chr%s" % chro) & (
                    strain.chromStart > 1000 * start) & (strain.chromStart < 1000 * end)]

                if oData:
                    return data

                x = np.array(data.chromStart / 2 + data.chromEnd / 2) / 1000  # kb
                y = np.array(data.signalValue)
            else:
                cell = pyBigWig.open(files[0])
                if end is None:
                    end = cell.chroms()['chr%s' % str(chro)]
                v = [np.nan if s is None else s for s in cell.stats(
                    "chr%s" % str(chro), start * 1000, end * 1000, nBins=int(end - start) // (resolution))]

                return np.arange(start, end + 100, resolution)[: len(v)], np.array(v)

    if experiment == "Faire":

        index = ["chrom", "chromStart", "chromEnd", "name", "score",
                 "strand", "signalValue", "pValue", "qValue", "peak"]
        chro = str(chromosome)
        strain = pd.read_csv(files[0], sep="\t", names=index)

        data = strain[(strain.chrom == "chr%s" % chro) & (
            strain.chromStart > 1000 * start) & (strain.chromStart < 1000 * end)]

        if oData:
            return data

        x = np.array(data.chromStart / 2 + data.chromEnd / 2) / 1000  # kb
        y = np.array(data.signalValue)

    if "MCM-beda" in experiment:

            #print(files[0])
            strain = pd.read_csv(files[0],sep="\t")
            #strain.MCM = smooth(strain.MCM2_ChEC)
            chromosome = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X",
                          11: "XI", 12: "XII", 13: "XIII", 14: "XIV", 15: "XV", 16: "XVI"}[chromosome]
            #print(chromosome)
            #print(strain)
            data = strain[(strain.chr == "chr%s" % chromosome) & (
                strain.coord > 1000 * start) & (strain.coord < 1000 * end)]
            #print(data)
            if oData:
                return data

            x = np.array(data.coord) / 1000  # kb
            #y = np.array(data.cerevisiae_MCM2ChEC_rep1_library_fragment_size_range_51bpto100bp)
            y = np.array(data.cerevisiae_MCM2ChEC_rep1_library_fragment_size_range_all)



    if "G4" in experiment:

        index = ["chrom", "chromStart", "chromEnd"]
        chro = str(chromosome)
        if "p" in experiment:
            ip = np.argmax(["plus" in f for f in files])
            print(files[ip],)
            strain = pd.read_csv(files[ip], sep="\t", names=index)
        elif "m" in experiment:
            ip = np.argmax(["minus" in f for f in files])
            print(files[ip])
            strain = pd.read_csv(files[ip], sep="\t", names=index)

        else:
            strain = pd.concat([pd.read_csv(files[ip], sep="\t", names=index) for ip in [0,1]])

        data = strain[(strain.chrom == "chr%s" % chro) & (
            strain.chromStart > 1000 * start) & (strain.chromStart < 1000 * end)]

        if oData:
            return data

        x = np.array(data.chromStart / 2 + data.chromEnd / 2) / 1000  # kb
        y = np.ones_like(x)

    if experiment == "GC":

        index = ["signalValue"]
        chro = str(chromosome)
        file = [f for f in files if "chr%s_" % chro in f]
        strain = pd.read_csv(file[0], sep="\t", names=index)
        strain["chromStart"] = np.arange(0, len(strain)*1000, 1000)
        strain["chromEnd"] = np.arange(0, len(strain)*1000, 1000)

        data = strain[(strain.chromStart > 1000 * start) & (strain.chromStart < 1000 * end)]

        if oData:
            return data

        x = np.array(data.chromStart / 2 + data.chromEnd / 2) / 1000  # kb
        y = np.array(data.signalValue)

    if "AT" in experiment :

        index = ["signalValue"]
        chro = str(chromosome)
        #file = [f for f in files if "chr%s_" % chro in f]
        strain = pd.read_csv(files[0], sep="\t")
        #print(strain.head())

        data = strain[(strain.chromStart > 1000 * start) & (strain.chromStart < 1000 * end)]

        if oData:
            return data

        x = np.array(data.chromStart / 2 + data.chromEnd / 2) / 1000  # kb
        y = np.array(data.signalValue)

    if experiment == "Ini":
        filename = files[0]
        index = ["chrom", "chromStart", "chromEnd"]
        chro = str(chromosome)
        strain = pd.read_csv(files[0], sep=",", names=index)

        data = strain[(strain.chrom == "%s" % chro) & (
                strain.chromStart > 1000 * start) & (strain.chromStart < 1000 * end)]

        if oData:
            data.chromStart /= 1000
            data.chromEnd /= 1000
            return data

        x = np.array(data.chromStart / 2 + data.chromEnd / 2) / 1000  # kb
        y = np.ones_like(x)
        return re_sample(x, y, start, end, resolution)

    if experiment == "HMM" :
        filename = files[0]
        index = ["chrom", "chromStart", "chromEnd", "ClassName", "u1",
                 "u2", "u3", "u4", "color"]
        chro = str(chromosome)
        strain = pd.read_csv(files[0], sep="\t", names=index)

        data = strain[(strain.chrom == "chr%s" % chro) & (
            strain.chromStart > 1000 * start) & (strain.chromStart < 1000 * end)]

        if oData:
            data.chromStart /= 1000
            data.chromEnd /= 1000
            return data

        x = np.array(data.chromStart / 2 + data.chromEnd / 2) / 1000  # kb
        y = np.array(data.color)
        return re_sample(x, y, start, end, resolution)

    if experiment == "RHMM":
        filename = files[0]
        try:
            index = ["chrom", "chromStart", "chromEnd", "ClassName", "u1",
                     "u2", "u3", "u4", "color"]

            chro = str(chromosome)
            strain = pd.read_csv(files[0], sep="\t", names=index)
            strain["ClassName"] = [int(class_n.split("_")[0]) for class_n in strain.ClassName]
        except:
            index = ["chrom", "chromStart", "chromEnd", "ClassName"]
            chro = str(chromosome)
            strain = pd.read_csv(files[0], sep="\t", names=index)

            strain["ClassName"] = [int(class_n[1:]) for class_n in strain.ClassName]
            inac=3
            trans=6
            #r = {2:inac,3:inac,4:inac,7:inac,8:inac,9:inac,10:inac,11:inac,
            #                              12:12,
            #                              13:13,
            #                              5:trans,6:trans
            #                              }
            for k,v in r.items():
                strain.ClassName[strain.ClassName==k]=v

        data = strain[(strain.chrom == "chr%s" % chro) & (
            strain.chromStart > 1000 * start) & (strain.chromStart < 1000 * end)]

        if oData:
            data.chromStart /= 1000
            data.chromEnd /= 1000
            return data

        #x = np.array(data.chromStart / 2 + data.chromEnd / 2) / 1000  # kb
        #y = np.array(data.ClassName)
        x = np.array([data.chromStart,data.chromEnd]).reshape((1, -1), order="F")[0] / 1000
        print(x[:10])
        y = np.array([data.ClassName, data.ClassName]).reshape((1, -1), order="F")[0]


        return x,y #re_sample(x, y, start, end, resolution)

    if experiment == "CNV":
        index = ["chrom", "chromStart", "chromEnd", "CNV", "Sig"]
        data = pd.read_csv(files[0], sep="\t", names=index)
        #print(data)
        # x = np.arange(start, end, resolution)
        # y = np.zeros_like(x, dtype=np.float)
        y = np.zeros(int(end / resolution - start / resolution))
        x = np.arange(len(y)) * resolution + start
        print(data.chrom.dtype)
        if str(data.chrom.dtype) == "int64":
            data = data[data.chrom == chromosome]
        else:
            data = data[data.chrom == str(chromosome)]
        print(data)
        for startv, endv, CNV in zip(data.chromStart, data.chromEnd, data.CNV):
            startv /= 1000
            endv /= 1000
            # deltas, indexes = overlap_fraction(startv / 1000, endv / 1000, resolution)
            # print(startv, endv, endv < start, startv > end)
            if endv < start or startv > end:
                continue
            # print("la", start, end)
            startr = max(start, startv)
            endr = min(end, endv)

            y[int((startr - start) / resolution):int((endr - start) / resolution)] = CNV
        if oData:
            return data
        else:
            return x, y

    if experiment == "NFR":

        index = ["chrom", "chromStart", "chromEnd"]
        chro = str(chromosome)
        print(files)
        strain = pd.read_csv(files[0], sep="\t", names=index, skiprows=1)

        data = strain[(strain.chrom == "chr%s" % chro) & (
            strain.chromStart > 1000 * start) & (strain.chromStart < 1000 * end)]

        if oData:
            return data

        x = np.array(data.chromStart / 2 + data.chromEnd / 2) / 1000  # kb
        y = np.ones_like(x)

    if experiment == "Bubble":

        index = ["chrom", "chromStart", "chromEnd", "signalValue"]
        chro = str(chromosome)
        print(files)
        strain = pd.read_csv(files[0], sep="\t", names=index, skiprows=1)

        data = strain[(strain.chrom == "chr%s" % chro) & (
            strain.chromStart > 1000 * start) & (strain.chromStart < 1000 * end)]

        x = np.array(data.chromStart / 2 + data.chromEnd / 2) / 1000  # kb
        y = np.array(data.signalValue)

    if experiment == "ORC1":

        index = ["chrom", "chromStart", "chromEnd"]
        chro = str(chromosome)
        print(files)
        strain = pd.read_csv(files[0], sep="\t", names=index, skiprows=1)

        data = strain[(strain.chrom == "chr%s" % chro) & (
            strain.chromStart > 1000 * start) & (strain.chromStart < 1000 * end)]

        x = np.array(data.chromStart / 2 + data.chromEnd / 2) / 1000  # kb
        y = np.ones_like(x)

    if (experiment in ["Mcm3","Mcm7","Orc2","Orc3"]) and strain =="Raji":
        print(files)
        for f in files:
            if "chr%s_" % str(chromosome) in f:
                # print(f)
                data = pd.read_csv(f)
                input = pd.read_csv(f.replace(experiment,"input"))
                break

        data=np.array(data[start:end],dtype=float)[::,0]
        input = np.array(input[start:end],dtype=float)[::,0]
        print(data.shape)
        print(data)
        x=np.arange(start,end)
        x0,data = re_sample(x, data, start, end, resolution)
        _,input = re_sample(x, input, start, end, resolution)
        data = data/input
        data[input<10]=np.nan
        return x0,data


    if experiment == "SNS":
        if strain == "K562":
            index = ["chrom", "chromStart", "chromEnd"]
            chro = str(chromosome)
            print(files)
            strain = pd.read_csv(files[0], sep="\t", names=index, skiprows=1)

            data = strain[(strain.chrom == "chr%s" % chro) & (
                strain.chromStart > 1000 * start) & (strain.chromStart < 1000 * end)]

            if oData:
                return data

            x = np.array(data.chromStart / 2 + data.chromEnd / 2) / 1000  # kb
            y = np.ones_like(x)
        else:
            index = ["chrom", "chromStart", "chromEnd","signalValue"]

            chro = str(chromosome)
            print(files)
            strain = pd.read_csv(files[0], sep="\t", names=index, skiprows=1)

            data = strain[(strain.chrom == "chr%s" % chro) & (
                    strain.chromStart > 1000 * start) & (strain.chromStart < 1000 * end)]

            x = np.array(data.chromStart / 2 + data.chromEnd / 2) / 1000  # kb
            y = data.signalValue

    if experiment == "MCMo":

        index = ["chrom", "chromStart", "chromEnd"]
        chro = str(chromosome)
        print(files)
        strain = pd.read_csv(files[0], sep="\t", names=index)

        data = strain[(strain.chrom == "chr%s" % chro) & (
            strain.chromStart > 1000 * start) & (strain.chromStart < 1000 * end)]

        if oData:
            return data

        x = np.array(data.chromStart / 2 + data.chromEnd / 2) / 1000  # kb
        y = np.ones_like(x)
        print(sum(y), len(y))

    if experiment == "Meth":

        index = ["chrom", "chromStart", "chromEnd", "name", "score",
                 "strand", "chromStart1", "chromEnd1", "bs", "signalValue", "p"]
        chro = str(chromosome)
        strain = pd.read_csv(files[0], sep="\t", names=index, skiprows=1)

        data = strain[(strain.chrom == "chr%s" % chro) & (
            strain.chromStart > 1000 * start) & (strain.chromStart < 1000 * end)]

        if oData:
            return data

        x = np.array(data.chromStart / 2 + data.chromEnd / 2) / 1000  # kb
        y = np.array(data.signalValue)

    if experiment == "Meth450":
        # chr16	53468112	53468162	cg00000029	721	+	53468112	53468162	255,127,0

        index = ["chrom", "chromStart", "chromEnd", "name", "signalValue",
                 "strand", "chromStart1", "chromEnd1", "bs"]
        chro = str(chromosome)
        strain = pd.read_csv(files[0], sep="\t", names=index, skiprows=1)

        data = strain[(strain.chrom == "chr%s" % chro) & (
            strain.chromStart > 1000 * start) & (strain.chromStart < 1000 * end)]

        if oData:
            return data

        x = np.array(data.chromStart / 2 + data.chromEnd / 2) / 1000  # kb
        y = np.array(data.signalValue)

    if experiment == "Constant":
        y = np.zeros(int(end / resolution - start / resolution)) + 1
        x = np.arange(len(y)) * resolution + start
        return x, y

    if experiment == "ORC2":

        strain = pd.read_csv(files[0],
                             skiprows=2, names=["chrom", "chromStart", "chromEnd"])

        chro = str(chromosome)
        data = strain[(strain.chrom == "chr%s" % chro) & (
            strain.chromStart > 1000 * start) & (strain.chromStart < 1000 * end)]

        if oData:
            return data

        x = np.array(data.chromStart / 2 + data.chromEnd / 2) / 1000  # kb
        y = np.array([1 for _ in range(len(x))])



    if experiment == "ExpGenes":

        strain = pd.read_csv(files[0],
                             skiprows=1, names=["chrom", "chromStart", "chromEnd", "signalValue", "gene_id", "tss_id"], sep="\t")

        chro = str(chromosome)
        # print(strain)
        # return strain
        data = strain[(strain.chrom == "chr%s" % chro) & (
            strain.chromStart > 1000 * start) & (strain.chromStart < 1000 * end)]
        data["chromStart"] /= (1000 * resolution)
        data["chromEnd"] /= (1000 * resolution)

        db = gffutils.FeatureDB(
            ROOT + "/external/ExpressedGenes/db_gtf.db", keep_order=True)

        sens = [db[gene].strand for gene in data["gene_id"]]
        data["strand"] = sens
        if oData or raw:
            return data
        else:
            raise "Only raw available"
        # x = np.array(data.chromStart / 2 + data.chromEnd / 2) / 1000  # kb
        # y = np.array([1 for _ in range(len(x))])

    if experiment[:-3] in marks and strain == "IMR90":

        strain = pd.read_csv(files[0],sep="\t")

        chro = str(chromosome)
        data = strain[(strain.chrom == "chr%s" % chro) & (
                strain.chromStart > 1000 * start) & (strain.chromStart < 1000 * end)]

        if oData:
            return data

        x = np.array(data.chromStart / 2 + data.chromEnd / 2) / 1000  # kb
        y = np.array(data.signalValue)
        return x,y
    if experiment[:-3] in marks:

        cell = pyBigWig.open(files[0])
        if end is None:
            end = cell.chroms()['chr%s' % str(chromosome)]

        # check end of chromosome:

        true_end = cell.chroms()['chr%s' % str(chromosome)]
        true_end /= 1000
        pad = False
        if true_end < end:
            print("Problem of size", true_end, end, chromosome)
            pad = end + 0

            end = int(true_end)
        # print("Problem of size", true_end, end, chromosome)
        #print(chromosome)
        #print(cell.chroms())
        #print(files[0])
        #print(start,end)
        v = [np.nan if s is None else s for s in cell.stats(
            "chr%s" % str(chromosome), int(start * 1000), int(end * 1000), nBins=int(int(end - start) // (resolution)))]
        if pad:
            print("padding...", int(pad/resolution)-len(v))

            v += [np.nan]*(int(pad/resolution)-len(v))

        return np.arange(start, end + 100, resolution)[:len(v)], np.array(v)

    if experiment in marks:
        index = ["chrom", "chromStart", "chromEnd", "name",
                 "score", "strand", "signalValue", "pValue", "qValue"]

        strain = pd.read_csv(files[0], sep="\t", names=index)

        chro = str(chromosome)
        if oData:
            return strain
        data = strain[(strain.chrom == "chr%s" % chro) & (
            strain.chromStart > 1000 * start) & (strain.chromEnd < 1000 * end)]

        x = np.array(data.chromStart / 2 + data.chromEnd / 2) / 1000  # kb
        y = np.array(data.signalValue)
        xp, yp = re_sample(x, y, start, end, resolution)

        if not bp:
            return xp, yp

        # if bpc:
        #    x = np.array(data.chromStart / 2 + data.chromEnd / 2) / 1000  # kb
        #    y = np.array(data.chromEnd - data.chromStart)
        #    xp, yp = re_sample(x, y, start, end, resolution)
        #    return xp, yp

        yp = np.zeros_like(yp)
        for startv, endv, v in zip(data["chromStart"], data["chromEnd"], data["signalValue"]):
            # if endv < 6000000:
            #        print(startv, endv, v)
            endv1 = min(endv, end * 1000)
            if endv1 != endv:
                print("End of peak outside", endv, end)
                endv = endv1
                print(endv, startv)

            start1 = min(startv, end * 1000)
            if start1 != startv:
                print("Start of peak outside", startv, end)
                startv = start1
                # print(endv, startv)

            deltas, indexes = overlap_fraction(startv / 1000, endv / 1000, resolution)
            deltas /= np.sum(deltas)
            if np.any(indexes - int(start / resolution) >= len(yp)):
                print("Out of bound")
                continue
            if bpc:
                yp[indexes - int(start / resolution)] += deltas * (endv - startv)  # * v
            else:
                yp[indexes - int(start / resolution)] += deltas * \
                    v * (endv-startv)/(resolution*1000)

            # print(startv, endv)
            # print(deltas, indexes, start, indexes - int(start / resolution))
            # return xp, yp
        yp[yp == 0] = np.nan
        return xp, yp

    if experiment == "MRT":
        if strain == "Cerevisae":
            index = ["i", "chrom", "chromStart",
                     "%HL: tp10", "%HL: tp12.5A", "%HL: tp12.5B", "%HL: tp15A", "%HL: tp15B", "%HL: tp17.5A",
                     "%HL: tp17.5B", "%HL: tp25A", "%HL: tp25B", "%HL: tp40A", "%HL: tp40B", "TrepA", "TrepB"]

            data = pd.read_csv(files[0], sep=",", names=index, skiprows=1)

            mini = np.min(data.TrepA/2+data.TrepB/2)
            maxi = np.max(data.TrepA/2+data.TrepB/2)
            # print(data.head())
            data = data[["chrom", "chromStart", "TrepA", "TrepB"]]
            data = data[(data.chrom == chromosome) & (
                data.chromStart > start) & (data.chromStart < end)]
            x = np.array(data.chromStart)
            # print(x)
            # exit()

            y = (data.TrepA/2+data.TrepB/2-mini)/(maxi-mini)
            # return ,

        else:
            #print(files)
            for f in files:
                if "Rep1" in f and "chr%s.dat" % str(chromosome) in f:
                    # print(f)
                    data = pd.read_csv(f)

                    break
                elif (strain == "Raji") and "chr%s_" % str(chromosome) in f:
                    print(f)
                    data = pd.read_csv(f)
                    data=2**data
                    #data=np.exp(data)/35
                    break
            #print(len(data))
            data[data < 0] = np.nan
            data = np.array(data)
            data = np.concatenate([np.array([data[0], data[0], data[0], data[0]]), data]) # Because centered at 5 kb
            y = np.array(data[int(start / 10): int(end / 10)])
            x = np.arange(len(y)) * 10 + start
            assert(len(x) == len(y))
    if experiment == "MRTstd":

        for f in files:
            if "Rep1" in f and "chr%s.dat" % str(chromosome) in f:
                # print(f)
                data = pd.read_csv(f,sep="\t")
                print(data.shape)
                break
        data = data.apply(pd.to_numeric, args=('coerce',))
        #data[data < 0] = np.nan
        data = np.array(data)
        data = np.concatenate([np.array([data[0], data[0], data[0], data[0]]), data],axis=0)
        mask = np.array(data[::,-1],dtype=np.bool)
        data[mask,::]==np.nan
        print(data.shape)
        y = np.array(data[int(start / 10): int(end / 10)])[::,:6]

        def delta_std(x):
            time = np.array([0, 1, 2, 3, 4, 5])/5
            time = time[np.newaxis,::] * np.ones((len(x)))[::,np.newaxis]
            mrt = np.sum(x*time, axis=1)[::,np.newaxis]
            return np.sum(x*(time-mrt) ** 2, axis=1) ** 0.5
        y = delta_std(y)
        x = np.arange(len(y)) * 10 + start
        assert (len(x) == len(y))

    if experiment == "RNA_seq":
        # print(files)
        for f in files:
            if "chr%s_" % str(chromosome) in f:
                # print(f)
                data = pd.read_csv(f)

                break

        data[data < 0] = np.nan
        data = np.array(data)
        #data = np.concatenate([np.array([data[0], data[0], data[0], data[0]]), data])
        y = np.array(data[int(start ): int(end )])
        x = np.arange(len(y)) * 1 + start
        assert(len(x) == len(y))

    if experiment.startswith("OKSeq"):
        # print(files)
        if strain == "Cerevisae":
            chromosome = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X",
                          11: "XI", 12: "XII", 13: "XIII", 14: "XIV", 15: "XV", 16: "XVI"}[chromosome]
        # print("chr%s.pol" % str(chromosome))
        for f in files:
            #print("File", f)

            if experiment == "OKSeqo" and "chr%s.pol" % str(chromosome) in f:
                data = pd.read_csv(f, names=["pol"])
                break
            if experiment == "OKSeqR" and "chr%s.R" % str(chromosome) in f:
                data = pd.read_csv(f, names=["R"])
                break
            if experiment == "OKSeqF" and "chr%s.F" % str(chromosome) in f:
                data = pd.read_csv(f, names=["F"])
                break
        if experiment in ["OKSeqS", "OKSeq","OKSeqF","OKSeqR"]:
            for f in files:

                if "chr%s.F" % str(chromosome) in f:
                    print(f)
                    data1 = pd.read_csv(f, names=["F"])
                    break
            for f in files:
                if "chr%s.R" % str(chromosome) in f:
                    data2 = pd.read_csv(f, names=["R"])
                    break
            # print(data2)

            if experiment == "OKSeqS":
                x = np.arange(start, end, 1)
                y = np.array(data1[start: end].F + data2[start: end].R)
            elif experiment == "OKSeq":
                #med = np.median(np.array(data1[start: end].F + data2[start: end].R))
                x = np.arange(start, end, 1)
                subR = data2[start: end].R
                #subR[subR<10]=np.nan
                subF = data1[start: end].F
                #subF[subF<10]=np.nan

                x0, R = re_sample(x, subR, start, end, resolution)
                x0, F = re_sample(x, subF, start, end, resolution)
                RFD = (R-F)/(R+F) * resolution
                med = np.median(R + F)/2
                #print(med)
                RFD[R < med/10] = np.nan
                RFD[F <  med/10] = np.nan
                #print("Nan",np.sum(np.isnan(RFD)))
                return x0, RFD
            elif experiment in ["OKSeqF", "OKSeqR"]:
                if experiment == "OKSeqF":
                    x = np.arange(start, end, 1)
                    y = np.array(data1[start: end].F)
                else:
                    x = np.arange(start, end, 1)
                    y = np.array(data2[start: end].R)
                return x,y

        else:
            # data[data<0] = np.nan
            if strain == "Cerevisae":
                # print("La")
                data1 = pd.read_csv(f.replace(".pol", ".R"), names=["R"])
                data2 = pd.read_csv(f.replace(".pol", ".F"), names=["F"])

                med = np.median(data1.R+data2.F)
                std = np.std(data1.R+data2.F)
                #print(med-2 * std)
                data.pol[((data1.R+data2.F) == 0) | ((data1.R+data2.F) < (10))] = np.nan
                # print(,)
                y = np.array(data[start: end])
                # print(np.sum(np.isnan(y)))

                #y = np.cumsum(y)+30
                x = np.arange(start, end, 1)

            else:

                x = np.arange(start, end, 1)
                y = np.array(data.pol[start: end])
        # print(len(data))
        # print(files)
        # print(data)
        try:
            assert(len(x) == len(y))
        except:
            if pad:
                yp = np.zeros_like(x) + np.nan
                if len(y.shape) == 2:
                    yp[:len(y)] = y[::, 0]
                else:
                    yp[:len(y)] = y
                y = yp
            else:
                print(len(x), len(y))
                raise

    if resolution < resolution_experiment:
        print("Warning experimental resolution lower than required resolution")
        raise "not implemented"

    if raw:
        return x, y
    else:
        return re_sample(x, y, start, end, resolution)
