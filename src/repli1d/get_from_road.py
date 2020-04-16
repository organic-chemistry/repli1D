import os
import subprocess

cell = {"K562":"E123",
        "GM12878" :  "E116" ,
        "Hela"  :   "E115" ,
        "IMR90" :   "E017"}

marks = [ "H3K4me1", "H3K4me3", "H3K27me3", "H3K36me3", "H3K9me3"]

marks += ["H2A.Z","H3K79me2","H3K9ac","H3K4me2"]

marks += ["H3K27ac","H4K20me1"]



root = "/mnt/data/data/roadmap/"
import time
"""
for k,v in cell.items():
    local_root = root + "/%s"%k
    os.makedirs(local_root,exist_ok=True)
    for mark in marks:
        # E123-H3K36me3.tagAlign
        file = "%s-%s.tagAlign.gz" % (v,mark)
        cmd = ["wget https://egg2.wustl.edu/roadmap/data/byFileType/alignments/consolidated/%s -O %s/%s" % (file,local_root,file)]
        print(cmd)
        #os.popen(cmd)
        compressed = "%s/%s" % (local_root,file)
        cmd += ["dtrx %s"%compressed]
        cmd += ["mv %s %s"%(file[:-3],compressed[:-3])]
        cmd += ["python src/repli1d/convert_Bw_to_csv.py --globalonly --file %s --output %s --resolution 5" % (compressed[:-3],local_root+"/%s_%s.csv"%(k,mark))]
        #cmd += ["rm %s"%compressed[:-3]]
        #print(cmd)
        for cm in cmd:
            print(cm)

            #process = subprocess.Popen(cm, shell=True, stdout=subprocess.PIPE)
            #process.wait()

    cm = "python src/repli1d/concat_and_rename_road.py --root /mnt/data/data/roadmap//%s/ --output /mnt/data/data/roadmap//%s/input_road.csv"%(k,k)
    print(cm)
    process = subprocess.Popen(cm, shell=True, stdout=subprocess.PIPE)
    process.wait()
            #os.popen(cm)
        #break
        #time.sleep(10*60)
"""
    #break
        #c
print("End")

cell ={"IMR90" :   "E017"}

for k,v in cell.items():
    local_root = root + "/%s"%k
    os.makedirs(local_root,exist_ok=True)
    for mark in marks:
        # E123-H3K36me3.tagAlign
        file = "%s-%s.broadPeak.gz" % (v,mark)
        cmd = ["wget https://egg2.wustl.edu/roadmap/data/byFileType/peaks/consolidated/broadPeak//%s -O %s/%s" % (file,local_root,file)]
        print(cmd)
        #os.popen(cmd)
        compressed = "%s/%s" % (local_root,file)
        cmd += ["dtrx %s"%compressed]
        compressed = compressed.replace(v,k)
        compressed = compressed.replace("K","k")
        cmd += ["mv %s %s"%(file[:-3],compressed[:-3])]

        #cmd += ["python src/repli1d/convert_Bw_to_csv.py --globalonly --file %s --output %s --resolution 5" % (compressed[:-3],local_root+"/%s_%s.csv"%(k,mark))]
        #cmd += ["rm %s"%compressed[:-3]]
        #print(cmd)
        for cm in cmd:
            print(cm)

            process = subprocess.Popen(cm, shell=True, stdout=subprocess.PIPE)
            process.wait()

