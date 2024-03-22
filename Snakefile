import snakemake
if int(snakemake.__version__[:1])<5:
   print("Old version",snakemake.__version__)
   def directory(val):
       return val


configfile: "config.yaml"

rule all:
    input:
        expand("{cell}_{resolution}.sh",cell="yeast",resolution=1)
rule create_dataset:
    output:
        "data/{cell}_{resolution}.csv"
    shell:
        "python src/repli1d/prepare_data.py --cell {wildcards.cell} --name {output} --resolution {wildcards.resolution}"

def isit(what="test",conf=None):
    if conf is None:
        conf = config
    if int(conf.get(what,0))==1:
        return f" --{what} "
    return ""
def reduce_lr(w):
    if int(config[w.cell]["reduce_lr"])==1:
        return "--reduce_lr"
    return ""


def get_input(w):
    print(w)
    inputs = [ config["data"]+f"/{w.cell}_{w.resolution}.csv"]
    if w.signal != "":
        inputs.append(f"{w.signal}")
    #print(inputs)
    return inputs

def on_input_signal(w):
    if w.signal == "":
        return ""
    else:
        inp = w.signal
        return f"--on_input_signal {inp} --sm 10"
rule prepare_script:
    input:
        get_input
    output:
        "{cell}_{resolution,\d+}{signal}.sh"
    params:
        fork_speed=lambda w: config[w.cell]["fork_speed"] ,
        replication_time=lambda w: config[w.cell]["replication_time"] ,
        chr_sub=lambda w: config[w.cell]["chr_sub"] ,
        introduction_time=lambda w: config[w.cell]["introduction_time"] ,
        window=lambda w: int(config[w.cell]["typical_domain_size"] // int(w.resolution)) ,# in bn
        masking=lambda w: int(0.5*config[w.cell]["typical_domain_size"] / 1000), # in kb
        cut_holes=lambda w: int(1.5*config[w.cell]["typical_domain_size"] / 1000) , # in kb
        nfilters=lambda w: config[w.cell]["nfilters"] ,
        dir =lambda w:config["output"]+f"/{w.cell}_{w.resolution}_inverse" ,
        res = lambda w:int(w.resolution) // 1000 ,
        options = lambda w : isit("test") + isit("pearson") + isit("save") + isit("exclude_noise")+ isit("grid_rfd_opti_only") + isit("RFDonly") + isit("logr",conf=config[w.cell]),
        reduce_lr = lambda w : reduce_lr(w) ,
        max_factor_reptime = lambda w: config[w.cell]["max_factor_reptime"],
        percentile = lambda w: config[w.cell]["percentile"],
        dori=20,
        on_input_signal =  lambda w: on_input_signal(w)

    wildcard_constraints:
        signal=".*" ,
        cell="[^/]*"

    threads: 32
    shell:
        "python src/repli1d/whole_pipeline_from_data_file.py --name_script {output} --root {params.dir} --data {input[0]} "
        "--speed {params.fork_speed} --max_epoch 50 --repTime {params.replication_time} --chr_sub {params.chr_sub} --window {params.window}"
        " --introduction_time {params.introduction_time} --resolution {params.res} --masking {params.masking} --cut_holes {params.cut_holes} "
        "--nfilters {params.nfilters} --threads {threads} {params.options} "
        "--max_factor_reptime {params.max_factor_reptime} {params.reduce_lr} --percentile {params.percentile} --dori {params.dori}"
        " {params.on_input_signal}"

        #RFDonly refers to the initial configuration computer ond delta of RFD


# rule prepare_script_on_input:
#     input:
#         dataf = config["data"]+"/{cell}_{resolution}.csv" ,
#         signal = config["data"]+"/{signalroot}.csv"
#     output:
#         "{cell}_{resolution,\d+}_{signalroot}.sh"
#     params:
#         fork_speed=lambda w: config[w.cell]["fork_speed"] ,
#         replication_time=lambda w: config[w.cell]["replication_time"] ,
#         chr_sub=lambda w: config[w.cell]["chr_sub"] ,
#         introduction_time=lambda w: config[w.cell]["introduction_time"] ,
#         window=lambda w: int(2*config[w.cell]["typical_domain_size"] // int(w.resolution)) ,# in bn
#         masking=lambda w: int(0.5*config[w.cell]["typical_domain_size"] / 1000), # in kb
#         cut_holes=lambda w: int(1.5*config[w.cell]["typical_domain_size"] / 1000) , # in kb
#         nfilters=lambda w: config[w.cell]["nfilters"] ,
#         dir =lambda w:config["output"]+f"/{w.cell}_{w.resolution}_{w.signalroot}" ,
#         res = lambda w:int(w.resolution) // 1000 ,
#         options = lambda w : isit("test") + isit("pearson") + isit("save") + isit("exclude_noise") + isit("RFDonly"),
#         reduce_lr = lambda w : reduce_lr(w) ,
#         max_factor_reptime = lambda w: config[w.cell]["max_factor_reptime"],
#         percentile = lambda w: config[w.cell]["percentile"],

#     threads: 8
#     shell:
#         "python src/repli1d/whole_pipeline_from_data_file.py --name_script {output} --root {params.dir} --data {input.dataf} "
#         "--speed {params.fork_speed} --max_epoch 2000 --repTime {params.replication_time} --chr_sub {params.chr_sub} --window {params.window}"
#         " --introduction_time {params.introduction_time} --resolution {params.res} --masking {params.masking} --cut_holes {params.cut_holes} "
#         "--nfilters {params.nfilters} --threads {threads} {params.options} "
#         "--max_factor_reptime {params.max_factor_reptime} {params.reduce_lr} --percentile {params.percentile} --on_input_signal {input.signal} --sm 10"




rule select_opti:
    input:
        rep="{name}"
    output:
        rep=directory("{name}/opti")
    run:
        import glob
        import pandas as pd
        wholecells=glob.glob(f"{input}/wholecell_*")
        maxi=0
        dir=None
        for wholecell in wholecells:
            data= pd.read_csv(wholecell+"/summary.csv")
            pearson = data["MRTp"].iloc[0]+data["RFDp"].iloc[0]
            print(wholecell,"pearson: ",pearson)
            if pearson>maxi:
                dir=wholecell
                maxi=pearson
        dir = dir.split("/")[-1]
        if dir == "":
            dir =dir.split("/")[-2]
        print("Maxi", dir)
        shell(f"cd {input} && ln -s {dir} opti")

rule create_opti_all:
    input:
        [f"results/{name}_5000_inverse/opti" for name in ["GM","Hela","K562"]]

marks =[ "H3K4me1", "H3K4me3", "H3K27me3", "H3K36me3", "H3K9me3",
        "H2A.Z","H3K79me2","H3K9ac","H3K4me2","H3K27ac","H4K20me1"]

list_cells =  ["K562","Hela","GM06990"]

def get_input_sp(w):
    return [ config["data"]+f"/{cell}_{w.res}_merged_histones_init.csv.gz"  for cell in list_cells]

def get_input(w):
    if w.cell == "all":
        return [ config["data"]+f"/{cell}_{w.res}_merged_histones_init.csv.gz"  for cell in list_cells]
    else:
        return config["data"]+f"/{w.cell}_{w.res}_merged_histones_init.csv.gz" 

#print(config)

rule mark_ml:
    input:
        inputp = get_input,
        inputsp = get_input_sp
        #Files created by snakemake -pf -c 1 create_all_merge --use-conda
    output:
        #nnweight=config["output"]+"/nn_{cell}_{res}/Noneweights.hdf5" ,
        #data=config["output"]+"/nn_{cell}_{res}/nn_{cell}_from_{cell}.csv"
        "gpu_nn_{cell}_{res}.sh"
    params:
        marks = lambda w :" ".join(marks) ,
        dir=lambda w : config["output"] + f"/nn_{w.cell}_{w.res}/" , 
        predict_files = lambda w : " ".join([config["data"]+f"/{c}_{w.res}_merged_histones_init.csv.gz" for c in list_cells]) 
    shell:
        "echo python src/repli1d/nn.py  --noenrichment --targets initiation "
        "--root {params.dir} --listfile {input.inputp}  --window 101 --wig 0 "
        "--predict_files {params.predict_files} --marks {params.marks} --cell {wildcards.cell} --datafile --sub 1 > {output}"

rule test_ml:
    input:
        data=config["output"]+"/nn_{cell}_{res}/nn_{cell}_from_None.csv"
    output:
        data=config["data"]+"/nn{cell}res{res}fromNone.csv"
    shell:
        "cp {input} {output}"

# example to run
rule run_sh_on_input:
    input:
        "{cell}_{resolution}_{signalroot}.sh"
    output:
        config["output"]+"/{cell}_{resolution,\d+}_{signalroot}/wholecell_0/global_profiles.csv"
    threads:8
    shell:
        "sh {input}"