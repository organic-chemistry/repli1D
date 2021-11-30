
repli1D
=======
Code used to simulate replication and to inverse MRT and RFD profiles to obtain
a Probability of Origin Density Lansdcape (PODLS)


Install
===========

```
git clone git@github.com:organic-chemistry/repli1D.git
cd  repli1D
conda create -c bioconda -n repli  keras tensorflow snakemake numpy joblib plotly
conda activate repli
python setup.py develop
```

Replicate results of the article
===========
data for K562 GM Hela and Cerevisae are available in data/ directory at a 5000 bp resolution for human cells and 1000 bp resolution for Cerevisae.
For each cell line, (here Hela for example) the following command:
```
snakemake -c 4 Hela_5000.sh --config output="my-folder/"
```
will create the file Hela_5000.sh needed to perform the optimisation. The paramater
-c 4 set the number of threads used perform the replication.
Then running
```
sh Hela.sh
```
will cary on the optimisation.
For each iteration i, it will create three folders (inside my-folder):
 - _RFD_to_init_small_opti_i # optimisation on the chromosome 2
 - wholecell_i       # the result of the whole cell simulation using I_i and the parameters from _RFD_to_init_small_opti_i/params.json
 - RFD_to_init_nn_i+1 # with the neural network that ouput I_i+1 in the file nn_global_profiles.csv.


Running the code on a new data (refered as celltype)
===========
This require to create a data file in the data folder name celltype_5000.csv with 5000 the resolution of the bins.
It is a csv file with column: chrom,chromStart,chromEnd,MRT,OKSeq
Then a new entry cellline must be added in the config.yaml file (by copying K562 of Cerevisae for example) and by adapting the parameter to the organism.
Then running,
```
snakemake -c 4 celltype_5000.sh --config output="my-folder/"
```
will generate the script celltype.sh that must be run to carry on the optimisation.
The paramater -c 4 set the number of threads used perform the replication

Description
===========

The code to simulate replication and obtain RFD and MRT, given an input PODLS is in src/repli1d/fast_sim_break.py

example of use are in notebook_demo:

[Typical simulation and comparison with experiment](notebook_demo/Typical_sim.ipynb)

[Example with casadet](notebook_demo/Sim_with_cascade.ipynb)



To inverse:
```
python src/repli1d/whole_pipeline_from_data_file.py --root /scratch/jarbona/yeast-inverse/ --speed 2.5 --max_epoch 2000 --repTime 30 --chr_sub 4 --window 41 --data data/yeast.csv  --introduction_time 5 --resolution 1
```

```
python src/repli1d/visu_signals.py --cell Hela --signal MRT , OKSeq , MCM
python src/repli1d/detect_and_simulate.py --cell Raji --ndiff 60 --experimental --name tmp/ --visu --signal peakRFDonly
```

```
python src/repli1d/detect_and_simulate.py --cell Cerevisae --signal MCM-beda  --experimental --name tmp/ --visu --resolution 5 --fspeed 0.24 --introduction_time 1 --mrt_res 5 --input --dori 1 --masking 25 --noise 0.0 --nsim 300 --wholecell --ndiff 20
```

```
python src/repli1d/whole_pipeline.py --cell Raji --savenoise --add_noise --root results/Raji_mask_bothopti_snoise_nnoise_std
```
