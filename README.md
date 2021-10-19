
repli1D
=======
Code used to simulate replication and to inverse MRT and RFD profiles to obtain
a Pobrability of Origin Density Lansdcape (PODLS)


Install
===========

```
git clone git@github.com:organic-chemistry/repli1D.git
cd  repli1D
conda create --name repli --file spec-file.txt
conda activate repli
python setup.py develop
conda install plotly
```

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
