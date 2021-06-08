
repli1D
=======


Install
===========

```
git clone git@github.com:organic-chemistry/repli1D.git
cd  repli1D
conda create --name repli --file spec-file.txt
conda activate repli
python setup.py develop

```

Description
===========

The code for the simulation of the replication is in src/repli1d/fast_sim_break.py

example of use are in notebook_demo:

[Typical simulation and comparison with experiment](notebook_demo/Typical_sim.ipynb)

[Example with casadet](notebook_demo/Sim_with_cascade.ipynb)
