#!/bin/bash


# create a conda environment called ocean_platform and delete deactivate all others
conda init bash > /dev/null
for i in $(seq ${CONDA_SHLVL}); do
    conda deactivate
done
conda create -y -n ocean_platform python=3.9.*
conda activate ocean_platform

# these packages have to be installed manually (otherwise pip fails to install dependencies)
conda install -y -c conda-forge cartopy ffmpeg gcc=12.1.0

# update pip
pip install --upgrade pip

# install jax library for cpu or gpu
pip install --upgrade "jax[cpu]==0.2.24"
#pip install --upgrade "jax[cuda]==0.2.24" -f https://storage.googleapis.com/jax-releases/jax_releases.html

# install Ray Lib with Dashboard and RL-lib
pip install -U ray[default]==1.13.0
pip install -U ray[rllib]==1.13.0

# install c3 type for python
pip install git+https://github.com/c3aidti/c3python

# install private hj_reachability library fork via token
pip install --upgrade git+https://dti-devops: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@github.com/MariusWiggert/hj_reachability_c3.git@4549bea7c3cda3b4d3bf90735bc1cba4703fb4ca
# To avoid the dimension bug with Sklearn 1.0.2 GPRegressor dimension (Probably not necessary when a new release available)
# pip install --upgrade git+https://github.com/scikit-learn/scikit-learn.git@e358bd77e0cde248e0ee8f67c29a72e330fcc0fe

# install other python requirements via pip
pip install -r setup/requirements.txt