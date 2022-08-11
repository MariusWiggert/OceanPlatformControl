#!/bin/bash


# deactivate all conda environments and create conda environment 'ocean_platform'
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

# install outdated jax==0.24 for hj_reachability
#pip install --upgrade "jax[cpu]==0.2.24" -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install --upgrade "jax[cuda]==0.2.24" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# install newest tensorflow after jax since it has to replace dependencies
pip install tensorflow>=2.9.1

# install Ray Lib with Dashboard and RL-lib
pip install -U ray[default]==1.13.0
pip install -U ray[rllib]==1.13.0

# install c3 type for python
pip install git+https://github.com/c3aidti/c3python

# install private hj_reachability library fork via token
pip install --upgrade git+https://dti-devops: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@github.com/MariusWiggert/hj_reachability_c3.git

# install other python requirements via pip
pip install -r setup/requirements.txt