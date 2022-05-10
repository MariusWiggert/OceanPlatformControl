#!/usr/bin/env zsh
conda init zsh

# create a conda environment called ocean_platform
conda create -n ocean_platform python=3.9.*
conda activate ocean_platform

# these packages have to be installed manually (otherwise pip fails to install dependencies)
conda install -c conda-forge cartopy ffmpeg

# install jax library for cpu or gpu
pip install --upgrade "jax[cpu]"
# pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

pip install ray[rllib]==1.12.0

# install private hj_reachability library fork via token
pip install --upgrade git+https://dti-devops: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@github.com/MariusWiggert/hj_reachability_c3.git@4549bea7c3cda3b4d3bf90735bc1cba4703fb4ca

# install other python requirements via pip
pip install -r requirements.txt

