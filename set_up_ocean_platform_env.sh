#!/usr/bin/env zsh
conda init zsh
# create a conda environment called ocean_platform
 conda create -n ocean_platform python=3.9

conda activate ocean_platform
conda install -c conda-forge cartopy ffmpeg

# install jax library for cpu or gpu
pip install --upgrade "jax[cpu]"
# pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
# install private hj_reachability library fork via token
pip install --upgrade git+https://MariusWiggert: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@github.com/MariusWiggert/hj_reachability_c3.git

# install other python requirements via pip
pip install -r requirements.txt

