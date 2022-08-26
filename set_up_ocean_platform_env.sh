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
pip install --upgrade git+https://dti-devops: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@github.com/MariusWiggert/hj_reachability_c3.git@4549bea7c3cda3b4d3bf90735bc1cba4703fb4ca
# To avoid the dimension bug with Sklearn 1.0.2 GPRegressor dimension (Probably not necessary when a new release available)
# pip install --upgrade git+https://github.com/scikit-learn/scikit-learn.git@e358bd77e0cde248e0ee8f67c29a72e330fcc0fe

# install other python requirements via pip
pip install -r requirements.txt

# install c3python
pip install git+https://github.com/c3aidti/c3python

# install local files in developer mode
pip install -e .

