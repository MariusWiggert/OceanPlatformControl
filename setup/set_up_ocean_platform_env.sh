#!/usr/bin/env bash

conda init bash

# create a conda environment called ocean_platform
conda create -n ocean_platform -c conda-forge python=3.6 parcels cartopy

# download & install the modified parcels version from Marius Github
conda activate ocean_platform
conda remove --force parcels
pip install git+https://github.com/MariusWiggert/parcels.git@master

# install other python requirements
pip install -r ./requirements_new.txt

