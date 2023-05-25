#!/usr/bin/env zsh -i
# create new conda env and activate it
conda create -n phykos_jupyter_env python=3.9 -y
conda activate phykos_jupyter_env

# install general requirements
conda install -y -c jupyter cartopy ffmpeg nomkl
pip install --upgrade pip
pip install git+https://github.com/c3aidti/c3python@0e87998fbdd22ceae6a17edd15e2b6fbf6569cae
pip install -r setup/requirements_minimal.txt

# hj_reachability needs to be installed after jax
pip install --upgrade git+https://MariusWiggert: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@github.com/MariusWiggert/hj_reachability_c3.git
# install the ocean_navigation_simulator as library
pip install git+https://MariusWiggert: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@github.com/MariusWiggert/OceanPlatformControl@38a4e4bdfb53bc7c873740b24d0bd2bc17358c00