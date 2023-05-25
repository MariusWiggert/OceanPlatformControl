#!/usr/bin/env zsh

conda install -y -c jupyter cartopy ffmpeg nomkl
pip install --upgrade pip
pip install git+https://github.com/c3aidti/c3python@0e87998fbdd22ceae6a17edd15e2b6fbf6569cae
pip install -r setup/requirements_minimal.txt

# hj_reachability needs to be installed after jax
pip install --upgrade git+https://MariusWiggert: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@github.com/MariusWiggert/hj_reachability_c3.git
pip install -e .