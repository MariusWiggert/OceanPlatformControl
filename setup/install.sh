#!/usr/bin/env zsh

conda install -y -c jupyter cartopy ffmpeg
pip install --upgrade pip
pip install git+https://github.com/c3aidti/c3python
pip install -r setup/requirements_minimal.txt

# hj_reachability needs to be installed after jax
pip install --upgrade git+https://MariusWiggert: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@github.com/MariusWiggert/hj_reachability_c3.git
pip install -e .