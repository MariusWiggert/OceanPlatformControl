#!/usr/bin/env zsh
#conda init zsh

# Create conda environment outside
# conda create -y -n ocean_minimal python=3.9
# conda activate ocean_minimal

# Setup ready
conda install -y -c jupyter cartopy ffmpeg
pip install --upgrade pip
pip install git+https://github.com/c3aidti/c3python
pip install -r setup/requirements_minimal_m1.txt 

# build casadi from source
git clone https://github.com/casadi/casadi.git casadi                                                                                
cd casadi
git checkout 3.5.5
mkdir build
cd build
cmake -DWITH_PYTHON=ON -DWITH_PYTHON3=ON ..
make
sudo make install
cd ../..
sudo rm -r ./casadi/

# Scipy installation
conda install scipy 

# Tensorflow and JAX installation/upgrade
conda install -c apple tensorflow-deps
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal
pip install --upgrade "jax[cpu]" 

# Resolving matplotlib vs. numpy issue
pip uninstall matplotlib
pip install --no-cache-dir "matplotlib==3.4.3"

# hj_reachability needs to be installed after jax (old link in comments for authentification error)
# pip install --upgrade git+https://dti-devops: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@github.com/MariusWiggert/hj_reachability_c3.git
pip install git+https://github.com//MariusWiggert/hj_reachability_c3.git

# Install the ocean navigation package
pip install -e .

echo 'Environment installed'