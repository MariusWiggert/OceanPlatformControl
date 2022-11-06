#!/usr/bin/env zsh
#conda init zsh


# conda create -y -n ocean_minimal python=3.9
# conda activate ocean_minimal


conda install -y -c jupyter cartopy ffmpeg
pip install --upgrade pip
pip install git+https://github.com/c3aidti/c3python
pip install -r setup/requirements_minimal_m1.txt

pip install -U numpy  

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

conda install scipy 

conda install -c -y apple tensorflow-deps
pip install -y tensorflow-macos
pip install -y tensorflow-metal


pip install --upgrade "jax[cpu]" 

# hj_reachability needs to be installed after jax
pip install --upgrade git+https://dti-devops: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@github.com/MariusWiggert/hj_reachability_c3.git


pip install -e .

echo 'Environment installed'