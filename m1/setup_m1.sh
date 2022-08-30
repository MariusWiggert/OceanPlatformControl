#!/usr/bin/env zsh
conda init zsh
# create a conda environment called ocean_platform
conda env create -f requirements_m1.yml

conda activate ocean_platform

pip install git+https://dti-devops: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@github.com/MariusWiggert/hj_reachability_c3.git@4549bea7c3cda3b4d3bf90735bc1cba4703fb4ca
pip install git+https://github.com/scikit-learn/scikit-learn.git@e358bd77e0cde248e0ee8f67c29a72e330fcc0fe


# brew install hdf5 netcdf
# git clone https://github.com/Unidata/netcdf4-python.git
# HDF5_DIR=$(brew --prefix hdf5) pip install ./netcdf4-python

conda install -c conda-forge netCDF4

brew install swig

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
sudo rm -r ./netcdf4-python/

echo 'Environment installed'