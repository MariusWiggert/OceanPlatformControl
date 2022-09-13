#!/bin/bash

VERSION="5"

if [ -f "env_$VERSION" ]; then
    echo -e "\033[0;32mOcean Platform Environment Version $VERSION already installed\033[0m"
else
    echo -e "\033[0;33mInstalling Ocean Platform Environment Version $VERSION\033[0m"

    # restore original .bashrc and add conda hooks
    #/bin/cp /etc/skel/.bashrc ~/
    #(which conda && echo 'eval "$(conda shell.bash hook)"' >> ~/.bashrc) || true
    conda init bash > /dev/null
    source ~/.bashrc

    # deactivate all conda environments and create conda environment 'ocean_platform'
    for i in $(seq ${CONDA_SHLVL}); do
        conda deactivate
    done
    conda create -y -n ocean_platform python=3.9.11
    conda activate ocean_platform

    # these packages have to be installed manually (otherwise pip fails to install dependencies)
    conda install -y -c conda-forge
    conda install -y -c jupyter cartopy ffmpeg

    # update pip
    pip install --upgrade pip

    # install outdated jax==0.24 for hj_reachability
    # cuda version can also run on cpu (errors can be ignored)
    #pip install --upgrade "jax[cpu]==0.2.24" -f https://storage.googleapis.com/jax-releases/jax_releases.html
    pip install --upgrade "jax[cuda]==0.2.24" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    # install newest tensorflow after jax since it has to replace dependencies (old flatbuffers)
    pip install tensorflow==2.9.1

    # install Ray Lib with Dashboard and RL-lib
    pip install -U ray[default,rllib]==1.13.0
    #pip install -U "ray[default,rllib] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp39-cp39-manylinux2014_x86_64.whl"
    # Nightly Install should remove some GRPC errors not captured in 1.13 (June 22)
    #pip install -U "ray[default,rllib] @ https://s3-us-west-2.amazonaws.com/ray-wheels/master/68b5d4302c51a3ead2ffbbb972ed65fb3eb18dc7/ray-3.0.0.dev0-cp39-cp39-manylinux2014_x86_64.whl"

    # install c3 type for python
    pip install git+https://github.com/c3aidti/c3python

    # install private hj_reachability library fork via token
    pip install --upgrade git+https://dti-devops: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@github.com/MariusWiggert/hj_reachability_c3.git

    # install other python requirements via pip
    pip install -r requirements.txt

    # install requirements needed by ray head node
    pip install -U azure-cli-core azure-identity azure-mgmt-compute azure-mgmt-network azure-mgmt-resource msrestazure

    # activate conda environment
    (conda activate ocean_platform &> /dev/null && echo 'conda activate ocean_platform' >> ~/.bashrc) || true

    touch "env_$VERSION"
fi