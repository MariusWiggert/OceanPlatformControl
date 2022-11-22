#!/bin/bash

VERSION="16"

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

    # install jax
    # cuda version can also run on cpu (errors can be ignored)
    gpu=$(lspci | grep -ci NVIDIA)
    if ((gpu > 0)); then
        pip install "jax[cuda]==0.3.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    else
        pip install "jax[cpu]==0.3.23" -f https://storage.googleapis.com/jax-releases/jax_releases.html
    fi

    pip install chex==0.1.5

    # install newest tensorflow after jax since it has to replace dependencies (old flatbuffers)
    pip install tensorflow==2.9.1

    # install Ray Lib with Dashboard and RL-lib
    # pip install -U ray[default,rllib]==1.13.0
    # pip install -U "ray[default,rllib] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp39-cp39-macosx_10_15_x86_64.whl"
    # pip install -U "ray[default,rllib] @ https://s3-us-west-2.amazonaws.com/ray-wheels/master/75e9722a4d9c0d8d5cb0c37eb6316553f9e0789e/ray-3.0.0.dev0-cp39-cp39-macosx_10_15_x86_64.whl"
    pip install -U "ray[default,rllib] @ https://s3-us-west-2.amazonaws.com/ray-wheels/master/7d6b43b77047b25e79a062246d1bc07cef3e09be/ray-3.0.0.dev0-cp39-cp39-manylinux2014_x86_64.whl"
    #pip install -U "ray[default,rllib] @ https://s3-us-west-2.amazonaws.com/ray-wheels/master/75e9722a4d9c0d8d5cb0c37eb6316553f9e0789e/ray-3.0.0.dev0-cp39-cp39-manylinux2014_x86_64.whl"

    # install c3 type for python
    pip install git+https://github.com/c3aidti/c3python

    # install private hj_reachability library fork via token
    pip install --upgrade git+https://dti-devops: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@github.com/MariusWiggert/hj_reachability_c3.git@09c4ad23cb0e0088d923cf2ef5a2d77722f29aa9

    # install other python requirements via pip
    pip install -r ~/OceanPlatformControl/setup/cluster-jerome/requirements.txt

    # install requirements needed by ray head node
    pip install -U azure-cli-core azure-identity azure-mgmt-compute azure-mgmt-network azure-mgmt-resource msrestazure

    # activate conda environment
    (conda activate ocean_platform &> /dev/null && echo 'conda activate ocean_platform' >> ~/.bashrc) || true

    touch "env_$VERSION"
fi