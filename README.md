# OceanPlatformControl Project

## Setup of environment
In any case install conda:
Install Anaconda OR miniconda (only one of them) [Link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html)

### If you only develop on OceanPlatformControl
```sh
# 1. Clone project repo: 
git clone https://github.com/MariusWiggert/OceanPlatformControl.git
# 2. In the terminal, navigate inside the project repo and run the install bash script
cd setup/
source  set_up_ocean_platform_env.sh
```

### Minimal setup
Recreate the minimal setup of Marius with hj_reachability (v0.4).
```sh
cd setup/
conda create -y -n ocean_minimal python=3.9.11
conda activate ocean_minimal
conda install -y -c jupyter cartopy ffmpeg
pip install --upgrade pip
pip install git+https://github.com/c3aidti/c3python
pip install -r requirements_minimal.txt
# hj_reachability needs to be installed after jax
pip install --upgrade git+https://dti-devops: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@github.com/MariusWiggert/hj_reachability_c3.git
```
### If you develop on OceanPlatformControl and hj_reachability simulatenously

3. Now you need to download our two repos and put them in the same top_folder
* top_folder
    * OceanPlatformControl
    * hj_reachability_c3
2. Clone project repo: 
`git clone https://github.com/MariusWiggert/OceanPlatformControl.git`
3. Clone hj_reachability repo:
`git clone https://github.com/MariusWiggert/hj_reachability_c3`
4. In the terminal, navigate inside the project repo, open the script:
`set_up_ocean_platform_env.sh`
Comment out line 13 (where hj_reachability is installed)
Now run the script `source set_up_ocean_platform_env.sh`


This will create a new conda environment called `ocean_platform` which contains all that is needed.










