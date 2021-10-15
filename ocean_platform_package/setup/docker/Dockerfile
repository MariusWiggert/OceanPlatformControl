FROM ubuntu:18.04

SHELL [ "/bin/bash", "--login", "-c" ]

# copy over some local files
COPY requirements.txt /tmp/
COPY entrypoint.sh /usr/local/bin/

# System packages 
RUN apt-get update && apt-get install -y git curl

# Install miniconda to $CONDA_DIR
ENV CONDA_DIR $HOME/miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p $CONDA_DIR -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

# make non-activate conda commands available
ENV PATH=$CONDA_DIR/bin:$PATH
# for later reference in entrypoint
ENV ENV_PREFIX $PWD/py3_parcels

# Python packages from conda
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
RUN conda init bash
RUN conda activate root
RUN conda create -n py3_parcels -c conda-forge python=3.6 parcels jupyter cartopy ffmpeg

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "py3_parcels", "/bin/bash", "-c"]
RUN conda remove --force parcels
RUN pip install git+https://github.com/MariusWiggert/parcels.git@master

# create a project directory inside user home
ENV PROJECT_DIR $HOME/ocean_platform
RUN mkdir $PROJECT_DIR

# install some pip things
RUN pip install -r /tmp/requirements.txt
# RUN pip install torch

WORKDIR $PROJECT_DIR

# ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]
# ENTRYPOINT ["conda", "run", "-n", "py3_parcels", "jupyter", "lab", "--no-browser", "--allow-root", "--ip", "0.0.0.0" ]

# default command will launch JupyterLab server for development
CMD ["conda", "run", "-n", "py3_parcels", "jupyter", "lab", "--no-browser", "--allow-root", "--ip", "0.0.0.0" ]