#! /bin/bash

rm -rf venv
# This is know to work with
# https://repo.continuum.io/miniconda/Miniconda3-py39_23.3.1-0-Linux-x86_64.sh
$HOME/minimamba/bin/python3 -m venv venv
# /usr/bin/python3 -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install -e ".[testing,notebook,distributed,gym,huggingface]"
# sparse stuff
# Note, this uses a custom repository which is not supported in setup.cfg
# See https://stackoverflow.com/questions/57689387/equivalent-for-find-links-in-setup-py
# TL;DR it should be the users responsitbility to install from strange places.
CUDA=cu117
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+${CUDA}.html
