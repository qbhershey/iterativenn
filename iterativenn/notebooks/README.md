* How to use this

Unfortunately, there are several steps to make this work, though this is better than all of my previous attempts at getting this to work.

** TL;DR

*** 01-rcp-data_gathering.ipynb
Download the data files that I use.

*** 02-rcp-block_autoencoder.ipynb
An example that shows an iterative NN and a normal NN are the same.  The gradients are
even identical.

*** 03-rcp-iterative_training.ipynb
An example of training an iterativeNN.  This is likely the most useful notebook
to start additional work from.

* Additional notes

** Setting up the runs
The file notebooks/02-rcp-updated_iterative.setup_dir.py sets up a bunch
of directories in data/to_run each containing a stand alone set of files along with a do.sh that does the actual run.

This is nice since it makes testing really easy.

** Paired Jupyter notebooks and Python files

Sometimes (or perhaps even often) it is convenient to have Jupyter notebooks and Python files that track each other.  I.e., they contain the same code, just in different formats. If this doesn't help you, then you can just pick one and ignore the other.  

However, to keep the .ipynb notebook and .py files synced you need to have jupytext installed.

Once it is installed you can link an .ipynb notebook and .py file by running:

jupytext --set-formats ipynb,py:percent notebook.ipynb  # Turn notebook.ipynb into a paired ipynb/py notebook
jupytext --sync notebook.ipynb                  # Update whichever of notebook.ipynb/notebook.py is outdated
