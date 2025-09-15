# Getting started

Here are some notebooks to help you get started with the iterativenn codebase.

# 1-rcp-time-series.ipynb  
This is the place to get started and shows a comparision of RNNs, LSTMs, GRU, and various sorts of INNs.   The problem in 
a simple time series problem that focuses on memory.  The notebook is stand alone and does not require any additional
iterfaces (such as Weights and Biases)

# 1a-rcp-time-series-batch.py  
This is a Python file that addressess the exact same problem at the notebook above, but in a fancier and more batch processing paradigm.
It uses:

- "click" to make handle command line options
- "wandb" for cloud based plotting

# 1a-rcp-time-series-batch.sh  
This script calls the above python script for doing a Monte-carlo series of runs.

# 2-rcp-least-squares-growing.ipynb  
This is a more complicated example that shows how INNs can used for a different training paradigm called a "continuation method".  
At the moment the code is general, but the specific example it refers to is an electro-magnetics problem from Maria Barger, so you will need to contact 
her to get the data.

# 3-rcp-multi-growing.ipynb
This is an more complicated example that shows how INNs can used  "continuation methods" in multiple stages.  
Again, at the moment the code is general, but the specific example it refers to is an electro-magnetics problem from Maria Barger, so you will need to contact 
her to get the data.
