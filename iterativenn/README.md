# iterativenn

## TL;DR

Create a python virtual environment using

```
python -m venv venv
. venv/bin/activate
```

Install the code an the required libraries in this environment

```
pip install -e ".[testing,notebook,distributed,gym,huggingface]"
```

Then do

```
cd notebooks/START_HERE
more README.md
```

to get started.

## How to setup the code
This project looks at iterative neural networks, and now the align with standard neural networks.

The best way to use is the code is to install a local Python virtual environment with:

```
python -m venv venv
. venv/bin/activate 
```

and the install the code in that environment using

```
pip install -e . 
```

or 

```
pip install -e ".[testing]"
```

to also allow unit testing.  *Perhaps even better*, there are more complete install
invocations below

## install everything
This is a good starting place for working on the code.  This will install everything you need, and likely
some things that you don't.

```
pip install -e ".[testing,notebook,distributed,gym,huggingface]"
```

## install everything with a CUDA which is new enough to work on A100s on Turing
This could actually be the default I think.  I mean, very old GPUs may not work, but you can use the above
for them.

```
pip install --extra-index-url https://download.pytorch.org/whl/cu116 -e ".[testing,notebook,distributed,gym,huggingface]"
```

## The full thing with sparse stuff
Note, the sparse code is currently a little flaky, so there is no need to do this until you want to live on the 
bleeding edge.

```
pip install --extra-index-url https://download.pytorch.org/whl/cu116 -e ".[testing,notebook,distributed,gym,huggingface,sparse]"
```

# Organization
Here is where the code lives:
## notebooks
This is the directory you are likely most interested in.  It contains Jupyter notebooks that
are examples of iterative neural networks.  These are a great place to start additional development.
In particular, the directory 

notebooks/START_HERE 

contains Jupter notebooks that are intended to help you get started.

## scripts
This is were utility scripts.  

## src
This is where shared code lives.  

## tests
This is where unit tests live.  You run the tests by doing:

pytest

in the main directory.

# Testing

To test to main code files just do:

```
pytest
```

To test the notebooks do:

```
pytest --nbmake --ignore-glob=**/.ipynb_checkpoints notebooks/
pytest --nbmake --ignore-glob=**/.ipynb_checkpoints --ignore-glob=**/develop notebooks/
```

Or, in fish shell

```
pytest --nbmake --ignore-glob=\*\*/.ipynb_checkpoints --ignore-glob=\*\*/develop notebooks/
```
