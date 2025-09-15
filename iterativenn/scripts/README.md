# Notes

This one just puts things online on wandb.ai

./Runner_new.py logger=wandb

multirun

./Runner_new.py -m

## Growing example

./Runner_new.py logger=wandb model=sequential2D

This will create a file called wandb_path.json in the root directory. This file contains the path to the wandb directory.
This is used to know which model to download.

./Runner_new.py logger=wandb model=wandb_sequential2D


