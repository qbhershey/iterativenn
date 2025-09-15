#!/usr/bin/env python

# "Weights and Biases" is a really nice logging library/web pag.
# See here for details:  https://wandb.ai/
# If you want to use this then you need a wandb account, which is free!  Just go to https://wandb.ai/signup to get started.

# Some basic libraries for processing yaml, printing, etc.
import hydra
from iterativenn.RunnerUtils import runner_main

@hydra.main(version_base=None, config_name="config.yaml", config_path="conf")
def main(cfg):
    runner_main(cfg)

if __name__ == '__main__':
    main()
