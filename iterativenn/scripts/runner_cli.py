#! /usr/bin/env python3

import click
import hydra
import omegaconf
import subprocess

from dask.distributed import Client
import multiprocessing
from joblib import Parallel, delayed

import pprint
import logging
logging.basicConfig()
# The call to basicConfig() should come before any calls to debug(), info() etc. As it’s intended 
# as a one-off simple configuration facility, only the first call will actually do anything: 
# subsequent calls are effectively no-ops.
logger = logging.getLogger(__name__)

from iterativenn.RunnerUtils import runner_main, runner_parallel_init
from iterativenn.utils import logger_factory

###############################################
#  This sets up the aliases for the commands
###############################################
class AliasedGroup(click.Group):
    def get_command(self, ctx, cmd_name):
        rv = click.Group.get_command(self, ctx, cmd_name)
        if rv is not None:
            return rv
        matches = [x for x in self.list_commands(ctx)
                   if x.startswith(cmd_name)]
        if not matches:
            return None
        elif len(matches) == 1:
            return click.Group.get_command(self, ctx, matches[0])
        ctx.fail('Too many matches: %s' % ', '.join(sorted(matches)))
###############################################

@click.version_option()
@click.option('-q', '--quiet', default=False, is_flag=True,
              help='Suppress all output except errors. (Default: False)')
@click.option('--debug', default=False, is_flag=True,
              help='Enable debug logging. (Default: False)')
@click.option('--backend', default='None',
              help='The parallel processing backend to use. (Default: None)')
@click.option('--workers', default=1,
              help='The number of parallel workers to use. (Default: 1)')
@click.option('--base-dir', default='/home/rcpaffenroth/iterativenn_logs/runner',
              help='The base directory to use for all logging and files. (Default: /home/rcpaffenroth/iterativenn_logs/runner)')
@click.option('--clean', default=False, is_flag=True,
              help='Clean the run directory. (Default: False)')              
@click.option('-d', '--dry-run', default=False, is_flag=True,
              help='Do a dry run, printing out commands but not running any code. (Default: False)')              
@click.option('--cuda', default=False, is_flag=True,
              help='Use CUDA for the run. (Default: False)')              
@click.group(cls=AliasedGroup)
@click.pass_context
def cli(ctx, quiet, debug, backend, workers, base_dir, clean, dry_run, cuda):
    """This is the main convenience script for doing runs.  Common use cases include:

    \b
    # One model, console logger
    runner_cli.py fast 
    # One model, 20 epochs, wandb logger, project runner_small
    runner_cli.py small
    # Several models, 20 epochs, tensorboard logger
    runner_cli.py medium
    # Several models, 50 epochs, wandb logger, 3 seeds
    runner_cli.py large
    # Several models, several datasets, 2 epochs, wandb logger, 1 seeds, 8 workers    
    runner_cli.py --workers 8 --backend joblib all --fast
    # This is the full run for actual results
    # Several models, several datasets, 100 epochs, wandb logger, 3 seeds, 8 workers    
    runner_cli.py --workers 8 --backend joblib all
    # The arxiv paper run
    runner_cli.py --workers 2 --backend joblib arxiv --fast
    """
    ctx.ensure_object(dict)
    # Everything else is for the Hydra compose    
    ctx.obj['overrides'] = [
        'base=long', 
        'base.profiler=null',
        f'base.base_dir={base_dir}',
    ]
    if cuda:
        ctx.obj['overrides'].append('base.use_cuda=True')
    else:
        ctx.obj['overrides'].append('base.use_cuda=False')

    ctx.obj['extra_overrides'] = [[]]
    if clean and click.confirm(f'Are you sure you want to remove {base_dir}/* ?'):
        subprocess.run(f'rm -rf {base_dir}/*', shell=True, check=True)

    # This bears some explanation.  We want to be able to control the
    # the logging level even when the logger is running in a separate
    # process.  For example, this happens when the logger is run
    # through Dask.
    if quiet:
        logging.getLogger().setLevel(logging.ERROR)
        ctx.obj['overrides'] += ['logger.console_level=error']
    elif debug:
        logging.getLogger().setLevel(logging.DEBUG)
        ctx.obj['overrides'] += ['logger.console_level=debug']
    else:
        logging.getLogger().setLevel(logging.INFO)
        ctx.obj['overrides'] += ['logger.console_level=info']
    
    # Arguments that are used by the runner itself
    ctx.obj['backend'] = backend
    ctx.obj['workers'] = workers
    ctx.obj['base_dir'] = base_dir
    ctx.obj['dry_run'] = dry_run

###################################################################
@cli.command()
@click.argument('overrides', nargs=-1)
@click.pass_context
def fast(ctx, overrides):
    ctx.obj['models'] = ['sequential2D_sparse']
    ctx.obj['datasets'] = ['tiny']
    ctx.obj['seeds'] = [1]

    ctx.obj['overrides'] += [
        f'base.max_epochs=2',
        f'logger=console',
    ]
    ctx.obj['overrides'] += overrides
    run(ctx.obj)
###################################################################
@cli.command()
@click.argument('overrides', nargs=-1)
@click.pass_context
def small(ctx, overrides):
    ctx.obj['models'] = ['sequential2D_sparse']
    ctx.obj['datasets'] = ['small']
    ctx.obj['seeds'] = [1]
    ctx.obj['overrides'] += [
        f'base.max_epochs=20',
        # f'logger=tensorboard',
        f'logger=wandb',
        f'logger.project=runner_small',
    ]
    ctx.obj['overrides'] += overrides
    run(ctx.obj)
###################################################################
@cli.command()
@click.argument('overrides', nargs=-1)
@click.pass_context
def medium(ctx, overrides):
    ctx.obj['models'] = [
        'RNN', 'GRU', 'LSTM', 
        'sequential2D_dense', 'sequential2D_sparse',
        'sequential2D_MLP',
    ]
    ctx.obj['datasets'] = ['small']
    ctx.obj['seeds'] = [1]
    ctx.obj['overrides'] += [
        'base.max_epochs=20',
        'logger=tensorboard',
    ]
    ctx.obj['overrides'] += overrides
    run(ctx.obj)
###################################################################
@cli.command()
@click.argument('overrides', nargs=-1)
@click.pass_context
def allvlast(ctx, overrides):
    ctx.obj['models'] = [
        'sequential2D_sparse'
    ]
    ctx.obj['datasets'] = ['small']
    ctx.obj['seeds'] = [1]
    ctx.obj['overrides'] += [
        'base.max_epochs=5',
        'logger=wandb',
        'logger.project=allvlast',
    ]
    ctx.obj['overrides'] += overrides
    ctx.obj['extra_overrides'] = [
        ['+data.sequence_type.evaluate_loss="all"'],
        ['+data.sequence_type.evaluate_loss="last"'],
    ]
    run(ctx.obj)
###################################################################
@cli.command()
@click.argument('overrides', nargs=-1)
@click.pass_context
def large(ctx, overrides):
    ctx.obj['models'] = [
        'RNN', 'GRU', 'LSTM', 
        'sequential2D_dense', 'sequential2D_sparse',
        'sequential2D_MLP', 
    ]
    ctx.obj['datasets'] = ['small']
    ctx.obj['seeds'] = [1, 2 ,3]
    ctx.obj['overrides'] += [
        'base.max_epochs=50',
        'logger=tensorboard',
    ]
    ctx.obj['overrides'] += overrides
    run(ctx.obj)
###################################################################
@cli.command()
@click.argument('overrides', nargs=-1)
@click.pass_context
def largeb(ctx, overrides):
    ctx.obj['models'] = [
        'RNN', 'GRU', 'LSTM', 
        'sequential2D_dense', 'sequential2D_sparse',
        'sequential2D_MLP', 
    ]
    ctx.obj['datasets'] = ['tiny']
    ctx.obj['seeds'] = [1, 2, 3]
    ctx.obj['overrides'] += [
        'base.max_epochs=50',
        'logger=wandb',
        'logger.project=temp2',
    ]
    ctx.obj['overrides'] += overrides
    run(ctx.obj)
###################################################################
@cli.command()
@click.argument('overrides', nargs=-1)
@click.option('--fast', default=False, is_flag=True,
              help='Fast run. (Default: False)')
@click.pass_context
def arxiv(ctx, overrides, fast):
    ctx.obj['overrides'] += ['logger=wandb']
    if fast:
        ctx.obj['models'] = [
            'sequential2D_sparse',
        ]
        ctx.obj['datasets'] = ['medium']
        ctx.obj['seeds'] = [1]
        ctx.obj['overrides'] += [
            'base.max_epochs=2',
            'logger.project=arxiv-2-epoch-6-7-2023',
        ]
    else:
        ctx.obj['models'] = [
            'RNN',  'sequential2D_dense', 'sequential2D_sparse','sequential2D_MLP',
        ]
        ctx.obj['datasets'] = ['medium', 'both_uniform', 'both_random']
        ctx.obj['seeds'] = [1, 2, 3, 4, 5]
        ctx.obj['overrides'] += [
            'base.max_epochs=20',
            'logger.project=arxiv-20-epoch-6-7-2023',
        ]
    ctx.obj['overrides'] += overrides
    ctx.obj['extra_overrides'] = [
        ['data.evaluate_loss="all"'],
        ['data.evaluate_loss="last"'],
    ]
    run(ctx.obj)
###################################################################
@cli.command()
@click.argument('overrides', nargs=-1)
@click.option('--fast', default=False, is_flag=True,
              help='Fast run. (Default: False)')
@click.pass_context
def all(ctx, overrides, fast):
    ctx.obj['models'] = [
        'RNN', 'GRU', 'LSTM', 
        'sequential2D_dense', 'sequential2D_sparse',
        'sequential2D_MLP',
    ]
    ctx.obj['datasets'] = ['medium', 'both_uniform', 'both_random', 'baseline_addition']
    if fast:
        ctx.obj['seeds'] = [1]
        ctx.obj['overrides'] += [
            'base.max_epochs=2',
            'logger=wandb',
            'logger.project=all-2-epoch-2-25-2023',
        ]
    else:
        ctx.obj['seeds'] = [1, 2, 3]
        ctx.obj['overrides'] += [
            'base.max_epochs=100',
            'logger=wandb',
            'logger.project=all-100-epoch-5-15-2023',
        ]
    ctx.obj['overrides'] += overrides
    run(ctx.obj)
###################################################################
@cli.command()
@click.argument('name', nargs=1)
@click.argument('overrides', nargs=-1)
@click.pass_context
def gym(ctx, name, overrides):
    ctx.obj['models'] = [
        f'{name}/description', 
        f'{name}/LSTM', 
        f'{name}/MLP', 
    ]
    ctx.obj['datasets'] = [f'{name}']
    ctx.obj['seeds'] = [1]
    ctx.obj['overrides'] += [
        'base.max_epochs=0',
        'logger=console',
        # 'logger.project=temp4',
    ]
    ctx.obj['overrides'] += overrides
    run(ctx.obj)
###################################################################
@cli.command()
@click.argument('overrides', nargs=-1)
@click.option('--fast', default=False, is_flag=True,
              help='Fast run. (Default: False)')
@click.pass_context
def gymall(ctx, overrides, fast):
    runs = []
    for name in ['lunar', 'cartpole', 'acrobot']:
        ctx.obj['models'] = [
            f'{name}/description', 
            f'{name}/LSTM', 
            f'{name}/MLP', 
        ]
        ctx.obj['datasets'] = [f'{name}']
        if fast:
            ctx.obj['seeds'] = [1]
            ctx.obj['overrides'] += [
                'base.max_epochs=2',
                'logger=wandb',
                'logger.project=gym-2-25-2023',
            ]
        else:
            ctx.obj['seeds'] = [1,2,3]
            ctx.obj['overrides'] += [
                'base.max_epochs=200',
                'logger=wandb',
                'logger.project=gym-2-25-2023',
            ]

        ctx.obj['overrides'] += overrides
        runs += generate_runs(ctx.obj)
    do_runs(ctx.obj, runs)
###################################################################

def run(runner_ctx):
    runs = generate_runs(runner_ctx)
    if runner_ctx['dry_run']:
        for i,run in enumerate(runs):
            print(f'Run {i+1}/{len(runs)}')
            print('=====================')
            print(omegaconf.OmegaConf.to_yaml(run))
    else:
        do_runs(runner_ctx, runs)

def generate_runs(runner_ctx):
    runs = []
    with hydra.initialize(version_base=None, config_path="conf"):
        futures = []
        for seed in runner_ctx['seeds']:
            for data in runner_ctx['datasets']:
                for model in runner_ctx['models']:
                    for extra_overrides in runner_ctx['extra_overrides']:
                        all_overrides = runner_ctx['overrides'] + extra_overrides + [
                            f'model={model}',        
                            f'data={data}',        
                            f'base.seed={seed}',
                            f'logger.name={model}_{data}_{seed}',
                        ]
                        cfg = hydra.compose(config_name="config", overrides=all_overrides)
                        runs += [cfg]
    return runs

def do_runs(runner_ctx, runs):
    # FIXME: This is based on the parameters of the
    # first run. This is not ideal since it might
    # not be the same as the other runs.
    runner_parallel_init(runs[0])
    if runner_ctx['backend'] == 'None':
        logger.info(f'There are {len(runs)} experiments')
        for to_run_idx in range(len(runs)):
            logger.debug(pprint.pformat(runs[to_run_idx]))
            runner_main_wrapper(runs[to_run_idx])
    elif runner_ctx['backend'] == 'dask':
        dask_run(runs, runner_ctx['workers'])
    elif runner_ctx['backend'] == 'multiprocessing':
        multiprocessing_run(runs, runner_ctx['workers'])
    elif runner_ctx['backend'] == 'joblib':
        joblib_run(runs, runner_ctx['workers'])
    elif runner_ctx['backend'] == 'count':      
        logger.info(f'There are {len(runs)} experiments')
    elif runner_ctx['backend'][:5] == 'index':
        to_run_idx = int(runner_ctx['backend'][6:])        
        runner_main_wrapper(runs[to_run_idx])
    else:
        raise NotImplementedError(f'Unknown backend {runner_ctx["backend"]}')

def dask_run(runs, workers):
    # dask_client = Client('tcp://127.0.0.1:8786')
    dask_client = Client(n_workers=workers, 
                         threads_per_worker=1,
                         processes=True)
    futures = []
    for cfg in runs:
        futures.append(dask_client.submit(runner_main_wrapper, cfg))
    for i,f in enumerate(futures):
        print('waiting for future', i)
        f.result()
        print('future is done', i)

def multiprocessing_run(runs, workers):
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(workers)
    # from https://docs.wandb.ai/guides/track/log/distributed-training
    # also see https://docs.wandb.ai/guides/integrations/hydra
    # where it is mentioned that 
    # If your process hangs when started, this may be caused by this 
    # known issue. To solve this, try to changing wandb's multiprocessing 
    # protocol either by adding an extra settings parameter 
    # to `wandb.init` as:
    # wandb.init(settings=wandb.Settings(start_method="thread"))
    futures = []
    for cfg in runs:
        futures.append(pool.apply_async(runner_main_wrapper, (cfg,)))

    for i,f in enumerate(futures):
        print('waiting for future', i)
        f.get()
        print('future is done', i)

def joblib_run(runs, workers):
    Parallel(n_jobs=workers)(delayed(runner_main_wrapper)(runs[i]) for i in range(len(runs)))

def runner_main_wrapper(cfg):
    # This is a wrapper to make sure we can run this with dask.  This
    # will run within the remote processor, so we can control properties
    # of the remote process here.
    runner_main(cfg)

# Needs to be wrapper in __main__ to work with dask
if __name__ == '__main__':
    cli(obj={})

