import hydra
from iterativenn.RunnerUtils import runner_main

def test_RunnerUtil():
    # From: https://hydra.cc/docs/advanced/unit_testing/
    with hydra.initialize(version_base=None, config_path="../scripts/conf"):
        # config is relative to a module
        cfg = hydra.compose(config_name="config.yaml")
        runner_main(cfg)

def test_RunnerUtil_config():
    datas = ['tiny.yaml']
    models = ['LSTM.yaml', 'MLP.yaml', 'description_tiny.yaml']
    # From: https://hydra.cc/docs/advanced/unit_testing/
    with hydra.initialize(version_base=None, config_path="../scripts/conf"):
        # config is relative to a module
        for data in datas:
            for model in models:
                cfg = hydra.compose('config', overrides=[f"model={model}", f"data={data}"])
                runner_main(cfg)
