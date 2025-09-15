import sys
import hydra
import wandb
import torch

sys.path.insert(0, './iterativenn/src/')

from iterativenn.utils.model_factory import ModelFactory
from Data.AdroitDoor.DataFactoryAD import score

project = 'adroit_door'
run_id = 'nb1bo0t2'
#run_id = '0o332z5v'
test_project = 'adroit_door_test'
test_runs = 10

wandb.restore(name='final.pt', run_path='qhershey/'+project+'/'+run_id, replace=True)
wandb.restore(name='config.yaml', run_path='qhershey/'+project+'/'+run_id, replace=True)

with hydra.initialize(version_base=None, config_path=''):
  cfg = hydra.compose(config_name="config.yaml")
  cfg['logger']['value']['project'] = test_project
  cfg['model']['value']['callbacks']['output']['func'] = 'all'

model = ModelFactory(cfg['model']['value'])
model.model = torch.load('final.pt')
average_model_reward, test_runs = score(model, test_runs)

print('Dataset Average Reward', 2923.658201933201)
print("Model Average Reward ", average_model_reward)