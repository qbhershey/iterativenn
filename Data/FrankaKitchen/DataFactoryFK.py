import minari
import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl


class Iter_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        bigx = item.observations['observation']
        for g in ['achieved_goal', 'desired_goal']:
            for t in ['bottom burner', 'kettle', 'light switch', 'microwave']:
                bigx = np.concatenate((bigx, item.observations[g][t]), axis=1)
        return {'x': torch.tensor(np.array(bigx[:-1, :])), 'y': torch.tensor(np.array(item.actions))}


class WrapperDataModule(pl.LightningDataModule):
    def __init__(self, dl_train, dl_val, dl_test):
        super().__init__()
        self.dl_train = dl_train
        self.dl_val = dl_val
        self.dl_test = dl_test

    def train_dataloader(self):
        return self.dl_train

    def val_dataloader(self):
        return self.dl_val

    def test_dataloader(self):
        return self.dl_test

def trivial_collate_fn(batch):
    return batch


def FrankaKitchen():
    dataset = minari.load_dataset('kitchen-mixed-v1')

    trainset, valset, testset = random_split(dataset, [501, 60, 60])

    dl_train = DataLoader(Iter_Dataset(trainset), batch_size=64, shuffle=True, collate_fn=trivial_collate_fn)
    dl_val = DataLoader(Iter_Dataset(valset), batch_size=64, shuffle=False, collate_fn=trivial_collate_fn)
    dl_test = DataLoader(Iter_Dataset(testset), batch_size=64, shuffle=False, collate_fn=trivial_collate_fn)

    return WrapperDataModule(dl_train, dl_val, dl_test)


def score(model, test_runs=10):
    model.eval()
    with torch.no_grad():
        goals = ['achieved_goal', 'desired_goal']
        tasks = ['bottom burner', 'kettle', 'light switch', 'microwave']
        dset = minari.load_dataset('kitchen-mixed-v1') 
        environments = []
        observations = []
        rewards = np.zeros(test_runs)
        dones = [False]*test_runs

        for t in range(test_runs):
          env = dset.recover_environment()
          obs, _ = env.reset()
          bigx = torch.unsqueeze(torch.FloatTensor(obs['observation']), 0)
          for goal in goals:
            for task in tasks:
                smallx = torch.unsqueeze(torch.FloatTensor(obs[goal][task]), 0)
                bigx = torch.cat((bigx, smallx), 1)
          yhats = [torch.nan]*len(bigx)
          observation = [{'x': bigx, 'y': yhats}]
          environments.append(env)
          observations += observation

        while not all(dones):
            actions = model(observations)
            for t in range(test_runs):
                obs, rew, ter, tru, _ = environments[t].step(actions[t].squeeze().numpy())
                bigx = torch.unsqueeze(torch.FloatTensor(obs['observation']), 0)
                for goal in goals:
                    for task in tasks:
                        smallx = torch.unsqueeze(torch.FloatTensor(obs[goal][task]), 0)
                        bigx = torch.cat((bigx, smallx), 1)
                bigx = torch.cat([observations[t]['x'], bigx])
                yhats = [torch.nan]*len(bigx)
                observations[t] = {'x': bigx, 'y': yhats}
                rewards[t] += rew
                dones[t] = ter or tru

        average_model_reward = rewards.mean()
        return average_model_reward, test_runs