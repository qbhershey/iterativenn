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
        return {'x': torch.tensor(np.array(item.observations[:-1, :])), 'y': torch.tensor(np.array(item.actions))}


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


def AdroitDoor():
    dataset = minari.load_dataset('door-expert-v1')

    trainset, valset, testset = random_split(dataset, [4500, 250, 250])

    dl_train = DataLoader(Iter_Dataset(trainset), batch_size=64, shuffle=True, collate_fn=trivial_collate_fn)
    dl_val = DataLoader(Iter_Dataset(valset), batch_size=64, shuffle=False, collate_fn=trivial_collate_fn)
    dl_test = DataLoader(Iter_Dataset(testset), batch_size=64, shuffle=False, collate_fn=trivial_collate_fn)

    return WrapperDataModule(dl_train, dl_val, dl_test)


def score(model, test_runs=10):
    model.eval()
    with torch.no_grad():
        environments = []
        observations = []
        rewards = np.zeros(test_runs)
        dones = [False]*test_runs

        for t in range(test_runs):
          env = gym.make('AdroitHandDoor-v1', max_episode_steps=200)
          obs, _ = env.reset()
          obs = torch.unsqueeze(torch.FloatTensor(obs), 0)
          yhats = [torch.nan]*len(obs)
          observation = [{'x': obs, 'y': yhats}]
          environments.append(env)
          observations += observation

        while not all(dones):
            actions = model(observations)
            for t in range(test_runs):
                obs, rew, ter, tru, _ = environments[t].step(actions[t].squeeze().numpy())
                obs = torch.unsqueeze(torch.FloatTensor(obs), 0)
                obs = torch.cat([observations[t]['x'], obs])
                yhats = [torch.nan]*len(obs)
                observations[t] = {'x': obs, 'y': yhats}
                rewards[t] += rew
                dones[t] = ter or tru

        average_model_reward = rewards.mean()
        return average_model_reward, test_runs