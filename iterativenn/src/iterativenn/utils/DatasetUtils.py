import random
from collections import defaultdict

import warnings

# The Gym stuff is in flux, so we need to suppress the deprecation warnings
# from https://quantumcomputing.stackexchange.com/questions/13181/
# suppress-deprecation-warnings-from-qiskit
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    import gymnasium as gym
    from stable_baselines3 import PPO

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class GymManager:
    """
    Manages input and output dimensions of the Gym envioronment.
    """
    def __init__(self, env):
        self.env = env

    def get_space_dims(self):
        if type(self.env.action_space)==gym.spaces.box.Box:
            output_space = self.env.action_space.shape[0]
        elif type(self.env.action_space) ==gym.spaces.discrete.Discrete:
            output_space = self.env.action_space.n
        else:
            raise NotImplementedError

        if type(self.env.observation_space)==gym.spaces.box.Box:
            input_space = self.env.observation_space.shape[0]
        elif type(self.env.observation_space) ==gym.spaces.discrete.Discrete:
            input_space = self.env.observation_space.n
        else:
            raise NotImplementedError
        return input_space, output_space


class GymImageSequence(Dataset):
    """
    Gym Environments: Acrobot-v1, LunarLander-v2, BipedalWalker-v3, CartPole-v1
    Action space:
    """
    def __init__(self, n_data_points=1000, min_copies=1, max_copies=4,
                 env_str="LunarLander-v2", model_str="MlpPolicy", episode_seq=False):
        self.actions = []
        self.observations = []
        self.env = gym.make(env_str)
        self.model = PPO(model_str, self.env, verbose=1, device='cpu')
        self.input_space, self.output_space = tuple(GymManager(self.env).get_space_dims())
        self.model.learn(total_timesteps=5000)
        self.min_copies = min_copies
        self.max_copies = max_copies
        self.get_model_action(n_data_points)
        self.obs_action_seq = []
        self.episode_seq = episode_seq
        self._create_input_seq()


    def get_model_action(self, n_data_points):
        obs, info = self.env.reset()
        for i in range(n_data_points):
            self.observations.append(obs)
            action, _states = self.model.predict(obs, deterministic=True)
            self.actions.append(action)
            obs, reward, done, truncated, info = self.env.step(action)
            # self.env.render() # not needed
            if done or truncated:
                obs, info = self.env.reset()


    def __len__(self):
        if self.obs_action_seq:
            return len(self.obs_action_seq)
        else:
            return len(self.actions)


    def _create_input_seq(self):
        for i in range(self.__len__() - self.max_copies):
            action_seq = self.actions[i:i + self.max_copies] # * 5
            obs_seq = self.observations[i:i + self.max_copies]
            self.obs_action_seq.append((obs_seq, action_seq))


    def __getitem__(self, idx):
        if self.episode_seq:
            obs_seq, action_seq = self.obs_action_seq[idx]
        else:
            obs_seq = []
            action_seq = []
            for i in range(self.max_copies):
                action_seq += [self.actions[idx]]
                obs_seq += [self.observations[idx]]

        return {'x': torch.tensor(np.array(obs_seq)), 'y': torch.tensor(np.array(action_seq))}


class CustomTensorDataset(Dataset):
    def __init__(self, data_x: torch.Tensor, data_y: torch.Tensor):
        self.data_x = data_x
        self.data_y = data_y

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]


class MemoryCopySequence(Dataset):
    def __init__(self, min_sequence_size=1, max_sequence_size=1, dimension=1, copies=1, size=1000):
        """Generate data for a simple memory problem
        for example, if sequence_size=3, dimension=2 and copies=2, then
        the data will be of the form:
        x = [[ 1, 2],
             [ 1, 2],
             [ 3, 4],
             [ 3, 4],
             [ 5, 6],
             [ 5, 6],
             [-1, -1],
             [-1, -1],
             [-1, -1]] 
        y = [[NaN, NaN],
             [NaN, NaN],
             [NaN, NaN],
             [NaN, NaN],
             [NaN, NaN],
             [NaN, NaN],
             [ 1, 2],
             [ 3, 4],
             [ 5, 6]] 
        
        Args:
            sequence_size (int, optional): The number of entries in the sequence. Defaults to 1.
            copies (int, optional): The number of times to show each entry. Defaults to 1.
        """
        self.min_sequence_size = min_sequence_size
        self.max_sequence_size = max_sequence_size
        self.dimension = dimension
        self.copies = copies
        self.size = size
        self.data = []
        for i in range(self.size):
            sequence_size = torch.randint(self.min_sequence_size, self.max_sequence_size+1, (1,)).item()
            data = torch.randint(1, 10, (sequence_size, self.dimension))
            
            x = []
            for i in range(sequence_size):
                for j in range(self.copies):
                    x.append(data[i])
            for i in range(sequence_size):
                x.append(torch.ones(self.dimension) * -1)

            y = []
            for i in range(sequence_size*self.copies):
                y.append(torch.ones(self.dimension) * torch.nan)
            for i in range(sequence_size):
                y.append(data[i])

            self.data.append({'x': torch.stack(x), 'y': torch.stack(y)})

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

class ImageSequence(Dataset):
    def __init__(self, images, min_copies=1, max_copies=1, 
                 transform=None, target_transform=None, evaluate_loss='all'):
        self.images = images
        self.min_copies = min_copies
        self.max_copies = max_copies
        self.transform = transform
        self.target_transform = target_transform
        self.evaluate_loss = evaluate_loss

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_sequence = []
        label_sequence = []

        image = self.images[idx][0]
        if self.transform:
            image = self.transform(image)

        label = self.images[idx][1]
        if self.target_transform:
            label = self.target_transform(label)

        for i in range(random.randint(self.min_copies, self.max_copies)):
            if self.evaluate_loss == 'all':
                label_sequence += [label]
            elif self.evaluate_loss == 'last':
                label_sequence += [torch.nan]
            else:
                raise ValueError('evaluate_loss must be "all" or "last"')
            image_sequence += [image]
        if self.evaluate_loss == 'last':
            label_sequence[-1] = label
        # It is easy to get into indexing hell, so we'll just make
        # these dictionaries to impose a little structure on the data.
        return {'x': image_sequence, 'y': label_sequence}
    
class RandomImageSequence(Dataset):
    def __init__(self, images, min_copies=1, max_copies=1, 
                 transform=None, target_transform=None, evaluate_loss='all'):
        self.images = images
        self.min_copies = min_copies
        self.max_copies = max_copies
        self.transform = transform
        self.target_transform = target_transform
        self.evaluate_loss = evaluate_loss

        self.label_map = defaultdict(lambda: [])
        for idx,sample in enumerate(images):
            self.label_map[sample[1]] += [idx]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_sequence = []
        label_sequence = []

        base_label = self.images[idx][1]
        for i in range(random.randint(self.min_copies, self.max_copies)):
            idx_same_label = random.choice(self.label_map[base_label])

            image = self.images[idx_same_label][0]
            if self.transform:
                image = self.transform(image)
            image_sequence += [image]

            if self.evaluate_loss == 'all':
                label = base_label
                if self.target_transform:
                    label = self.target_transform(base_label)
                label_sequence += [label]
            elif self.evaluate_loss == 'last':
                label_sequence += [torch.nan]
            else:
                raise ValueError('evaluate_loss must be "all" or "last"')
        if self.evaluate_loss == 'last':
            label = base_label
            if self.target_transform:
                label = self.target_transform(base_label)
            label_sequence[-1] = [label]
        # It is easy to get into indexing hell, so we'll just make
        # these dictionaries to impose a little structure on the data.
        return {'x': image_sequence, 'y': label_sequence}

class AdditionImageSequence(Dataset):
    def __init__(self, images, copies=1, transform=None):
        self.images = images
        self.copies = copies
        self.transform = transform

        self.label_map = defaultdict(lambda: [])
        for idx,sample in enumerate(images):
            self.label_map[sample[1]] += [idx]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_sequence = []
        label_sequence = []

        images = []
        labels = []
        # The two images we want to add modulo 10
        # The first is just the normal image in the sequence
        # NOTE: The following line is strange. It is required for this
        # to be iterable.  I.e., without it we can just keep returning
        # new examples.  I believe that error this raises when idx is out
        # of range is used to stop the iteration.
        base_image, base_label = self.images[idx]
        labels += [base_label]
        images += [base_image]

        # The second is randomly chosen
        labels += [random.randint(0, 9)]
        idx_label = random.choice(self.label_map[labels[1]])
        images += [self.images[idx_label][0]]

        # The final image is blank to indicate we want to compute the sum 
        # of the previous labels modulo 10
        labels += [(labels[0]+labels[1])%10]
        images += [images[0]*0.0]

        # The first set of images have label_1
        for image, label in zip(images, labels):
            for i in range(self.copies):
                if self.transform:
                    image = self.transform(image)
                image_sequence += [image]
                label_sequence += [label]
        # It is easy to get into indexing hell, so we'll just make
        # these dictionaries to impose a little structure on the data.
        return {'x': image_sequence, 'y': label_sequence}

# Add a noise transform
# This is from https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745/2
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_transform(name):
    # Here we allow ourselves to have a fine grained control over 
    # the data we are using.  In particular, we want to control the difficultly 
    # of the problem. These transforms give us the ability to do this.
    if name == 'baseline':
        # From https://github.com/pytorch/examples/blob/master/mnist/main.py
        transform = transforms.Compose([
            # This image starts as a PILImage and resize works on that
            # This is just to make them all the same
            transforms.Resize((50, 50)),
            # However, the normalization, and actual learning on the image
            # need a tensor.
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif name == 'baseline_small':
        # From https://github.com/pytorch/examples/blob/master/mnist/main.py
        transform = transforms.Compose([
            # This image starts as a PILImage and resize works on that
            # This is just to make them all the same
            transforms.Resize((10, 10)),
            # However, the normalization, and actual learning on the image
            # need a tensor.
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif name == 'perspective_small':
        # A somewhat harder problem with random perspective, though only a small amount of transform.
        transform = transforms.Compose([
            # Theses happen to the PILImage
            transforms.Resize((50, 50)),
            transforms.RandomPerspective(p=1.0, fill=128, distortion_scale=0.1),
            # However, the normalization, and actual learning on the image
            # need a tensor.
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    elif name == 'perspective':
        # A somewhat harder problem with random perspective
        transform = transforms.Compose([
            # Theses happen to the PILImage
            transforms.Resize((50, 50)),
            transforms.RandomPerspective(p=1.0, fill=128, distortion_scale=0.5),
            # However, the normalization, and actual learning on the image
            # need a tensor.
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    elif name == 'erase':
        # A somewhat harder problem with random perspective
        transform = transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.ToTensor(),
            # The RandomErasing needs a tensor.
            transforms.RandomErasing(p=1.0),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif name == 'noise':
        # A somewhat harder problem with random perspective
        transform = transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.ToTensor(),
            # Add noise to the images
            AddGaussianNoise(std=2.0),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif name == 'both':
        # A somewhat harder problem with random perspective
        transform = transforms.Compose([
            # This is a strange one.  We need to do a bit of back and forth.
            # We start as a PILImage and resize works on that
            transforms.Resize((50, 50)),
            # We then briefly go to a tensor for the RandomErasing
            transforms.ToTensor(),
            transforms.RandomErasing(p=1.0, scale=(0.02, 0.05)),
            # The we go back to a PILImage for the RandomPerspective
            transforms.ToPILImage(),
            transforms.RandomPerspective(p=1.0, fill=128, distortion_scale=0.5),
            # Then we go back to a tensor for the normalization and processing
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    else:
        raise ValueError('Unknown transform %s' % name)
    return transform


