import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json

# import the yaml processing library
import yaml

from iterativenn.nn_modules.MaskedLinear import MaskedLinear

class MultiplicativeFish(nn.Module):
    def __init__(self, W1, b1, W2, b2):
        super().__init__()
        self.W1 = W1
        self.b1 = torch.nn.Parameter(b1)    
        self.W2 = W2
        self.b2 = torch.nn.Parameter(b2)

    def forward(self, x):
        return (self.W1(x) + self.b1) * (self.W2(x) + self.b2)

    @staticmethod
    def get_identity(z_size):
        # Create a MultiplicativeFish that is the identity function
        W1 = MaskedLinear.from_description([z_size], [z_size], block_types=[['D']], initialization_types=[[1]], trainable=[[True]], bias=False)
        b1 = torch.zeros(z_size, requires_grad=True)
        W2 = MaskedLinear.from_description([z_size], [z_size], block_types=[['D']], initialization_types=[[0]], trainable=[[True]], bias=False) 
        b2 = torch.ones(z_size, requires_grad=True)
        return MultiplicativeFish(W1, b1, W2, b2)    
    
class DiscontinuousFish(nn.Module):
    def __init__(self, W1, b1, W2, b2):
        super().__init__()
        self.W1 = W1
        self.b1 = torch.nn.Parameter(b1)    
        self.W2 = W2
        self.b2 = torch.nn.Parameter(b2)

    def forward(self, x):
        return torch.where(x < 0, self.W1(x) + self.b1, self.W2(x) + self.b2)
    
    @staticmethod
    def get_identity(z_size):
        # Create a DiscontinuousFish that is the identity function
        W1 = MaskedLinear.from_description([z_size], [z_size], block_types=[['D']], initialization_types=[[1]], trainable=[[True]], bias=False)
        b1 = torch.zeros(z_size, requires_grad=True)
        W2 = MaskedLinear.from_description([z_size], [z_size], block_types=[['D']], initialization_types=[[1]], trainable=[[True]], bias=False) 
        b2 = torch.zeros(z_size, requires_grad=True)
        return DiscontinuousFish(W1, b1, W2, b2)   

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def load_data(name):
    # Read the json info file
    with open(f'4-data/info.json') as f:
        info = json.load(f)
    # Read the start data
    x_start = pd.read_parquet(f'4-data/{name}_start.parquet')
    # Read the target data
    x_target = pd.read_parquet(f'4-data/{name}_target.parquet')
    return x_start, x_target, info

def load_config():
    with open('manifold_batch.yml', 'r') as file:
        return yaml.safe_load(file)

def test():
    X = torch.rand(1000, z_size)

    # These should be the identity map, this confirms that the implementation is correct
    with torch.no_grad():
        map = MultiplicativeFish.get_identity(z_size)
        Y = map(X)  
        assert torch.isclose(X, Y).all(), "MultiplicativeFish is not the identity map"

    with torch.no_grad():
        map = DiscontinuousFish.get_identity(z_size)
        Y = map(X)  
        assert torch.isclose(X, Y).all(), "DiscontinuousFish is not the identity map"

    # A slighlyt harder test.  The composition of two identity maps should be the identity map
    with torch.no_grad():
        map1 = MultiplicativeFish.get_identity(z_size)
        map2 = DiscontinuousFish.get_identity(z_size)
        map = torch.nn.Sequential(map1, map2)
        Y = map(X)  
        assert torch.isclose(X, Y).all(), "MultiplicativeFish \circ DiscontinuousFish is not the identity map"

# Turn a pandas dataframe into a pytorch tensor
def df_to_tensor(df):
    return torch.tensor(df.values, dtype=torch.float32)

# a dataloader which returns a batch of start and target data
class Data(torch.utils.data.Dataset):
    def __init__(self, x_start, x_target):
        self.x_start = x_start
        self.x_target = x_target
    def __len__(self):
        return len(self.x_start)
    def __getitem__(self, idx):
        return self.x_start[idx], self.x_target[idx]

if __name__ == '__main__':
    device = get_device()
    config = load_config()
    x_start, x_target, info = load_data(config['name'])

    padding_size = config['padding_size']
    test_size = config['test_size']    
    x_size = x_start.shape[1]
    z_size = x_size+padding_size

    # Make two pytorch tensor datasets from the start and target data
    x_start_tensor = df_to_tensor(x_start)
    x_target_tensor = df_to_tensor(x_target)

    # Pad the tensors with zeros
    x_start_tensor = F.pad(x_start_tensor, (0, padding_size))
    x_target_tensor = F.pad(x_target_tensor, (0, padding_size))

    x_start_tensor_test = x_start_tensor[:test_size]
    x_target_tensor_test = x_target_tensor[:test_size]

    x_start_tensor = x_start_tensor[test_size:]
    x_target_tensor = x_target_tensor[test_size:]

    # map = DiscontinuousFish.get_identity(z_size)
    map = MultiplicativeFish.get_identity(z_size)

    # map1 = MultiplicativeFish.get_identity(z_size)
    # map2 = DiscontinuousFish.get_identity(z_size)
    # map = torch.nn.Sequential(map1, map2)

    map.to(device)
    
    train_data = Data(x_start_tensor, x_target_tensor)
    test_data = Data(x_start_tensor_test, x_target_tensor_test)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=512, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=512, shuffle=False)


    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(map.parameters(), lr=0.0002)

    max_epochs = config['max_epochs']
    max_iterations = config['max_iterations']
    losses = torch.zeros(max_epochs, max_iterations)
    # Train the model
    for epoch in range(max_epochs):
        for batch_idx, (start, target) in enumerate(train_loader):
            optimizer.zero_grad()
            start = start.to('cuda')
            target = target.to('cuda')

            total_loss = 0
            mapped = start
            for i in range(max_iterations):
                mapped = map(mapped)
                tmp_loss = criterion(mapped[:, :x_size], target[:, :x_size])
                losses[epoch, i] = tmp_loss.item()
                total_loss += tmp_loss
                # if name == 'MNIST':
                #     # A version of oracle conditioning for MNIST
                #     # The first 28*28 dimensions are the image which we know and keep fixed.
                #     mapped = torch.concatenate([target[:, :x_size-1], mapped[:, x_size-1:]], dim=1) 

            total_loss.backward()

            optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Total Loss {total_loss.item():.2e}')
            for i in range(max_iterations):
                print(f'Iter {i}: {losses[epoch, i]:.2e}', end=', ')
            print()  

# Add weights and biases stuff


# for i in range(max_iterations):
#     plt.semilogy(losses[:, i].detach().numpy(), label=f'Iteration {i}')
#     plt.legend()    


# # # Model visualization

# # In[13]:


# I = torch.eye(z_size).to('cuda')
# W1 = map.W1(I).detach().cpu().numpy()
# plt.imshow(W1, cmap='plasma')
# plt.colorbar()


# # In[14]:


# # Plot just the parts of the image where the valye is less than zero
# plt.imshow(np.where(W1 < 0, W1, 0), cmap='plasma')
# plt.colorbar()


# # In[15]:


# np.where(W1 < -0.5)


# # In[16]:


# b1 = map.b1.detach().cpu().numpy()
# plt.plot(b1)


# # In[17]:


# W2 = map.W2(I).detach().cpu().numpy()
# plt.imshow(W2, cmap='plasma')
# plt.colorbar()


# # In[18]:


# b2 = map.b2.detach().cpu().numpy()
# plt.plot(b2)


# # In[19]:


# plt.plot(W1[x_size-1, :x_size-1],label='W1_end')
# plt.plot(W1[x_size, :x_size-1],label='W1')
# plt.legend()


# # In[20]:


# plt.plot(W2[x_size-1, :x_size-1], label='W1_end')
# plt.plot(W2[x_size-2, :x_size-2], label='W1')
# plt.legend()


# # # Training data

# # In[21]:


# start_batch, target_batch = next(iter(train_loader))
# start_batch = start_batch.to('cuda')
# target_batch = target_batch.to('cuda')

# mapped_batch = start_batch
# for i in range(max_iterations):
#     mapped_batch = map(mapped_batch)

# plot_idx = 0

# start_batch = start_batch.to('cpu')
# mapped_batch = mapped_batch.to('cpu').detach()
# target_batch = target_batch.to('cpu')

# plt.plot(start_batch[plot_idx, :x_size],label='start')
# plt.plot(mapped_batch[plot_idx, :x_size],label='mapped')
# plt.plot(target_batch[plot_idx, :x_size],label='target')
# plt.legend()


# # In[22]:


# plt.plot(target_batch[plot_idx, :x_size]-start_batch[plot_idx, :x_size], label='target-start')
# plt.plot(target_batch[plot_idx, :x_size]-mapped_batch[plot_idx, :x_size], label='target-mapped')
# plt.legend()


# # # Problem specific visualization

# # In[23]:


# if name == "MNIST":
#     plt.imshow(start_batch[plot_idx, :(x_size-1)].reshape(28, 28))
#     plt.imshow(mapped_batch[plot_idx, :(x_size-1)].reshape(28, 28))
#     plt.imshow(target_batch[plot_idx, :(x_size-1)].reshape(28, 28))

#     print(start_batch[plot_idx, x_size-1])
#     print(mapped_batch[plot_idx, x_size-1])
#     print(target_batch[plot_idx, x_size-1])

#     for i in range(start_batch.shape[0]):
#         print(f'{start_batch[i, x_size-1]} {mapped_batch[i, x_size-1]} {target_batch[i, x_size-1]}')


# # In[24]:


# if name == "MNIST":
#     y_hat = start_batch[:, x_size-1]
#     y = target_batch[:, x_size-1]
#     plt.scatter(y_hat, y)


# # In[25]:


# if name == "MNIST":
#     y_hat = mapped_batch[:, x_size-1]
#     y = target_batch[:, x_size-1]
#     plt.scatter(y_hat, y)


# # In[26]:


# if name == 'circle':
#     plt.scatter(target_batch[:, 0], target_batch[:, 1], label='target')
#     plt.scatter(mapped_batch[:, 0], mapped_batch[:, 1], label='mapped')
#     plt.scatter(start_batch[:, 0], start_batch[:, 1], label='start')


# # In[27]:


# if name == 'LunarLander':
#     plot_idx = 3
#     plt.plot(start_batch[plot_idx, 0:101], start_batch[plot_idx, 101:202], 'o-', label='start')
#     plt.plot(mapped_batch[plot_idx, 0:101], mapped_batch[plot_idx, 101:202], 'o-', label='mapped')
#     plt.plot(target_batch[plot_idx, 0:101], target_batch[plot_idx, 101:202], 'o-', label='target')
#     plt.legend()



# # # Testing

# # In[28]:


# test_start_batch, test_target_batch = next(iter(test_loader))
# test_start_batch = test_start_batch.to('cuda')
# test_target_batch = test_target_batch.to('cuda')

# test_mapped_batch = test_start_batch
# for i in range(4):
#     test_mapped_batch = map(test_mapped_batch)

# test_start_batch = test_start_batch.to('cpu')
# test_mapped_batch = test_mapped_batch.to('cpu').detach()
# test_target_batch = test_target_batch.to('cpu')


# # In[29]:


# if name == "MNIST":
#     plt.imshow(test_start_batch[plot_idx, :(x_size-1)].reshape(28, 28))
#     plt.imshow(test_mapped_batch[plot_idx, :(x_size-1)].reshape(28, 28))
#     plt.imshow(test_target_batch[plot_idx, :(x_size-1)].reshape(28, 28))

#     print(test_start_batch[plot_idx, x_size-1])
#     print(test_mapped_batch[plot_idx, x_size-1])
#     print(test_target_batch[plot_idx, x_size-1])

#     for i in range(test_start_batch.shape[0]):
#         print(f'{test_start_batch[i, x_size-1]} {test_mapped_batch[i, x_size-1]} {test_target_batch[i, x_size-1]}')


# # In[30]:


# if name == "MNIST":
#     y_hat = test_start_batch[:, x_size-1]
#     y = test_target_batch[:, x_size-1]
#     plt.scatter(y_hat, y)


# # In[31]:


# if name == "MNIST":
#     y_hat = test_mapped_batch[:, x_size-1]
#     y = test_target_batch[:, x_size-1]
#     plt.scatter(y_hat, y)


# # In[ ]:




