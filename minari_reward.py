import sys
import minari

available = list(minari.list_local_datasets().keys())
known = {'hammer-expert-v1': 12388.34541238637, 
        'door-expert-v1': 2923.658201933201, 
        'kitchen-mixed-v1': 342.76650563607086, 
        'kitchen-partial-v1': 290.86312399355876, 
        'kitchen-complete-v1': 377.8421052631579}

if len(sys.argv) ==2: name = sys.argv[1]
else:
    print('Available Datasets: ', available)
    name = input('Dataset: ')

if known.get(name, False): print(known[name])
elif name not in available: print('Please choose from the available options')
else: 
    dataset = minari.load_dataset(name)
    total_rewards = 0
    for episode in dataset: total_rewards += episode.rewards.sum()
    dataset_average_reward = total_rewards / len(dataset)
    print(dataset_average_reward)