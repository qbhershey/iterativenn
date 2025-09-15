from iterativenn.utils.DatasetUtils import ImageSequence,RandomImageSequence,AdditionImageSequence
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import random_split


def test_ImageSequence():
    # data
    dataset = MNIST('~/work/data', train=True, download=True, 
                    transform=transforms.ToTensor())
    mnist_train,_ , _ = random_split(dataset, [100, 100, 59800])

    # Turn the images into image sequences
    mnist_train = ImageSequence(mnist_train, min_copies=2, max_copies=4)

    for item in mnist_train:
        assert len(item['x']) >= 2 and len(item['x']) <= 4, "ImageSequence x should be a sequence of length 2 to 4"
        assert len(item['y']) >= 2 and len(item['y']) <= 4, "ImageSequence y should be a sequence of length 2 to 4"

def test_RandomImageSequence():
    # data
    dataset = MNIST('~/work/data', train=True, download=True, 
                    transform=transforms.ToTensor())
    mnist_train,_ , _ = random_split(dataset, [100, 100, 59800])

    # Turn the images into image sequences
    mnist_train = RandomImageSequence(mnist_train, min_copies=2, max_copies=4)

    for item in mnist_train:
        assert len(item['x']) >= 2 and len(item['x']) <= 4, "ImageSequence x should be a sequence of length 2 to 4"
        assert len(item['y']) >= 2 and len(item['y']) <= 4, "ImageSequence y should be a sequence of length 2 to 4"

def test_AdditionImageSequence():
    # data
    dataset = MNIST('~/work/data', train=True, download=True, 
                    transform=transforms.ToTensor())
    train_size=32
    mnist_train,_ , _ = random_split(dataset, [train_size, 100, 60000-(train_size+100)])

    # Turn the images into image sequences
    mnist_train = AdditionImageSequence(mnist_train, copies=3)

    for item in mnist_train:
        assert len(item['x']) == 9 and len(item['x']) == 9, "ImageSequence x should be a sequence of length 9"
        assert len(item['y']) == 9 and len(item['y']) == 9, "ImageSequence y should be a sequence of length 9"
