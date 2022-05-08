import idx2numpy
import numpy as np
from torch.utils.data import Dataset

#unzip dataset to test_images, test_labels, train_images, train_labels as numpy

file = r'/home/zacharyyeh/Datasets/MNIST/t10k-images.idx3-ubyte'
test_images = idx2numpy.convert_from_file(file)
file = r'/home/zacharyyeh/Datasets/MNIST/t10k-labels.idx1-ubyte'
test_labels = idx2numpy.convert_from_file(file)
file = r'/home/zacharyyeh/Datasets/MNIST/train-images.idx3-ubyte'
train_images = idx2numpy.convert_from_file(file)
file = r'/home/zacharyyeh/Datasets/MNIST/train-labels.idx1-ubyte'
train_labels = idx2numpy.convert_from_file(file)

#Build children class TrainSet, TestSet from parent class Dataset 

class TrainSet(Dataset):
    def __init__(self, transform = None):
        self.train_images = train_images
        self.train_labels = train_labels
        self.n_samples = train_images.shape[0]
        #Important!!!! add a channel to images [28, 28] to [1, 28, 28] so that i can be seen as a image
        self.train_images = np.resize(self.train_images, (self.n_samples, 1, 28,28))
        #Transpouse to fit ToTensor input form!!!! [samples c h w] to [samples h w c]
        self.train_images = np.transpose(self.train_images, (0, 2, 3, 1))
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_images[index], self.train_labels[index]
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return self.n_samples

class TestSet(Dataset):
    def __init__(self, transform = None):
        self.test_images = test_images
        self.test_labels = test_labels
        self.n_samples = test_images.shape[0]
        self.test_images = np.resize(self.test_images, (self.n_samples, 1, 28,28))
        self.test_images = np.transpose(self.test_images, (0, 2, 3, 1))
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.test_images[index], self.test_labels[index]
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return self.n_samples
