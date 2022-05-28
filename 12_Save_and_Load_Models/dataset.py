import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import pickle

#unpicke dataset to batches
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#unpickled data are dict use b'labels' , b'data to access them
batch1 = unpickle(r'/home/zacharyyeh/Datasets/CIFAR10/data_batch_1')
batch2 = unpickle(r'/home/zacharyyeh/Datasets/CIFAR10/data_batch_2')
batch3 = unpickle(r'/home/zacharyyeh/Datasets/CIFAR10/data_batch_3')
batch4 = unpickle(r'/home/zacharyyeh/Datasets/CIFAR10/data_batch_4')
batch5 = unpickle(r'/home/zacharyyeh/Datasets/CIFAR10/data_batch_5')
batch_test = unpickle(r'/home/zacharyyeh/Datasets/CIFAR10/test_batch')
train_images = np.concatenate((batch1[b'data'], batch2[b'data'], batch3[b'data'], batch4[b'data'], batch5[b'data']))
train_labels = np.concatenate((batch1[b'labels'], batch2[b'labels'], batch3[b'labels'], 
                                batch4[b'labels'], batch5[b'labels']))
test_images = batch_test[b'data']
test_labels = batch_test[b'labels']

#Build children class TrainSet, TestSet from parent class Dataset 

class TrainSet(Dataset):
    def __init__(self, transform = None):
        self.train_images = train_images #[samples, flatten]
        self.train_labels = train_labels
        self.n_samples = train_images.shape[0]
        self.train_images = np.vstack(self.train_images).reshape(-1, 3, 32, 32)
        self.train_images = np.transpose(self.train_images, (0, 2, 3, 1))
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_images[index], self.train_labels[index]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
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
        self.test_images = np.vstack(self.test_images).reshape(-1, 3, 32, 32)
        self.test_images = np.transpose(self.test_images, (0, 2, 3, 1))
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.test_images[index], self.test_labels[index]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return self.n_samples