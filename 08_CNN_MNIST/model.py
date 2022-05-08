import torch.nn as nn

#Build a Convolutional Neural Network

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.Conv = nn.Conv2d(in_channels =1, out_channels = 10, kernel_size = 3, stride = 1, padding = 1)
        self.ReLU = nn.ReLU()
        self.MaxPool = nn.MaxPool2d(2)
        self.Linear = nn.Linear(in_features = 1960 , out_features= 10)
    
    def forward(self, x):
        output = self.Conv(x) #input(batch_size ,1, 28, 28), output(batch_size, 10, 28, 28)
        output = self.ReLU(output)
        output = self.MaxPool(output) #input(batch_size, 10, 28, 28), output(batch_size, 10, 14, 14)
        #flatten input(batch_size, 10, 14, 14), output(batch_size, 1960)
        output = output.view(-1, 1960)
        #Linear input(bath_size, 1960) output(batch_size, 10)
        output = self.Linear(output)
        #we don't have to do softmax here, because we are use CrossEntropy, which softmax is included
        return output