import torch.nn as nn


#Build a Fully connected Neural Network

class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes):
        super(FNN, self).__init__()
        self.Linear1 = nn.Linear(input_size, hidden_size)
        self.ReLU = nn.ReLU()
        self.Linear2 = nn.Linear(hidden_size, n_classes)
        #we don't have to do softmax here, because we are use CrossEntropy, which softmax is included
    def forward(self, x):
        output = self.Linear1(x)
        output = self.ReLU(output)
        output = self.Linear2(output)
        return output

