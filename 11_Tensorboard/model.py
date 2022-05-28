import torch.nn as nn
import torchvision

#make our own VGG with torchvision.models.vgg16
#But we modify the fully connected layers
class VGG16(nn.Module):
    def __init__(self, num_classes=10):  
        super(VGG16, self).__init__()
        net = torchvision.models.vgg16(pretrained = True) 
        net.classifier = nn.Sequential() 
        self.features = net 
        self.classifier = nn.Sequential(  
            nn.Linear(512, 512),  
            nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

#Build a Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.Conv = nn.Conv2d(in_channels =3, out_channels = 10, kernel_size = 3, stride = 1, padding = 1)
        self.ReLU = nn.ReLU()
        self.MaxPool = nn.MaxPool2d(2)
        self.Linear = nn.Linear(in_features = 2560 , out_features= 10)
    
    def forward(self, x):
        output = self.Conv(x) #input(batch_size ,3, 32, 32), output(batch_size, 10, 32, 32)
        output = self.ReLU(output)
        output = self.MaxPool(output) #input(batch_size, 10, 32, 32), output(batch_size, 10, 16, 16)
        #flatten input(batch_size, 10, 14, 14), output(batch_size, 2560)
        output = output.view(-1, 2560)
        #Linear input(bath_size, 2560) output(batch_size, 10)
        output = self.Linear(output)
        #we don't have to do softmax here, because we are use CrossEntropy, which softmax is included
        return output