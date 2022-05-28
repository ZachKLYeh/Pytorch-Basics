import torch.nn as nn
import torchvision

#make our own VGG with torchvision.models.vgg16
#But we modify the fully connected layers
class VGGNet(nn.Module):
    def __init__(self, num_classes=10):  
        super(VGGNet, self).__init__()
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