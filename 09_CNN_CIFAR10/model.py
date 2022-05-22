import torch.nn as nn
import torchvision

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

    def featuremap(self, x):
        return self.Conv(x)

#Build a vgg16 model
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv6 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv7 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv8 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv9 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv10 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv11 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv12 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv13 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(in_features = 512, out_features = 512)
        self.linear2 = nn.Linear(in_features = 512, out_features = 512)
        self.linear3 = nn.Linear(in_features = 512, out_features = 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        #(3, 32, 32)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.maxpool(out)
        #(64, 16, 16)
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = self.maxpool(out)
        #(128, 8, 8)
        out = self.relu(self.conv5(out))
        out = self.relu(self.conv6(out))
        out = self.relu(self.conv7(out))
        out = self.maxpool(out)
        #(256, 4, 4)
        out = self.relu(self.conv8(out))
        out = self.relu(self.conv9(out))
        out = self.relu(self.conv10(out))
        out = self.maxpool(out)
        #(512, 2, 2)
        out = self.relu(self.conv11(out))
        out = self.relu(self.conv12(out))
        out = self.relu(self.conv13(out))
        out = self.maxpool(out)
        #(512, 1, 1)
        out = out.view(-1, 512)
        #(512)
        out = self.relu(self.linear1(out))
        #out = self.dropout(out)
        #(64)
        out = self.relu(self.linear2(out))
        #out = self.dropout(out)
        #(64)
        out = self.linear3(out)
        #(10)
        return out



#Build a vgg16 sequential model
class VGG16BN(nn.Module):
    def __init__(self):
        super(VGG16BN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, (3,3), 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3), 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, (3,3), 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3,3), 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, (3,3), 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3,3), 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3,3), 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, (3,3), 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3,3), 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3,3), 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, (3,3), 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3,3), 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3,3), 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = out.view(-1, 512)
        out = self.classifier(out)
        return out

#slightly modify torchvition vggnet for compairson
class VGGNet(nn.Module):
    def __init__(self, num_classes=10):  # num_classes
        super(VGGNet, self).__init__()
        net = torchvision.models.vgg16(pretrained=False) 
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