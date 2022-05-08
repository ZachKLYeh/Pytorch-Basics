import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
import dataset
import model

#Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#declare parameters
input_size = 32*32 #image size = 32x32 --flatten 1024
n_classes = 10
n_epoch = 10
batch_size = 100
learning_rate = 0.001

#Make dataset with transform included
composed_transform = transforms.Compose([transforms.ToTensor(),
                     transforms.Normalize(mean = (0.5, 0.5, 0.5 ), std = (0.5, 0.5, 0.5))])

train_dataset = dataset.TrainSet(transform = composed_transform)
test_dataset = dataset.TestSet(transform = composed_transform)

#Make dataloader
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = 16, shuffle = False)

#Load model
model = model.CNN().to(device)   #accuracy around 60%
#model = model.VGG16().to(device)  #accuracy around 70%
#model = model.VGG16BN().to(device)  #include batchnorm, accuracy nearly 80%
#model = model.VGGNet(num_classes = 10).to(device)  #accuracy around 76%

#Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

#Make Training Loop
steps = len(train_loader)
print('Start training...')

for epoch in range(n_epoch):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        #Note: Labels are now dtype torch.uint8, should be change to torch.long for CrossEntropyLoss
        labels = labels.type(torch.long)
        labels = labels.to(device)
        #forward pass
        pred_labels = model(images)
        loss = criterion(pred_labels, labels)

        #backward pass
        loss.backward()

        #update gradients
        optimizer.step()
        optimizer.zero_grad()

        #print information in a epoch
        if (i+1) % 100 == 0:
            print(f'epoch: {epoch+1}/{n_epoch}, step: {(i+1)}/{steps}, loss: {loss.item():.3f}')

print('Training is completed')

#Test the model
#Note: In test case, we do not want to calculate the gradients
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    #you can also use for i, (images, labels) in enumerate(test_loader):
    #but now we don't care batch imformation, simple use this
    for images, labels in test_loader:
        #flatten
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        #torch.max(tensor, dimention) will return [max tensor value, index] in a dimention of a tensor
        _, predicted = torch.max(output, 1)
        n_samples = n_samples + labels.shape[0]
        #Note (predicted == labels) is still a tensor with one element. We need to use item() to get a value
        #then we can compute divition
        n_correct = n_correct + (predicted == labels).sum().item()

acc = n_correct / n_samples
print(f'test accuracy: {acc:.3f}')

#See some predictions of the model
with torch.no_grad():
    examples = iter(test_loader)
    data, actual_labels = examples.next()
    data = data.to(device)
    predicted_labels = model(data)
    _, predicted_labels = torch.max(predicted_labels, 1)

print('predicted labels:\n', predicted_labels.cpu().numpy())
print('actual labels:\n', actual_labels.numpy().squeeze())