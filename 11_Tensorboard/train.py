import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import model
import dataset
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn

from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter('runs/CIFAR10-simple-CNN')
writer = SummaryWriter('runs/CIFAR10-VGG-Pretrained')

#Set device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

#declare parameters
input_size = 32*32 #image size = 32x32 --flatten 1024
n_classes = 10
n_epoch = 10
batch_size = 100
learning_rate = 0.001

#declare classes for confution matrix
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#Make dataset with transform included
#mean = [0.485, 0.456, 0.406]
#std = [0.229, 0.224, 0.225]
mean = (0.5, 0.5, 0.5 )
std = (0.5, 0.5, 0.5)

composed_transform = transforms.Compose([transforms.ToTensor(),
                     transforms.Normalize(mean = mean, std = std)])

train_dataset = dataset.TrainSet(transform = composed_transform)
test_dataset = dataset.TestSet(transform = composed_transform)

#Make dataloader
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = 16, shuffle = False)

#Show some examples to examinate dataloader
examples = iter(test_loader)
example_images, example_labels = examples.next()
#denormalize to visualize images
for i in range(16):
    example_images[i] = example_images[i]/2+0.5
#show the sampled images to tensorboard
#note: make grid images before upload to summarywriter
img_grid = torchvision.utils.make_grid(example_images)
writer.add_image('CIFAR10', img_grid)

#Load model
#model = model.CNN().to(device)
model = model.VGG16().to(device)   
#let's freeze the pretrained layers
model.features.requires_grad_(False)

#Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

#Make Training Loop
steps = len(train_loader)
print('Start training...')
#declare running parameters for tensorboard
runtime_loss = 0.0
runtime_correct = 0
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
        #record runtime information
        runtime_loss += loss.item()
        _, predicted = torch.max(pred_labels, 1)
        runtime_correct += (predicted == labels).sum().item() 
        #print information in a epoch
        if (i+1) % 100 == 0:
            print(f'epoch: {epoch+1}/{n_epoch}, step: {(i+1)}/{steps}, loss: {loss.item():.3f}')
            #calculate average percision and loss per step
            writer.add_scalar('training loss', runtime_loss / 100, epoch+1)
            writer.add_scalar('accuracy', runtime_correct / 100, epoch+1)
            #clear runtime information
            runtime_loss = 0.0
            runtime_correct = 0

print('Training is completed')

#Test the model
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    #save predictoins and tragets in linespace to make a confution matrix
    cf_pred = []
    cf_actual = []
    #parameters for tensorboard pr curve
    #pr_prob record the probability predicted
    #pr_pred record whether the prediciton is correct
    pr_prob = []
    pr_pred = []
    for images, labels in test_loader:
        #flatten
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        _, predicted = torch.max(output, 1)
        #pr curve require a prediction in probability between[0, 1]. So we must implement with softmax
        #softmax for a batch
        pr_prob_batch = [nn.functional.softmax(outputs, dim=0) for outputs in output]
        pr_pred.append(predicted)
        pr_prob.append(pr_prob_batch)
        #transform from tensor to numpy
        predicted = predicted.cpu().numpy()
        cf_pred.extend(predicted)
        labels = labels.cpu().numpy()
        cf_actual.extend(labels)

        n_samples = n_samples + labels.shape[0]
        #Note (predicted == labels) is still a tensor with one element. We need to use item() to get a value
        #then we can compute divition
        n_correct = n_correct + (predicted == labels).sum().item()
acc = n_correct / n_samples
print(f'model accuracy: {acc:.3f}')

#create pr curve for tensorboard
#use torch.cat to transfrom list to tensor
pr_prob = torch.cat([torch.stack(batch) for batch in pr_prob]) #[10000, 10]
pr_pred = torch.cat(pr_pred) #[10000]
prob_all = []
pred_all = []
for i in range(10):
        pred_i = pr_pred == i #transfrom a list of labels to a list of boolin
        prob_i = pr_prob[:, i] #given the class predicted probability
        pred_all.append(pred_i)
        prob_all.append(prob_i)
        writer.add_pr_curve(f'class {str(i)} PR curve', pred_i, prob_i, global_step=0)
writer.add_pr_curve('PR curve for all classes', torch.cat(pred_all), torch.cat(prob_all), global_step=0)

#make confution matrix
#confution_matrix form a 2d array from two seperate 1d array
cf_matrix = confusion_matrix(cf_actual, cf_pred)
#include dataframe showing classes of prediction
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix, axis = 1), index = [i for i in classes], columns = [i for i in classes])
cf_img = sn.heatmap(df_cm, annot = True).get_figure()
writer.add_figure('Confution Matrix', cf_img)

#See some predictions of the model
with torch.no_grad():
    examples = iter(test_loader)
    data, actual_labels = examples.next()
    data = data.to(device)
    predicted_labels = model(data)
    _, predicted_labels = torch.max(predicted_labels, 1)

print('predicted labels:\n', predicted_labels.cpu().numpy())
print('actual labels:\n', actual_labels.numpy().squeeze())