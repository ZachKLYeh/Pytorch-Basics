import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dataset
import model

#Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#declare parameters
input_size = 784  #image size [28, 28], flatten to [784]
hidden_size = 1000
n_classes = 10
n_epoch = 2
batch_size = 100
learning_rate = 0.001

#define a tensor transform
class TensorTransform:
    def __call__(self, input):
        features, labels = input
        #features must be float type for foward pass(gradient calc), label tensor must be int(one-hot encoded)
        # if we use from_numpy, it thus inheret the dtype of desinated numpy, in this case, integer
        return torch.tensor(features, dtype = torch.float32), torch.tensor(labels, dtype = int)

#Make dataset with transform included
train_dataset = dataset.TrainSet(transform = TensorTransform())
test_dataset = dataset.TestSet(transform = TensorTransform())

#Make dataloader
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

#Make model with parameters
model = model.FNN(input_size = input_size, hidden_size = hidden_size, n_classes = n_classes).to(device)

#Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

#Make Training Loop
steps = len(train_loader)
print('Start training...')

for epoch in range(n_epoch):
    for i, (images, labels) in enumerate(train_loader):
        #flatten our image to fit model input, and move it to GPU
        #[100, 1, 28, 28] to [100, 784]
        #reshape(dimention, desinated size)
        images = images.reshape(-1, 28*28).to(device)
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
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        output = model(images)
        #the output is a probability distribution. We want to tranform them into integer(predicted state)
        #torch.max(tensor, dimention) will return [max tensor value, index] in a dimention of a tensor
        #here we only want the index, not the linear output(float), actually it is the predicted number
        #Note: _ here is not probability because we didn't apply softmax
        _, predicted = torch.max(output, 1)
        n_samples = n_samples + labels.shape[0]
        #Note:(predicted == labels) is still a tensor with one element. We need to use item() to get a value
        #then we can compute divition
        n_correct = n_correct + (predicted == labels).sum().item()

acc = n_correct / n_samples
print(f'test accuracy: {acc:.3f}')