import torch
import model
import dataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

#model path
MODEL_FILE = 'CNN.pth'
MODEL_STATE_FILE = 'CNN_state_dict.pth'

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Make test dataset with transform included
composed_transform = transforms.Compose([transforms.ToTensor(),
                     transforms.Normalize(mean = (0.5, 0.5, 0.5 ), std = (0.5, 0.5, 0.5))])
test_dataset = dataset.TestSet(transform = composed_transform)
test_loader = DataLoader(dataset = test_dataset, batch_size = 16, shuffle = False)

#there are two ways to declare model
#the second is recommended

#(1)load the whole model
model = torch.load("CNN.pth")
#(2)load model state dict
'''
model = model.CNN().to(device)
model.load_state_dict(torch.load("CNN_state_dict.pth"))
'''

#model.eval() will turn off dropout and batchnorms for evaluation
model.eval()

#Test the model
#Note: In test case, we do not want to calculate the gradients
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    #you can also use for i, (images, labels) in enumerate(test_loader):
    #but now we don't care batch imformation, simple use this
    for images, labels in test_loader:
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
