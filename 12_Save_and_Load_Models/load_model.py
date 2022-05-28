import torch
import model

#model state dictionary path
FILE = 'CNN.pth'

device = torch.device('cpu')

#since it is a dictionary, we must declare model structure first
#then load parameters
model = model.CNN()
for params in model.parameters():
    print(params)
#load parameters
#as default model parameters will load at the same device which it is saved
#if you want to use model on a different device, specify map_location to destination
model.load_state_dict(torch.load(FILE, map_location=device))
#as defalut a model locate at cpu, if your parameters is on dfferent device, make sure to specify
model.to(device)
#evaluation mode
model.eval()

for params in model.parameters():
    print(params)