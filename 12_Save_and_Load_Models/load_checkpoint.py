import torch
import model

FILE = "Checkpoint.pth"

#declare model and optimizer to load the saved parameters
model = model.CNN()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

#load checkpoint
checkpoint = torch.load(FILE)
model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optimizer_state"])
epoch = checkpoint["epoch"]
model.eval()

print(optimizer.state_dict())
