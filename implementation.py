import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_data =datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

test_load = DataLoader(test_data, batch_size=128, shuffle=False)

# Get one image from the dataset
data_iter = iter(test_load)
image, label = next(data_iter)
target = image[127].unsqueeze(0)

# save the image
save_image(target, 'target.png')

# initialize random data using Gaussian Distribution
rand_data = torch.normal(0.5, 0.1, size=(1, 3, 32, 32), requires_grad=True)

# define loss funciton using L2 norm or Mean Squared Error
loss_func = nn.MSELoss()

# define optimize function using sgd with learning rate 0.1
optimize = optim.SGD([rand_data], lr=0.1)

# number of steps
steps = 100000

# loop for optimizition
for step in range(steps):
    optimize.zero_grad() # reset gradient
    loss = loss_func(rand_data, target) # computing loss between outputs
    loss.backward() # backward pass for computing gradients
    optimize.step() # update the parameters based on gradients

    if (step % 100) == 0:
        print(f"step: {step}, Loss: {loss.item()}")

save_image(rand_data, 'recovered_img.png')