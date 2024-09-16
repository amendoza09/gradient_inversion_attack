import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

# define out data model CIFAR10
class CIFAR10_model(nn.Module):
    # stuff to define data model
    def __init__(self):
        super(CIFAR10Model, self).__init__()

# initialize CIFAR10
model = CIFAR10Model()

# og_data = take an image from data model to compare random noise to

# initialize random data using Gaussian Distribution
rand_data = torch.normal(0.5, 0.1, size=(1, 3, 32, 32), requires_grad=True)

# define loss funciton using L2 norm or Mean Squared Error
loss_func = nn.MSELoss()

# define optimize function using sgd with learning rate 0.1
optimize = optim.SGD([input], lr=0.1)

# loop for optimizition
for step in range(100):
    optimize.zero_grad() # reset gradient
    output = model(input) # 
    loss = loss_func(output, model(og_data)) # computing loss between outputs
    loss.backward() # backward pass for computing gradients
    optimize.step() # update the parameters based on gradients

    print(f"step: {step}, Loss: {loss.item()}")