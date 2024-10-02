import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from cifar10_models.resnet import resnet50


def main():
    
    # define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # resnet size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # load test data
    test_data = datasets.CIFAR10(root='./data',
                                 train=False,
                                 download=True,
                                 transform=transform)
    test_load = DataLoader(test_data, batch_size=1, shuffle=False)

    # Get one image from the dataset
    data_iter = iter(test_load)
    target_image, target_label = next(data_iter)
    
    #load model
    myModel = resnet50(pretrained=True)
    myModel.eval()
        
    # cross entropy
    crossE = nn.CrossEntropyLoss()

    target_image.requires_grad_(True)
    
    # forward pass for target
    output = myModel(target_image)

    # get label of original image
    target_label = target_label.item()

    # calculating loss and gradient for target image
    target_label_tensor = torch.tensor([target_label], dtype=torch.long)
    loss = crossE(output, target_label_tensor)
    loss.backward()

    #getting gradients of the original image
    target_grad = target_image.grad.detach().clone()

    # initialize random data using Gaussian Distribution
    rand_data = torch.normal(0.5, 0.1, size=(1, 3, 224, 224), requires_grad=True)

    # optimizer to update random image
    optimize = optim.SGD([rand_data], lr=0.1)

    # loss function
    loss_func = nn.MSELoss()

    steps = int(input("number of steps: "))
    for i in range(steps):
        optimize.zero_grad()

        # forward pass for random image
        output_rand = myModel(rand_data)
        
        # calculate loss respect to original label
        loss_rand = crossE(output_rand, target_label_tensor)
        loss_rand.backward(create_graph=True)
        
        rand_data_grad = torch.autograd.grad(loss_rand, rand_data, create_graph=True)[0]

        # compare gradients
        grad_loss = loss_func(rand_data_grad, target_grad)
        # backward pass the gradient loss
        grad_loss.backward()
        
        # update parameters
        optimize.step()

        print(f"Step: {i}, Loss: {grad_loss.item()}")

    save_image(target_image, 'target_image.png')
    save_image(rand_data, 'recovereed_image.png')


if __name__ == "__main__":
    main()