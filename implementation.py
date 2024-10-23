import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.transforms as transform
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from cifar10_models.resnet import resnet18


def main():
    
    # define transforms
    transform = transforms.Compose([
        transforms.Resize((32, 32)), # resnet size
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4822, 0.4465),
                             (0.2471, 0.2435, 0.2616))
    ])

    # load test data
    test_data = datasets.CIFAR10(root='./data',
                                 train=False,
                                 download=True,
                                 transform=transform)
    test_load = DataLoader(test_data, batch_size=28, shuffle=False)
    
    
    #load model
    myModel = resnet18(pretrained=True)
    myModel.eval()

    # cross entropy
    crossE = nn.CrossEntropyLoss()

    # Get one image anc its label from the dataset
    data_iter = iter(test_load)
    image, label = next(data_iter)
    target_image = image[25].unsqueeze(0)
    target_label = label[25].unsqueeze(0)
    
    
    target_image.requires_grad_(True)
    
    # forward and backward passing allows calculation of gradient
    # model processes target_image to create tensor, or forward pass to the model
    output_target = myModel(target_image)
    # find cross entropy between models output and the true label
    loss = crossE(output_target, target_label)
    # make sure all gradients are reset to not affect gradient calculation
    myModel.zero_grad()
    loss.backward()

    # getting gradients with respect to model parameter
    target_grad = [param.grad.clone() for param in myModel.parameters()]
    
    # initialize random data using Gaussian Distribution
    rand_data = torch.normal(0.5, 0.1, size=(1, 3, 32, 32), requires_grad=True)

    # optimizer to update random image
    optimize = optim.SGD([rand_data], lr=0.1)
    
    # loss function using Mean Squared Error
    mse_loss = nn.MSELoss()

    steps = int(input("number of steps: "))
    
    # optimization loop
    for i in range(steps):
        # reset gradient so future calculations are not influenced
        optimize.zero_grad()

        # forward pass the random image
        output_rand = myModel(rand_data)
        
        # calculate loss w respect to original label
        loss_rand = crossE(output_rand, target_label)
        
        # reset models gradients
        myModel.zero_grad()
        
        # backward pass for random image
        loss_rand.backward(create_graph=True)
        
        # save gradients from the model parameters based off the random image
        rand_data_grad = [param.grad.clone() for param in myModel.parameters()]

        # compare gradients between the random and target image, use sum 
        all_grad_losses = sum(mse_loss(rand_data_grad, target_grad) for rand_data_grad, target_grad in zip(rand_data_grad, target_grad))
        
        # backward pass the gradient loss
        all_grad_losses.backward()
        
        # update parameters
        optimize.step()

        if i%5 == 0:
            print(f"Step: {i}, Loss: {all_grad_losses.item()}")
            save_image(rand_data, 'recovereed_image.png')

    save_image(target_image, 'target_image.png')
    


if __name__ == "__main__":
    main()
