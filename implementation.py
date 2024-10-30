import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transform
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from pytorch_resnet_cifar10.resnet import resnet20

# modifications:
# change model to resnet20
# remove relu to sigmoid and remove strides
# change resnet.py to new resnet file
# use lbfgs instead of sgd

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
    test_load = DataLoader(test_data, batch_size=1, shuffle=False)
    steps = int(input("number of steps: "))
    
    # cross entropy
    crossE = nn.CrossEntropyLoss()

    # Get one image anc its label from the dataset
    target_image, target_label = next(iter(test_load))
    target_label = torch.tensor([target_label[0]], dtype=torch.long)
    
    #load model
    myModel = resnet20()
    myModel.eval()
    
    target_image.requires_grad_(True)
    
    output_target = myModel(target_image)
    loss = crossE(output_target, target_label)
     
    myModel.zero_grad()
    loss.backward()

    # getting gradients with respect to model parameter
    target_grad = [param.grad.clone() for param in myModel.parameters()]
    
    # initialize random data using Gaussian Distribution
    rand_data = torch.normal(mean=0.5, std=0.1, size=(1,3,32,32), requires_grad=True)

    # optimizer to update random image
    optimize = optim.LBFGS([rand_data], lr=1, max_iter=steps, history_size=100)
    
    # loss function using Mean Squared Error
    mse_loss = nn.MSELoss()

    def closure():
        
        output_rand = myModel(rand_data)
        loss_rand = crossE(output_rand, target_label)
        loss_rand.backward(retain_graph=True)  # Retain graph to access it later
        
        # Calculate gradients for random data
        rand_data_grad = [param.grad.clone().detach() for param in myModel.parameters()]
                
        # Compute the loss for matching gradients
        grad_loss = sum(mse_loss(rand_data_grad, target_grad) for rand_data_grad, target_grad in zip(rand_data_grad, target_grad))

        grad_loss.backward()  # Backpropagate the gradient loss
        
        return grad_loss
    
    # optimization loop
    for i in range(steps):
        # reset gradient so future calculations are not influenced
        optimize.step(closure)
        optimize.zero_grad()
        
        if i%10 == 0:
            current_loss = closure().item()
            print(f"Step: {i}, Loss: {current_loss}, Target label: {target_label.item()}")
            save_image(rand_rand, 'recovered_image.png')

    save_image(target_image, 'target_image.png')
    


if __name__ == "__main__":
    main()
