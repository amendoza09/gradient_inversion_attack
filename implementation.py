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
    test_load = DataLoader(test_data, batch_size=28, shuffle=False)
    
    
    #load model
    myModel = resnet20()
    myModel.eval()
    
    # cross entropy
    crossE = nn.CrossEntropyLoss()

    # Get one image anc its label from the dataset
    img_index = 20
    data_iter = iter(test_load)
    image, label = next(data_iter)
    target_image = image[img_index].unsqueeze(0)
    target_label = label[img_index].unsqueeze(0)
    
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
    optimize = optim.LBFGS([rand_data], lr=1, max_iter=20, history_size=100)
    
    def closure():
        optimize.zero_grad()
        output_rand = myModel(rand_data)
        loss_rand = crossE(output_rand, target_label)
        loss_rand.backward()
        return loss_rand
    
    # loss function using Mean Squared Error
    mse_loss = nn.MSELoss()

    steps = int(input("number of steps: "))
    
    # optimization loop
    for i in range(steps):
        # reset gradient so future calculations are not influenced
        optimize.step(closure)
        
        if i%10 == 0:
            print(f"Step: {i}, Loss: {closure().item()}, Target label: {target_label.item()}")
            save_image(rand_data, 'recovered_image.png')

    save_image(target_image, 'target_image.png')
    


if __name__ == "__main__":
    main()
