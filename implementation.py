import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from cifar10_models.resnet import resnet50

# define data model
def load_model():
    model = models.resnet50(pretrained=True)
    model.eval
    return model

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
    target_image = next(data_iter)
    
    #load model
    model = load_model()
        
    # cross entropy
    crossE = nn.CrossEntropyLoss()

    # forward pass of target image
    target_image.requries_grad_(True)
    # gradient with respect to target image
    output = model(target_image)

    # get gradient of original image
    target_label = output.argmax(dim=1)

    # calculating loss and gradient for target image
    loss = crossE(output, target_label)
    loss.backward()

    #getting gradients of the original image
    target_grad = target_image.grad.detatch().clone()

    # initialize random data using Gaussian Distribution
    rand_data = torch.normal(0.5, 0.1, size=(1, 3, 224, 224), requires_grad=True)

    # optimizer to update random image
    optimize = optim.SGD([rand_data], lr=0.1)

    # loss function
    loss_func = nn.MSGLoss()

    steps = int(input("number of steps: "))
    for i in range(steps):
        optimize.zero_grad()

        # forward pass for random image
        output_rand = model(rand_data)
        loss_rand = nn.CrossEntropyLoss()(output_rand, target_label)
        loss_rand.backward()

        # compare gradients
        grad_loss = loss_func(rand_data.grad, target_grad)
        # backward pass the gradient loss
        grad_loss.backward()

        # update parameters
        optimize.step()

        if i%50 == 0:
            print(f"Step: {steps}, Loss: {grad_loss.item()}")

    save_image(target_image, 'target_image.png')
    save_image(rand_data, 'recovereed_image.png')


if __name__ == "__main__":
    main()
