import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import grad
from PyTorch_CIFAR10.cifar10_models.resnet import resnet18


def deep_leakage(model, origin_grad):
    # initialize random data using Gaussian Distribution
    dummy_data = torch.normal(mean=0.5, std=0.1, size=(1,3,32,32), requires_grad=True)

    # optimizer to update random image
    optimize = optim.LBFGS([dummy_data], lr=1, max_iter=300, history_size=100)

    # loss function using Mean Squared Error
    mse_loss = nn.MSELoss()
    crossEnt = nn.CrossEntropyLoss()
    steps = int(input("Enter number of steps: "))
    for i in range(steps):
        def closure():
            optimize.zero_grad()

            reconstructed_output = model(dummy_data)

            reconstructed_loss = crossEnt(reconstructed_output, origin_grad)
            # Calculate gradients for random data
            reconstructed_grad = torch.autograd.grad(reconstructed_loss, model.parameters(), create_graph=True)

            grad_loss = sum(mse_loss(reconstructed_grad, origin_grad)
                            for reconstructed_grad, origin_grad in zip(reconstructed_grad, target_grad))

            grad_loss.backward()  # Backpropagate the gradient loss
            return grad_loss

        # reset gradient so future calculations are not influenced
        loss_val = optimize.step(closure)
        if i%10 == 0:
            print(f"Step: {i}, Loss: {loss_val.item()}")
            save_image(dummy_data.data, 'reconstructed_image.png')
    return dummy_data

def visualize_dummy_data(dummy_data):
    # Convert to numpy for visualization
    dummy_data_np = dummy_data.detach().numpy()

    # Plot the first few images
    num_images = min(5, dummy_data_np.shape[0])
    plt.figure(figsize=(10, 2))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(dummy_data_np[i].transpose(1, 2, 0))  # Change shape for plotting
        plt.axis('off')
    plt.show()

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

    # define model
    myModel = resnet18(pretrained=True)
    myModel.eval()

    # cross entropy
    crossEnt = nn.CrossEntropyLoss()

    # Get one image and its label from the dataset
    image, label = next(iter(test_load))
    target_image = image[0].unsqueeze(0)
    target_label = image[0].unsqueeze(0)
    target_label = torch.tensor([target_label], dtype=torch.long)
    save_image(target_image, 'target_image.png')

    output = myModel(target_image)
    loss = crossEnt(output, target_label)

    myModel.zero_grad()
    loss.backward()

    # getting gradients with respect to model parameter
    target_grads = [param.grad.clone().detach() for param in myModel.parameters()]
                                                                                                      94,1          85%
