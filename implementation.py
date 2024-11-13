import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PyTorch_CIFAR10.cifar10_models.resnet import resnet18
from torch.autograd import grad
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.utils import save_image
from lenet import LeNet, weights_init


def deep_leakage(model, origin_grad, origin_label):
    # initialize random data using Gaussian Distribution
    dummy_data = torch.normal(mean=0.5, std=0.1, size=(1, 3, 32, 32),
                              requires_grad=True,
                              device=device)

    # optimizer to update random image
    optimize = optim.LBFGS([dummy_data], lr=1.0, max_iter=300, history_size=100)

    # loss function using Mean Squared Error
    mse_loss = nn.MSELoss()
    crossEnt = nn.CrossEntropyLoss()
    # steps = int(input("Enter number of steps: "))
    steps = 300

    def closure():
        optimize.zero_grad()

        output = model(dummy_data)
        loss = crossEnt(output, origin_label)

        # Calculate gradients for random data
        reconstructed_grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        # Compute the loss for matching gradients
        grad_loss = sum(
            mse_loss(reconstructed_g, origin_g)
            for reconstructed_g, origin_g in zip(reconstructed_grad, origin_grad)
        )

        grad_loss.backward()  # Backpropagate the gradient loss
        return grad_loss

    for i in range(steps):
        # reset gradient so future calculations are not influenced
        loss_val = optimize.step(closure)
        if i % 10 == 0:
            print(f"Step: {i}, Loss: {loss_val.item()}")
            save_image(dummy_data.data, "reconstructed_image.png")
    return dummy_data


def visualize_dummy_data(dummy_data):
    # Convert to numpy for visualization
    dummy_data_np = dummy_data.detach().cpu().numpy()

    # Plot the first few images
    num_images = min(5, dummy_data_np.shape[0])
    plt.figure(figsize=(10, 2))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(dummy_data_np[i].transpose(1, 2, 0))  # Change shape for plotting
        plt.axis("off")
    plt.show()


def main():
    # define transforms
    transform = transforms.Compose([
        # transforms.Resize((32, 32)),  # resnet size
        transforms.ToTensor(),
        # transforms.Normalize((0.4915, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])
    # load test data
    test_data = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_load = DataLoader(test_data, batch_size=1, shuffle=False)

    # define model
    # myModel = resnet18(pretrained=False)
    myModel = LeNet(channel=3, hidden=768, num_classes=10)
    myMOdel.apply(weights_init)
    myModel = myModel.to(device)
    # myModel.eval()

    # cross entropy
    crossEnt = nn.CrossEntropyLoss()

    # Get one image and its label from the dataset
    image, label = next(iter(test_load))
    target_label = torch.tensor([label], dtype=torch.long, device=device)
    save_image(image, "target_image.png")

    image, label = image.to(device), label.to(device)
    output = myModel(image)
    loss = crossEnt(output, target_label)

    myModel.zero_grad()
    loss.backward()

    # getting gradients with respect to model parameter
    target_grads = [param.grad.detach().clone() for param in myModel.parameters()]

    dummy_data = deep_leakage(myModel, target_grads, target_label)
    visualize_dummy_data(dummy_data)


if __name__ == "__main__":
    device = torch.device('cuda:0')
    main()
