import torch
import certifi
import ssl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from pytorch_resnet_cifar10.resnet import resnet20
#from PyTorch_CIFAR10.cifar10_models.resnet import resnet18

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

def deep_leakage(model, origin_grad, origin_label):
    # initialize random data using Gaussian Distribution
    dummy_data = torch.normal(mean=0.5, std=0.1, size=(1, 3, 32, 32),
                              requires_grad=True,
                              device=device)

    # optimizer to update random image
    optimize = optim.LBFGS([dummy_data],
                           lr=0.1,
                           max_iter=100, 
                           history_size=100,
                           )

    # loss function using Mean Squared Error
    mse_loss = nn.MSELoss()
    crossEnt = nn.CrossEntropyLoss()
    # steps = int(input("Enter number of steps: "))
    steps = 1200
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

    def closure():
        dummy_data.data.clamp_(0, 1)
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
        if i % 1 == 0:
            print(f"Step {i} | grad_loss: {loss_val.item():.8f} | "
                f"dummy mean: {dummy_data.data.mean():.4f} | "
                f"dummy std: {dummy_data.data.std():.4f}")
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
        #transforms.Normalize((0.4915, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])
    # load test data
    test_data = datasets.CIFAR10(
        root="./pytorch_resnet_cifar10/data", train=False, download=False, transform=transform
    )
    test_load = DataLoader(test_data, batch_size=1, shuffle=False)

    # define model
    myModel = resnet20().to(device)
    # checkpoint = torch.load('./pytorch_resnet_cifar10/save_resnet20/model.th', 
                        # map_location=device)
    # myModel.load_state_dict(checkpoint['state_dict'])
    # myModel.apply(weights_init)
    myModel.eval()

    # cross entropy
    crossEnt = nn.CrossEntropyLoss()

    # Get one image and its label from the dataset
    image, label = next(iter(test_load))
    image, label = image.to(device), label.to(device)
    save_image(image, "ground_truth.png")

    
    output = myModel(image)
    loss = crossEnt(output, label)

    myModel.zero_grad()
    loss.backward()

    # getting gradients with respect to model parameter
    target_grads = [param.grad.detach().clone() for param in myModel.parameters()]

    dummy_data = deep_leakage(myModel, target_grads, label)
    visualize_dummy_data(dummy_data)


if __name__ == "__main__":
    device = torch.device('cpu')
    main()