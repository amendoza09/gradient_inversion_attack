import argparse
import matplotlib.pyplot as plt
import torch
import certifi
import ssl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.utils import save_image
from lenet import LeNet, weights_init

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

def parse_args():
    parser = argparse.ArgumentParser(description="Deep Leakage from Gradients")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use: 'auto' (GPU if available, else CPU), 'cpu', or 'cuda'",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Number of optimization steps (default: 300)",
    )
    parser.add_argument(
        "--image-index",
        type=int,
        default=0,
        help="Index of the image to use from the test dataset (default: 0)",
    )
    return parser.parse_args()

def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif device_arg == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device

def deep_leakage(model, origin_grad, origin_label, device, steps=300):
    dummy_data = torch.normal(
        mean=0.5, std=0.1, size=(1, 3, 32, 32),
        requires_grad=True,
        device=device,
    )

    optimize = optim.LBFGS([dummy_data], lr=0.5, max_iter=20, history_size=100)
    mse_loss = nn.MSELoss()
    crossEnt = nn.CrossEntropyLoss()
    history = []  # collect snapshots here

    def closure():
        optimize.zero_grad()
        output = model(dummy_data)
        loss = crossEnt(output, origin_label)
        reconstructed_grad = torch.autograd.grad(
            loss, model.parameters(), create_graph=True
        )
        grad_loss = sum(
            mse_loss(reconstructed_g, origin_g)
            for reconstructed_g, origin_g in zip(reconstructed_grad, origin_grad)
        )
        grad_loss.backward()
        return grad_loss

    for i in range(steps):
        loss_val = optimize.step(closure)
        if i % 50 == 0:
            print(f"Step: {i}, Loss: {loss_val.item():.9f}")
            # save snapshot every 50 steps
            snapshot = dummy_data.detach().cpu().numpy()[0].transpose(1, 2, 0)
            snapshot = (snapshot - snapshot.min()) / (snapshot.max() - snapshot.min())  # normalize for display
            history.append(snapshot)

    save_image(dummy_data.data, "reconstructed_image.png")
    return dummy_data, history

def reconstruct_label(origin_grad, model):
    # The true label corresponds to the most negative gradient 
    # in the final fully connected layer's weight gradients
    last_weight_grad = origin_grad[-2]  # second to last = FC weight grad
    reconstructed_label = torch.argmin(torch.sum(last_weight_grad, dim=-1))
    return reconstructed_label.reshape((1,))

def visualize_dummy_data(dummy_data, history, target_image):
    # normalize target for display
    target_np = target_image.detach().cpu().numpy()[0].transpose(1, 2, 0)
    target_np = (target_np - target_np.min()) / (target_np.max() - target_np.min())

    num_snapshots = len(history)
    cols = 7
    rows = (num_snapshots + cols - 1) // cols  # dynamic rows based on steps

    plt.figure(figsize=(cols * 2, rows * 2 + 2))

    for i, snapshot in enumerate(history):
        plt.subplot(rows + 1, cols, i + 1)
        plt.imshow(snapshot)
        plt.title(f"iter={i * 10}", fontsize=7)
        plt.axis("off")

    # show target image in last row
    plt.subplot(rows + 1, cols, num_snapshots + 1)
    plt.imshow(target_np)
    plt.title("target", fontsize=7)
    plt.axis("off")

    plt.suptitle("Reconstruction Progress", fontsize=12)
    plt.tight_layout()
    plt.show()
    
def main():
    args = parse_args()
    device = get_device(args.device)

    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
    ])

    # STL10 — same size, 100 classes instead of 10
    # test_data = datasets.STL10(root="./data/", split="test", download=True, transform=transform)

    # CIFAR10 - 10 classes
    # test_data = datasets.CIFAR10(
    #    root="./data/", train=False, download=False, transform=transform
    #)
    # CIFAR100 - 100 classes
    test_data = datasets.CIFAR100(
        root="./data/", train=False, download=False, transform=transform
    )

    test_load = DataLoader(test_data, batch_size=1, shuffle=False)

    # Iterate to the correct index — remove the next(iter()) call below
    for i, (image, label) in enumerate(test_load):
        if i == args.image_index:
            break

    # Define model
    myModel = LeNet(channel=3, hidden=768, num_classes=100)
    myModel.apply(weights_init)
    myModel = myModel.to(device)

    crossEnt = nn.CrossEntropyLoss()

    # Use the image from the loop above — do NOT call next(iter()) again
    save_image(image, "target_image.png")

    image = image.to(device)
    output = myModel(image)

    real_label = label.to(device)
    loss = crossEnt(output, real_label)
    myModel.zero_grad()
    loss.backward()

    target_grads = [param.grad.detach().clone() for param in myModel.parameters()]

    target_label = reconstruct_label(target_grads, myModel)

    print(f"\n{'='*40}")
    print(f"  Image index       : {args.image_index}")
    print(f"  Real label        : {label.item()} ({test_data.classes[label.item()]})")
    print(f"  Reconstructed     : {target_label.item()} ({test_data.classes[target_label.item()]})")
    print(f"  Image shape       : {image.shape}")
    print(f"  Saved to          : target_image.png")
    print(f"{'='*40}\n")

    target_image = image.cpu()

    dummy_data, history = deep_leakage(myModel, target_grads, target_label, device, steps=args.steps)
    visualize_dummy_data(dummy_data, history, target_image)
    
if __name__ == "__main__":
    main()