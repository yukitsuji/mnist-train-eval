import argparse
import os
import torch
from torchvision import datasets, transforms


def test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(test_loader.dataset)
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = torch.load(os.path.join(args.model, "model.pkl"))
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset2 = datasets.MNIST("../data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=64)
    acc = test(model, test_loader)
    os.makedirs(os.path.split(os.path.abspath(args.out))[0], exist_ok=True)
    with open(args.out, "w") as f:
        f.write(str(acc))


if __name__ == "__main__":
    main()
