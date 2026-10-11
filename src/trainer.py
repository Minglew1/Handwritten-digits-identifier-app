import os
import torch
import torch.nn as nn
import torch.optim as optim

from src.model import DigitClassifier
from src.dataset import get_data_loaders


def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    train_size = len(train_loader.dataset)

    for batch_id, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if batch_id % 100 == 0:
            processed = batch_id * len(data)
            print(f"\tLoss: {loss.item():>7.6f} [{processed:>5d}/{train_size:>5d}]")


def evaluate(model, test_loader, loss_fn, device):
    model.eval()
    test_size = len(test_loader.dataset)
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / test_size
    print(f"\t\tValidation Accuracy: {accuracy:>0.1f}% | Avg Loss: {avg_loss:>8.6f}\n")


def get_or_train_model(model_path: str = "mnist_model.pth", device: str = "cpu") -> nn.Module:
    model = DigitClassifier().to(device)

    if os.path.exists(model_path):
        print(f"Loading pre-trained model from '{model_path}'...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model

    print("No pre-trained model found. Starting training...")
    torch.manual_seed(123)
    train_loader, test_loader = get_data_loaders()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    episodes = 10
    for t in range(episodes):
        print(f"Iteration {t+1}/{episodes}\n-------------------------------")
        train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        evaluate(model, test_loader, loss_fn, device)

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to '{model_path}'")
    model.eval()
    return model