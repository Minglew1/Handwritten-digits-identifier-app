from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(batch_size: int = 128):
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        transforms.ToTensor()
    ])

    train_set = datasets.MNIST(root='./data', train=True, transform=train_transform, download=True)
    test_set = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader