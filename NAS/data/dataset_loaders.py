import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.v2 as v2
from NAS.config import Config
class MNIST:

    def __init__(self):
        """
        Initialize MNIST dataset loader with transforms and data path.
        """
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])
        self.data_path = Config.data_path + '/mnist'

    def get_dataloaders(self, batch_size, num_workers=0, pin_memory=False):
        """
        Get train and test dataloaders for MNIST dataset.
        Args:
            batch_size (int): Batch size for dataloaders.
            num_workers (int): Number of worker threads.
            pin_memory (bool): Whether to use pinned memory.
        Returns:
            tuple: (train_loader, test_loader)
        """
        mnist_train = datasets.MNIST(self.data_path, train=True, download=True, transform=self.transform)
        mnist_test = datasets.MNIST(self.data_path, train=False, download=True, transform=self.transform)
        train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, persistent_workers=True, pin_memory=pin_memory)
        test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers, persistent_workers=True,pin_memory=pin_memory)
        return train_loader, test_loader

class SVHN:

    def __init__(self):
        """
        Initialize SVHN dataset loader with transforms and data path.
        """
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((28, 28)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ToDtype(torch.float32,scale = True),
            v2.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        self.data_path = Config.data_path + '/svhn'

    def get_dataloaders(self, batch_size, num_workers=1):
        """
        Get train and test dataloaders for SVHN dataset.
        Args:
            batch_size (int): Batch size for dataloaders.
            num_workers (int): Number of worker threads.
        Returns:
            tuple: (train_loader, test_loader)
        """
        mnist_train = datasets.SVHN(self.data_path, split = 'train', download=True, transform=self.transform)
        mnist_test = datasets.SVHN(self.data_path, split = 'test', download=True, transform=self.transform)
        train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers, pin_memory=True)
        return train_loader, test_loader

class CIFAR10:

    def __init__(self, transform=None, download=False):
        """
        Initialize CIFAR10 dataset loader with transforms and data path.
        Args:
            transform: Transformations to apply.
            download (bool): Whether to download the dataset.
        """
        self.data_path = Config.data_path + '/cifar10'
        self.dataset_train = datasets.CIFAR10(self.data_path,transform = transform,download=download, train=True)
        self.dataset_val = datasets.CIFAR10(self.data_path,transform = transform,download=download, train=False)

    def get_dataloaders(self, batch_size, num_workers=0, pin_mem=False):
        """
        Get train and validation dataloaders for CIFAR10 dataset.
        Args:
            batch_size (int): Batch size for dataloaders.
            num_workers (int): Number of worker threads.
            pin_mem (bool): Whether to use pinned memory.
        Returns:
            tuple: (train_loader, val_loader)
        """
        pers_work = False
        if num_workers > 0:
            pers_work = True
        train_loader = DataLoader(self.dataset_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, persistent_workers=pers_work, pin_memory = pin_mem)
        val_loader = DataLoader(self.dataset_val, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers, persistent_workers=pers_work, pin_memory=pin_mem)
        return train_loader, val_loader

class EuroSAT:

    def __init__(self):
        """
        Initialize EuroSAT dataset loader with transforms and data path.
        """
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.data_path = Config.data_path + '/eurosat'


    def get_dataloaders(self, batch_size, num_workers=0):
        """
        Get train and validation dataloaders for EuroSAT dataset.
        Args:
            batch_size (int): Batch size for dataloaders.
            num_workers (int): Number of worker threads.
        Returns:
            tuple: (train_loader, val_loader)
        """
        pers_work = False
        if num_workers > 0:
            pers_work = True
        
        train_val = datasets.EuroSAT(self.data_path, download=True, transform=self.transform)
        generator = torch.Generator().manual_seed(0)
        train_dataset, val_dataset = torch.utils.data.random_split(train_val, [0.8, 0.2], generator=generator)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, persistent_workers=pers_work)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers, persistent_workers=pers_work)
        return train_loader, val_loader


class FakeData:

    def __init__(self):
        """
        Initialize FakeData dataset loader with transforms.
        """
        self.transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def get_dataloaders(self, batch_size=32, num_workers=1):
        """
        Get train and test dataloaders for FakeData dataset.
        Args:
            batch_size (int): Batch size for dataloaders.
            num_workers (int): Number of worker threads.
        Returns:
            tuple: (train_loader, test_loader)
        """
        train = datasets.FakeData(size = 64, image_size = (3, 28, 28), num_classes = 10, transform = self.transform)
        test = datasets.FakeData(size = 64, image_size = (3, 28, 28), num_classes = 10, transform = self.transform)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)
        return train_loader, test_loader
