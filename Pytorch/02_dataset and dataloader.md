# dataset and dataloader

Dataset stores the samples and their corresponding labels
DataLoader wraps an iterable around the Dataset to enable easy access to the samples.

## pre-loaded datasets 

Pytorch 提供了一些 pre-loaded datasets, 例如: CIFAR, MNIST 等, 完整列表見 [Docs > torchvision.datasets](https://pytorch.org/vision/stable/datasets.html#)

Example: load the Fashion-MNIST dataset from TorchVision

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
```

```python
torchvision.datasets.FashionMNIST(
    root: str,  # test/train data 儲存的 path
    train: bool = True, # 說明是 test or train data
    transform: Optional[Callable] = None, # feature transformations
    target_transform: Optional[Callable] = None, # label transformations
    download: bool = False) # 如果位於 root 上還沒有該 dataset 則進行下載
```

## 自定義 dataset