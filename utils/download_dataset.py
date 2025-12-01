import urllib.request, zipfile, os

url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
out = "tiny-imagenet-200.zip"
urllib.request.urlretrieve(url, out)

with zipfile.ZipFile(out, 'r') as z:
    z.extractall('tiny-imagenet')

import os
import shutil

with open('tiny-imagenet/tiny-imagenet-200/val/val_annotations.txt') as f:
    for line in f:
        fn, cls, *_ = line.split('\t')
        os.makedirs(f'tiny-imagenet/tiny-imagenet-200/val/{cls}', exist_ok=True)

        shutil.copyfile(f'tiny-imagenet/tiny-imagenet-200/val/images/{fn}', f'tiny-imagenet/tiny-imagenet-200/val/{cls}/{fn}')

shutil.rmtree('tiny-imagenet/tiny-imagenet-200/val/images')

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T

transform = {
    'train': T.Compose([
    T.Resize((224, 224)),  # Resize to fit the input dimensions of the network (224 è la dimensione standard per i CNN)
    T.RandomHorizontalFlip(), # Randomly flip the images for data augmentation
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]),
    'val': T.Compose([
    T.Resize((224, 224)),  # Resize to fit the input dimensions of the network (224 è la dimensione standard per i CNN)
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
}

tiny_imagenet_dataset_train = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/train', transform=transform['train']) # applica la trasformazione del 'train'
tiny_imagenet_dataset_val = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/val', transform=transform['val'])       # applica la trasformazione del 'val'

print(f"Length of train dataset: {len(tiny_imagenet_dataset_train)}")
print(f"Length of val dataset: {len(tiny_imagenet_dataset_val)}")

train_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_train, batch_size=32, shuffle=True, num_workers=2) # reduced from 8 to 2 the workers as suggested by the warning
val_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_val, batch_size=32, shuffle=False)                   # batch_size può essere aumentata a 64 se continua ad essere overfitting

