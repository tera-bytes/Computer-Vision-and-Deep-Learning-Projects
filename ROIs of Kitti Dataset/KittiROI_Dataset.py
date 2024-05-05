import os
import fnmatch
import torch
from torch.utils.data import Dataset
from torchvision import transforms, models
import cv2

class KittiROIDataset(Dataset):
    def __init__(self, label_file, img_dir, transform=None, target_transform=None):
        labels = []
        with open(label_file, "r") as f:
            for line in f:
                labels.append(line)

        self.img_labels = labels
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels[idx].split()[0]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        label = int(self.img_labels[idx].split()[1])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __next__(self):
        if (self.num >= self.max):
            raise StopIteration
        else:
            self.num += 1
            return self.__getitem__(self.num-1)
