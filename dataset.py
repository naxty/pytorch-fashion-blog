from torch.utils.data.dataset import Dataset
import torch
import os
import gzip
import numpy as np

class FashionData(Dataset):

    def __init__(self, image_path, label_path, DATA_PATH="./fashionmnist/"):
        images_path = os.path.join(DATA_PATH, image_path)
        labels_path = os.path.join(DATA_PATH, label_path)
        with open(labels_path, 'rb') as lbpath:
            self.labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                   offset=8)
        with open(images_path, 'rb') as imgpath:
            self.images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                   offset=16).reshape(len(self.labels), 28, 28)
        self.data_length = len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        image = torch.from_numpy(image)
        image = image.type("torch.FloatTensor")
        label = torch.from_numpy(np.array(label))
        label = label.type("torch.LongTensor")
        return (image, label)
        
    def __len__(self):
        return self.data_length