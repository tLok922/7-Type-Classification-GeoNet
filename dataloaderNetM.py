import os
from torch.utils import data
from torchvision import transforms
import scipy.io as spio
import numpy as np
import skimage
import torch

"""Custom Dataset compatible with prebuilt DataLoader."""
class DistortionDataset(data.Dataset):
    def __init__(self, distortedImgDir, flowDir, transform, distortion_types, data_num):
        #29956/31871 removed
        self.distorted_image_paths = []
        self.displacement_paths = []
        self.distortion_types = distortion_types

        for fs in os.listdir(distortedImgDir):
            types = fs.split('_')[0]
            if types in distortion_types:
                self.distorted_image_paths.append(os.path.join(distortedImgDir, fs))

        for fs in os.listdir(flowDir):
            types = fs.split('_')[0]
            if types in distortion_types:
                self.displacement_paths.append(os.path.join(flowDir, fs))

        self.distorted_image_paths.sort()
        self.displacement_paths.sort()
        self.transform = transform

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        distorted_image_path = self.distorted_image_paths[index]
        displacement_path = self.displacement_paths[index]

        distorted_image = skimage.io.imread(distorted_image_path)
        displacement = spio.loadmat(displacement_path)
        displacement_x = displacement['u'].astype(np.float32)[np.newaxis,:]
        displacement_y = displacement['v'].astype(np.float32)[np.newaxis,:]

        label_type = os.path.basename(distorted_image_path).split('_')[0]
        label = self.distortion_types.index(label_type)

        if self.transform is not None:
            trans_distorted_image = self.transform(distorted_image)
        else:
            trans_distorted_image = torch.tensor(distorted_image.transpose(2,0, 1), dtype=torch.float32) / 255.0

        return trans_distorted_image, displacement_x, displacement_y, label

    def __len__(self):
        return len(self.distorted_image_paths)

def collate_fn(batch):
    images, disx, disy, labels = zip(*batch)
    images = torch.stack(images, dim=0) #.unsqueeze(0)

    disx = torch.stack([torch.from_numpy(np.array(dx,dtype=np.float32))for dx in disx],dim=0)
    disy = torch.stack([torch.from_numpy(np.array(dy,dtype=np.float32))for dy in disy],dim=0)
    labels = torch.tensor(labels, dtype=torch.int)

    return images, disx, disy, labels


def get_loader(distortedImgDir, flowDir, batch_size, distortion_type, data_num):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = DistortionDataset(distortedImgDir, flowDir, transform, distortion_type, data_num)
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    return data_loader
