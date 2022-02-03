from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode):
        super(ChallengeDataset, self).__init__()
        self.data = data
        self.mode = mode
        if self.mode == 'train':
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.RandomGrayscale(p=0.1),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(train_mean, train_std)
            ])
        else:
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(train_mean, train_std)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        data = self.data.to_numpy()
        image = gray2rgb(imread(data[index][0]))
        defect = np.array(data[index, 1:], dtype=float)
        #defect = (np.array(defect)).reshape(-1, 1)  # reshape to (L,1) array where L is no. of type of defects

        return self._transform(image), torch.tensor(defect)
