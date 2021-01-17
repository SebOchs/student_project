import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# data loader for a given data set
class T5Dataset(Dataset):

    def __init__(self, filename):
        self.data = np.load(filename, allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        input = data['input']
        output = data['answer']
        label = data['label']
        return torch.tensor(input).long(), torch.tensor(output).long(), torch.tensor(label).long()
