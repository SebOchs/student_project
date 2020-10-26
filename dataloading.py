import numpy as np
import torch
from torch.utils.data import Dataset


# data loader for a given data set
class SemEvalDataset(Dataset):

    def __init__(self, filename):
        self.data = np.load(filename, allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text, lab = self.data[index]
        return torch.tensor(text).long(), torch.tensor(lab).long()