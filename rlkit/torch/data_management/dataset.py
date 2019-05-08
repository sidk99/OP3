import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import ConcatDataset


class Dataset:
    def __init__(self, torch_dataset, batchsize=8):
        self.dataset = torch_dataset
        self.batchsize = batchsize

        self._dataloader = DataLoader(torch_dataset, batch_size=batchsize, shuffle=True)


    def set_batchsize(self, batchsize):
        self._dataloader = DataLoader(self.dataset, batch_size=batchsize, shuffle=True)

    def add(self, dataset):
        self.dataset = ConcatDataset([self.dataset, dataset])
        self._dataloader = DataLoader(self.dataset, batch_size=self.batchsize, shuffle=True)

    @property
    def dataloader(self):
        return self._dataloader

class BlocksDataset(Dataset):
    def __init__(self, torch_dataset, batchsize=8):
        super().__init__(torch_dataset, batchsize)
        if len(self.dataset.tensors) == 2:
            self.action_dim = self.dataset.tensors[1].shape[-1]
        else:
            self.action_dim = 0

