import torch
from torch.utils.data.dataloader import DataLoader


class Dataset:
    def __init__(self, torch_dataset, batchsize=8):
        self.dataset = torch_dataset

        self._dataloader = DataLoader(torch_dataset, batch_size=8, shuffle=True)


    def add(self):
        pass

    @property
    def dataloader(self):
        return self._dataloader
