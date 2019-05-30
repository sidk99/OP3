
from collections import OrderedDict

from rlkit.core.timer import timer

from rlkit.core import logger
import torch
import ray
import os

class RayVAETrainer:
    def __init__(self, trainer, train_dataset, test_dataset, variant, num_epochs):
        self.t = trainer
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.variant = variant
        self._start_epoch = 0
        self.epoch = self._start_epoch
        self.num_epochs = num_epochs



    def _train(self):

        epoch = self.epoch
        variant = self.variant
        t = self.t
        save_period = variant['save_period']

        should_save_imgs = (epoch % save_period == 0)
        train_stats = t.train_epoch(epoch)
        test_stats = t.test_epoch(epoch, train=False, batches=1,
                   save_reconstruction=should_save_imgs)
        train_stats.update(test_stats)
        t.test_epoch(epoch, train=True, batches=1,
                    save_reconstruction=should_save_imgs)

        torch.save(t.model.state_dict(), open(os.getcwd() + '/model_params.pkl', "wb"))
        
        done = False
        if epoch == self.num_epochs:
            done = True
        return train_stats, done


    def to(self, device):
        self.t.model.to(device)

    def _begin_epoch(self):
        timer.reset()

    def _end_epoch(self):
        self.epoch += 1

