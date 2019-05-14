
from collections import OrderedDict

from rlkit.core.timer import timer

from rlkit.core import logger

class RayVAETrainer:
    def __init__(self, trainer, train_dataset, test_dataset, variant):
        self.t = trainer
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.variant = variant
        self._start_epoch = 0
        self.epoch = self._start_epoch


    def _train(self):

        epoch = self.epoch
        variant = self.variant
        t = self.t
        save_period = variant['save_period']
        #for epoch in range(variant['num_epochs']):
        should_save_imgs = (epoch % save_period == 0)
        # train_stats = t.train_epoch(epoch)
        # test_stats = t.test_epoch(epoch, save_vae=False, train=False, record_stats=True, batches=1,
        #            save_reconstruction=should_save_imgs)
        # train_stats.update(test_stats)
        #t.test_epoch(epoch, save_vae=False, train=True, record_stats=False, batches=1,
        #             save_reconstruction=should_save_imgs)
            #torch.save(m.state_dict(), open(logger._snapshot_dir + '/params.pkl', "wb"))

        #return train_stats, False
        return {'train/mse': 0}, True


    def to(self, device):
        pass

    def _begin_epoch(self):
        timer.reset()

    def _end_epoch(self):
        self.epoch += 1

