from os import path as osp
import numpy as np
import torch
from torch import optim
from torchvision.utils import save_image
from rlkit.core import logger
from rlkit.core.serializable import Serializable
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.pytorch_util import from_numpy


class IodineTrainer(Serializable):
    def __init__(
            self,
            train_dataset,
            test_dataset,
            model,
            train_seedsteps,
            test_seedsteps,
            batch_size=128,
            log_interval=0,
            gamma=0.5,
            lr=1e-3,
    ):
        self.quick_init(locals())
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.beta = model.beta
        self.gamma = gamma
        self.imsize = model.imsize

        model.to(ptu.device)

        self.model = model
        self.representation_size = model.representation_size
        self.input_channels = model.input_channels
        self.imlength = model.imlength
        self.train_seedsteps = train_seedsteps
        self.test_seedsteps = test_seedsteps

        self.lr = lr
        params = list(self.model.parameters())
        self.optimizer = optim.Adam(params, lr=self.lr)
        self.train_dataset, self.test_dataset = train_dataset, test_dataset

        self.batch_size = batch_size

        self.vae_logger_stats_for_rl = {}
        self._extra_stats_to_log = None

    def prepare_tensors(self, tensors):
        imgs = tensors[0].to(ptu.device) / 255.
        if len(tensors) == 2:
            return imgs, tensors[1].to(ptu.device)
        else:
            return imgs, None


    def train_epoch(self, epoch):
        self.model.train()
        losses, log_probs, kles, mses = [], [], [], []

        for batch_idx, tensors in enumerate(self.train_dataset.dataloader):
            obs, actions = self.prepare_tensors(tensors)
            self.optimizer.zero_grad()

            # schedule for doing refinement or physics
            # refinement = 0, physics = 1
            # when only doing refinement predict same image
            # when only doing physics predict next image
            schedule = np.random.randint(0, 2, (self.model.T,))
            schedule[:4] = 0
            x_hat, mask, loss, kle_loss, x_prob_loss, mse, final_recon, lambdas = self.model(obs, actions=actions, schedule=schedule)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([x for x in self.model.parameters()], 5.0)
            self.optimizer.step()

            losses.append(loss.item())
            log_probs.append(x_prob_loss.item())
            kles.append(kle_loss.item())
            mses.append(mse.item())

            if self.log_interval and batch_idx % self.log_interval == 0:
                print(x_prob_loss.item(), kle_loss.item())

        logger.record_tabular("train/epoch", epoch)
        logger.record_tabular("train/Log Prob", np.mean(log_probs))
        logger.record_tabular("train/KL", np.mean(kles))
        logger.record_tabular("train/loss", np.mean(losses))
        logger.record_tabular("train/mse", np.mean(mses))


    def test_epoch(
            self,
            epoch,
            save_reconstruction=True,
            save_vae=True,
            record_stats=True,
            train=True,
            batches=1,
    ):
        T = 9
        schedule = np.ones((T,))
        schedule[:4] = 0

        self.model.eval() #TODO Get around needing gradients during eval mode
        losses, log_probs, kles, mses = [], [], [], []
        dataloader = self.train_dataset.dataloader if train else self.test_dataset.dataloader
        for batch_idx, tensors in enumerate(dataloader):
            obs, actions = self.prepare_tensors(tensors)
            self.optimizer.zero_grad()

            x_hats, masks, loss, kle_loss, x_prob_loss, mse, final_recon, lambdas = self.model(obs, actions=actions, schedule=schedule)

            losses.append(loss.item())
            log_probs.append(x_prob_loss.item())
            kles.append(kle_loss.item())
            mses.append(mse.item())


            if batch_idx == 0 and save_reconstruction:
                t_sample = np.cumsum(schedule)
                t_sample[t_sample >= obs.shape[1]] = obs.shape[1] - 1
                ground_truth = obs[0][t_sample].unsqueeze(0)
                K = self.model.K
                imsize = ground_truth.shape[-1]

                m = torch.stack([m[0] for m in masks]).permute(1, 0, 2, 3).unsqueeze(2).repeat(1, 1, 3, 1, 1)
                x = torch.stack(x_hats)[:, :K].permute(1, 0, 2, 3, 4)
                rec = (m * x)
                full_rec = rec.sum(0, keepdim=True)

                comparison = torch.cat([ground_truth, full_rec, m, rec], 0).view(-1, 3, imsize, imsize)

                save_dir = osp.join(logger.get_snapshot_dir(),
                                    '%s_r%d.png' % ('train' if train else 'val', epoch))
                save_image(comparison.data.cpu(), save_dir, nrow=T)
            break

        if record_stats:
            logger.record_tabular("test/Log Prob", np.mean(log_probs))
            logger.record_tabular("test/KL", np.mean(kles))
            logger.record_tabular("test/loss", np.mean(losses))
            logger.record_tabular("test/mse", np.mean(mses))
            logger.dump_tabular()
        if save_vae:
            logger.save_itr_params(epoch, self.model)  # slow...

    def debug_statistics(self):
        """
        Given an image $$x$$, samples a bunch of latents from the prior
        $$z_i$$ and decode them $$\hat x_i$$.
        Compare this to $$\hat x$$, the reconstruction of $$x$$.
        Ideally
         - All the $$\hat x_i$$s do worse than $$\hat x$$ (makes sure VAE
           isn’t ignoring the latent)
         - Some $$\hat x_i$$ do better than other $$\hat x_i$$ (tests for
           coverage)
        """
        debug_batch_size = 64
        data = self.get_batch(train=False)
        reconstructions, _, _ = self.model(data)
        img = data[0]
        recon_mse = ((reconstructions[0] - img) ** 2).mean().view(-1)
        img_repeated = img.expand((debug_batch_size, img.shape[0]))

        samples = ptu.randn(debug_batch_size, self.representation_size)
        random_imgs, _ = self.model.decode(samples)
        random_mses = (random_imgs - img_repeated) ** 2
        mse_improvement = ptu.get_numpy(random_mses.mean(dim=1) - recon_mse)
        stats = create_stats_ordered_dict(
            'debug/MSE improvement over random',
            mse_improvement,
        )
        stats.update(create_stats_ordered_dict(
            'debug/MSE of random decoding',
            ptu.get_numpy(random_mses),
        ))
        stats['debug/MSE of reconstruction'] = ptu.get_numpy(
            recon_mse
        )[0]
        return stats
