from os import path as osp
import numpy as np
import torch
from torch import optim
from torchvision.utils import save_image
from multiworld.core.image_env import normalize_image
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
            batch_size=128,
            log_interval=0,
            gamma=0.5,
            lr=1e-3,
            do_scatterplot=False,
            normalize=False,
            mse_weight=0.1,
            is_auto_encoder=False,
            background_subtract=False,
    ):
        self.quick_init(locals())
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.beta = model.beta
        self.gamma = gamma
        self.imsize = model.imsize
        self.do_scatterplot = do_scatterplot

        model.to(ptu.device)

        self.model = model
        self.representation_size = model.representation_size
        self.input_channels = model.input_channels
        self.imlength = model.imlength

        self.lr = lr
        params = list(self.model.parameters()) + self.model.lambdas
        self.optimizer = optim.Adam(params, lr=self.lr)
        self.train_dataset, self.test_dataset = train_dataset, test_dataset
        assert self.train_dataset.dtype == np.uint8
        assert self.test_dataset.dtype == np.uint8
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.batch_size = batch_size

        self.normalize = normalize
        self.mse_weight = mse_weight
        self.background_subtract = background_subtract

        if self.normalize or self.background_subtract:
            self.train_data_mean = np.mean(self.train_dataset, axis=0)
            self.train_data_mean = normalize_image(
                np.uint8(self.train_data_mean)
            )
        self.vae_logger_stats_for_rl = {}
        self._extra_stats_to_log = None

    def get_dataset_stats(self, data):
        torch_input = ptu.from_numpy(normalize_image(data))
        mus, log_vars = self.model.encode(torch_input)
        mus = ptu.get_numpy(mus)
        mean = np.mean(mus, axis=0)
        std = np.std(mus, axis=0)
        return mus, mean, std

    def _kl_np_to_np(self, np_imgs):
        torch_input = ptu.from_numpy(normalize_image(np_imgs))
        mu, log_var = self.model.encode(torch_input)
        return ptu.get_numpy(
            - torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        )

    def _reconstruction_squared_error_np_to_np(self, np_imgs):
        torch_input = ptu.from_numpy(normalize_image(np_imgs))
        recons, *_ = self.model(torch_input)
        error = torch_input - recons
        return ptu.get_numpy((error ** 2).sum(dim=1))

    def get_batch(self, train=True):
        dataset = self.train_dataset if train else self.test_dataset
        ind = np.random.randint(0, len(dataset), self.batch_size)
        samples = normalize_image(dataset[ind, :])
        if self.normalize:
            samples = ((samples - self.train_data_mean) + 1) / 2
        if self.background_subtract:
            samples = samples - self.train_data_mean
        return ptu.from_numpy(samples)

    def get_debug_batch(self, train=True):
        dataset = self.train_dataset if train else self.test_dataset
        X, Y = dataset
        ind = np.random.randint(0, Y.shape[0], self.batch_size)
        X = X[ind, :]
        Y = Y[ind, :]
        return ptu.from_numpy(X), ptu.from_numpy(Y)

    def train_epoch(self, epoch, sample_batch=None, batches=20, from_rl=False):
        self.model.train()
        losses = []
        log_probs = []
        kles = []
        mses = []
        for batch_idx in range(batches):
            if sample_batch is not None:
                data = sample_batch(self.batch_size)
                next_obs = data['next_obs']
            else:
                next_obs = self.get_batch()
            self.optimizer.zero_grad()
            x_hat, mask, loss, kle_loss, x_prob_loss, mse = self.model(next_obs, seedsteps=11)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([x for x in self.model.parameters()] + self.model.lambdas, 5.0)
            #torch.nn.utils.clip_grad_norm_(self.model.lambdas, 5.0)  # TODO Clip other gradients?
            self.optimizer.step()

            losses.append(loss.item())
            log_probs.append(x_prob_loss.item())
            kles.append(kle_loss.item())
            mses.append(mse.item())

            if self.log_interval and batch_idx % self.log_interval == 0:
                print(x_prob_loss.item(), kle_loss.item())


        if from_rl:
            self.vae_logger_stats_for_rl['Train VAE Epoch'] = epoch
            self.vae_logger_stats_for_rl['Train VAE Log Prob'] = np.mean(
                log_probs)
            self.vae_logger_stats_for_rl['Train VAE KL'] = np.mean(kles)
            self.vae_logger_stats_for_rl['Train VAE Loss'] = np.mean(losses)
        else:
            logger.record_tabular("train/epoch", epoch)
            logger.record_tabular("train/Log Prob", np.mean(log_probs))
            logger.record_tabular("train/KL", np.mean(kles))
            logger.record_tabular("train/loss", np.mean(losses))
            logger.record_tabular("train/mse", np.mean(mses))
            #logger.record_tabular('train/mask_loss', np.mean(m_losses))

    def test_epoch(
            self,
            epoch,
            save_reconstruction=True,
            save_vae=True,
            from_rl=False,
            record_stats=True,
            train=True,
            batches=1
    ):

        self.model.eval() #TODO Get around needing gradients during eval mode
        losses = []
        log_probs = []
        kles = []
        mses = []
        for batch_idx in range(batches):
            self.optimizer.zero_grad()
            next_obs = self.get_batch(train=train)
            T = next_obs.shape[1]
            x_hats, masks, loss, kle_loss, x_prob_loss, mse = self.model(next_obs, seedsteps=5)

            losses.append(loss.item())
            log_probs.append(x_prob_loss.item())
            kles.append(kle_loss.item())
            mses.append(mse.item())


            if batch_idx == 0 and save_reconstruction:
                ground_truth = next_obs[0].unsqueeze(0)
                K = self.model.K
                imsize = ground_truth.shape[-1]

                m = torch.stack([m[0] for m in masks]).permute(1, 0, 2, 3).unsqueeze(2).repeat(1, 1, 3, 1, 1) # K, T, 3, imsize, imsize
                x = torch.stack(x_hats)[:, :K].permute(1, 0, 2, 3, 4)
                rec = (m * x)
                full_rec = rec.sum(0, keepdim=True)
                comparison = torch.cat([ground_truth, full_rec, m, x], 0).view(-1, 3, imsize, imsize)

                save_dir = osp.join(logger.get_snapshot_dir(),
                                    '%s_r%d.png' % ('train' if train else 'val', epoch))
                save_image(comparison.data.cpu(), save_dir, nrow=T)



        if from_rl:
            self.vae_logger_stats_for_rl['Test VAE Epoch'] = epoch
            self.vae_logger_stats_for_rl['Test VAE Log Prob'] = np.mean(
                log_probs)
            self.vae_logger_stats_for_rl['Test VAE KL'] = np.mean(kles)
            self.vae_logger_stats_for_rl['Test VAE loss'] = np.mean(losses)
            self.vae_logger_stats_for_rl['VAE Beta'] = self.beta
        else:
            #for key, value in self.debug_statistics().items():
            #    logger.record_tabular(key, value)
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

    def dump_samples(self, epoch):
        self.model.eval()
        sample = ptu.randn(64, self.representation_size)
        sample = self.model.decode(sample)[0].cpu()
        save_dir = osp.join(logger.get_snapshot_dir(), 's%d.png' % epoch)
        save_image(
            sample.data.view(64, 3, self.imsize, self.imsize),
            save_dir
        )

    def _dump_imgs_and_reconstructions(self, idxs, filename):
        imgs = []
        recons = []
        for i in idxs:
            img_np = self.train_dataset[i]
            img_torch = ptu.from_numpy(normalize_image(img_np))
            recon, *_ = self.model(img_torch)

            img = img_torch.view(self.input_channels, self.imsize, self.imsize)
            rimg = recon.view(self.input_channels, self.imsize, self.imsize)
            imgs.append(img)
            recons.append(rimg)
        all_imgs = torch.stack(imgs + recons)
        save_file = osp.join(logger.get_snapshot_dir(), filename)
        save_image(
            all_imgs.data,
            save_file,
            nrow=4,
        )
