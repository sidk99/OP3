from collections import OrderedDict
from os import path as osp
import numpy as np
import torch
from rlkit.torch.iodine.iodine import create_schedule
from torch import optim
from torchvision.utils import save_image
from rlkit.core import logger
from rlkit.core.serializable import Serializable
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.pytorch_util import from_numpy
import os

class IodineTrainer(Serializable):
    def __init__(
            self,
            train_dataset,
            test_dataset,
            model,
            train_T=5,
            test_T=5,
            max_T=None,
            seed_steps=4,
            schedule_type='single_step_physics',
            batch_size=128,
            log_interval=0,
            lr=1e-3,
    ):
        self.quick_init(locals())
        self.log_interval = log_interval
        self.batch_size = batch_size

        self.seed_steps = seed_steps
        self.train_T = train_T
        self.test_T = test_T
        self.max_T = max_T
        self.seed_steps = seed_steps
        self.schedule_type = schedule_type

        model.to(ptu.device)

        self.model = model

        self.lr = lr
        params = list(self.model.parameters())
        self.optimizer = optim.Adam(params, lr=self.lr)
        self.train_dataset, self.test_dataset = train_dataset, test_dataset

        self.batch_size = batch_size

        self.vae_logger_stats_for_rl = {}
        self._extra_stats_to_log = None

    def prepare_tensors(self, tensors):
        imgs = tensors[0].to(ptu.device) / 255. #Normalize image to 0-1
        if len(tensors) == 2:
            return imgs, tensors[1].to(ptu.device)
        else:
            return imgs, None #Action is none

    def get_schedule_type(self, epoch):
        if 'curriculum' in self.schedule_type:
            rollout_len = epoch // 20 + 1
            if epoch % 20 == 0:
                torch.save(self.state_dict(), open(logger.get_snapshot_dir() + '/params.pkl', "wb"))
            return 'curriculum_{}'.format(rollout_len)
        else:
            return self.schedule_type


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
            schedule_type = self.get_schedule_type(epoch)
            schedule = create_schedule(True, self.train_T, schedule_type, self.seed_steps, self.max_T)
            # print("obs: {}".format(obs.shape))
            x_hat, mask, loss, kle_loss, x_prob_loss, mse, final_recon, lambdas = self.model(input=obs, actions=actions, schedule=schedule)
            # if (loss.mean().item() > 1e8):
            #     print("MASSIVE LOSS!")
            #     print(loss, schedule, kle_loss, mse)
            #     continue
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_([x for x in self.model.parameters()], 5.0)
            self.optimizer.step()

            losses.append(loss.mean().item())
            log_probs.append(x_prob_loss.mean().item())
            kles.append(kle_loss.mean().item())
            mses.append(mse.mean().item())

            if self.log_interval and batch_idx % self.log_interval == 0:
                print(x_prob_loss.item(), kle_loss.item())

        stats = OrderedDict([
            ("train/epoch", epoch),
            ("train/Log Prob", np.mean(log_probs)),
            ("train/KL", np.mean(kles)),
            ("train/loss", np.mean(losses)),
            ("train/mse", np.mean(mses))
        ])

        return stats


    def test_epoch(
            self,
            epoch,
            save_reconstruction=True,
            train=True,
            batches=1,
    ):

        schedule_type = self.get_schedule_type(epoch)
        schedule = create_schedule(False, self.test_T, schedule_type, self.seed_steps, self.max_T)

        self.model.eval()
        losses, log_probs, kles, mses = [], [], [], []
        dataloader = self.train_dataset.dataloader if train else self.test_dataset.dataloader
        for batch_idx, tensors in enumerate(dataloader):
            obs, actions = self.prepare_tensors(tensors)
            x_hats, masks, loss, kle_loss, x_prob_loss, mse, final_recon, lambdas = self.model(obs, actions=actions, schedule=schedule)

            losses.append(loss.mean().item())
            log_probs.append(x_prob_loss.mean().item())
            kles.append(kle_loss.mean().item())
            mses.append(mse.mean().item())


            if batch_idx == 0 and save_reconstruction:
                t_sample = np.cumsum(schedule)
                t_sample[t_sample >= obs.shape[1]] = obs.shape[1] - 1
                ground_truth = obs[0][t_sample].unsqueeze(0)
                imsize = ground_truth.shape[-1]

                m = masks[0].permute(1, 0, 2, 3, 4).repeat(1, 1, 3, 1, 1) # (K, T, ch, imsize, imsize)
                x = x_hats[0].permute(1, 0, 2, 3, 4)
                rec = (m * x)
                full_rec = rec.sum(0, keepdim=True)

                comparison = torch.cat([ground_truth, full_rec, m, rec], 0).view(-1, 3, imsize, imsize)
                # import pdb; pdb.set_trace()
                save_dir = osp.join(logger.get_snapshot_dir(),
                                    '%s_r%d.png' % ('train' if train else 'val', epoch))

                # save_image(comparison.data.cpu(), save_dir, nrow=self.test_T)
                save_image(comparison.data.cpu(), save_dir, nrow=len(schedule))
            if batch_idx >= batches - 1:
                break


        stats = OrderedDict([
            ("test/Log Prob", np.mean(log_probs)),
            ("test/KL", np.mean(kles)),
            ("test/loss", np.mean(losses)),
            ("test/mse", np.mean(mses))
        ])

        return stats

