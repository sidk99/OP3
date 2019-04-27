import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import run_experiment
import numpy as np
import pickle
import torch
from argparse import ArgumentParser
import imageio
from rlkit.envs.blocks.mujoco.block_stacking_env import BlockEnv

class MPC:

    def __init__(self, model, env, n_actions, mpc_steps):
        self.model = model
        self.env = env
        self.n_actions = n_actions
        self.mpc_steps = mpc_steps

    def run(self, goal_image):
        goal_image_tensor = ptu.from_numpy(np.moveaxis(goal_image, 2, 0)).unsqueeze(0) # (1, 3, imsize, imsize)
        rec_goal_image, goal_latents = self.model.refine(goal_image_tensor, hidden_state=None)  # (K, rep_size)

        obs = self.env.reset()
        for mpc_step in range(self.mpc_steps):
            action, goal_idx = self.step_mpc(obs, goal_latents)
            obs = self.env.step(action)
            if goal_latents.shape[0] == 1:
                break
            # remove matching goal latent from goal latents
            goal_latents = torch.stack([goal_latents[i] for i in set(range(goal_latents.shape[0])) - set([goal_idx])])


    def cost_func(self, l1, l2):
        # l1 is (..., rep_size) l2 is (..., rep_size)
        return torch.pow(l1 - l2, 2).mean(-1)

    def model_step_batched(self, obs, actions, bs=8):
        # Handle large obs in batches
        n_batches = obs.shape[0] // bs
        outputs = [[], []]
        for i in range(n_batches):
            start_idx = i * bs
            end_idx = min(start_idx + bs, obs.shape[0])

            obs_latents, pred_obs = self.model.step(obs[start_idx:end_idx], actions[start_idx:end_idx])
            outputs[0].append(obs_latents)
            outputs[1].append(pred_obs)

        return torch.stack(outputs[0]), torch.cat(outputs[1])

    def step_mpc(self, obs, goal_latents):
        # obs is
        # goal latents is (<K, rep_size)
        actions = np.stack([self.env.sample_action() for _ in range(self.n_actions)])

        obs_rep = ptu.from_numpy(np.moveaxis(obs, 2, 0)).unsqueeze(0).repeat(self.n_actions, 1, 1, 1)
        pred_obs, obs_latents = self.model_step_batched(obs_rep,
                                                   ptu.from_numpy(actions))
        # obs_latents is (n_actions, K, rep_size)
        best_goal_idx = 0
        best_action_idx = 0
        best_cost = 0
        best_latent_idx = 0

        # Compare against each goal latent
        for i in range(goal_latents.shape[0]):

            cost = self.cost_func(goal_latents[i].view(1, 1, -1), obs_latents) # cost is (n_actions, K)
            cost, latent_idx = cost.min(-1) # take min among K

            min_cost, action_idx = cost.min(0) # take min among n_actions

            if min_cost <= best_cost:
                best_goal_idx = i
                best_action_idx = action_idx
                best_cost = min_cost
                best_latent_idx = latent_idx[action_idx]

        best_action = actions[best_action_idx]

        return best_action, best_goal_idx


def main(variant):

    model_file = variant['model_file']
    goal_file = variant['goal_file']

    #model_file = '/home/jcoreyes/objects/rlkit/output/04-25-iodine-blocks-physics-actions/04-25-iodine-blocks-physics-actions_2019_04_25_11_36_24_0000--s-98913/params.pkl'
    #goal_file = '/home/jcoreyes/objects/object-oriented-prediction/o2p2/planning/executed/mjc_4.png'

    model = pickle.load(open(model_file, 'rb'))
    #model.cuda()

    env = BlockEnv(5)
    mpc = MPC(model, env, n_actions=16, mpc_steps=5)


    goal_image = imageio.imread(goal_file)
    mpc.run(goal_image)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-f', '--modelfile', type=str, default=None)
    parser.add_argument('-g', '--goalfile', type=str, default=None)
    args = parser.parse_args()

    variant = dict(
        algorithm='MPC',
        modelfile=args.modelfile,
        goalfile=args.goalfile,
    )


    run_experiment(
        main,
        exp_prefix='MPC',
        mode='here_no_doodad',
        variant=variant,
        use_gpu=True,  # Turn on if you have a GPU
    )



