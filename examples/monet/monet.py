from rlkit.torch.monet.monet import MonetVAE
from rlkit.torch.conv_networks import BroadcastCNN
import rlkit.torch.monet.monet as monet
from rlkit.torch.monet.unet import UNet
from rlkit.torch.monet.monet_trainer import MonetTrainer
import rlkit.torch.pytorch_util as ptu
from rlkit.pythonplusplus import identity
from rlkit.launchers.launcher_util import run_experiment
from rlkit.core import logger
import numpy as np
import h5py

def load_dataset(data_path, train=True):
    hdf5_file = h5py.File(data_path, 'r')  # RV: Data file
    if 'clevr' in data_path:
        return np.array(hdf5_file['features'])
    else:
        if train:
            feats = np.array(hdf5_file['training']['features'])
        else:
            feats = np.array(hdf5_file['test']['features'])
        data = feats.reshape((-1, 64, 64, 3))
        data = (data * 255).astype(np.uint8)
        data = np.swapaxes(data, 1, 3)
        return data


def train_vae(variant):
    #train_path = '/home/jcoreyes/objects/rlkit/examples/monet/clevr_train_10000.hdf5'
    #test_path = '/home/jcoreyes/objects/rlkit/examples/monet/clevr_test.hdf5'

    train_path = '/home/jcoreyes/objects/RailResearch/DataGeneration/ColorTwoBallSmall.h5'
    test_path = '/home/jcoreyes/objects/RailResearch/DataGeneration/ColorTwoBallSmall.h5'

    train_data = load_dataset(train_path, train=True)
    test_data = load_dataset(test_path, train=False)

    train_data = train_data.reshape((train_data.shape[0], -1))
    test_data = test_data.reshape((test_data.shape[0], -1))
    #logger.save_extra_data(info)
    logger.get_snapshot_dir()
    variant['vae_kwargs']['architecture'] = monet.imsize64_monet_architecture #monet.imsize84_monet_architecture
    variant['vae_kwargs']['decoder_output_activation'] = identity
    variant['vae_kwargs']['decoder_class'] = BroadcastCNN

    attention_net = UNet(in_channels=4, n_classes=1, up_mode='upsample', depth=3,
                         padding=True)
    m = MonetVAE(
        **variant['vae_kwargs'],
        attention_net=attention_net
    )

    m.to(ptu.device)
    t = MonetTrainer(train_data, test_data, m,
                       **variant['algo_kwargs'])
    save_period = variant['save_period']
    for epoch in range(variant['num_epochs']):
        should_save_imgs = (epoch % save_period == 0)
        t.train_epoch(epoch)
        t.test_epoch(
            epoch,
            save_reconstruction=should_save_imgs,
        )
        if should_save_imgs:
            t.dump_samples(epoch)
    logger.save_extra_data(m, 'vae.pkl', mode='pickle')


if __name__ == "__main__":
    variant = dict(
        vae_kwargs = dict(
            imsize=64,
            representation_size=16,
            input_channels=4,
            decoder_distribution='gaussian_identity_variance'
        ),
        algo_kwargs = dict(
            beta=0.5,
            gamma=0.5,
            batch_size=16,
            lr=1e-4,
            log_interval=0,
        ),
        num_epochs=1500,
        algorithm='VAE',
        save_period=5,
    )


    run_experiment(
        train_vae,
        exp_prefix='vae-clevr',
        mode='here_no_doodad',
        variant=variant,
        use_gpu=True,  # Turn on if you have a GPU
    )



