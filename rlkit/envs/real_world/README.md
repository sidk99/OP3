# Poke Dataset
Data in `poke`

Saves hdf5 file in `poke/poke.h5`

Saves gifs in `poke/imgs`

Command: `python real_world_env.py --root poke`

    {'test': 795, 'train': 10872}
    {'num_frames': 20, 'action_dim': 5, 'image_res': 64}


# Kevin Dataset
Data in `kevin`

Saves hdf5 file in `kevin/kevin.h5`

Saves gifs in `kevin/imgs`

Command `CUDA_VISIBLE_DEVICES=0 python real_world_env.py --root kevin`

    {'train': 1500, 'test': 0, 'val': 166}
    {'num_frames': 15, 'image_res': 64, 'action_dim': 2}

# VisInt-solid
Data in `solid`

Saves hdf5 file in `solid/solid.h5`

Saves gifs in `solid/imgs`

Command `CUDA_VISIBLE_DEVICES=0 python real_world_env.py --root solid`

    {'train': 7143, 'val': 414, 'test': 374}
    {'image_res': 64, 'action_dim': 4, 'num_frames': 30}

# VisInt-cloth
