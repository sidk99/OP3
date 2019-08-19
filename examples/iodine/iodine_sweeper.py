from doodad.easy_sweep import DoodadSweeper
import doodad.mount as mount

MOUNTS = [
    mount.MountLocal(local_dir='/home/rishiv/Research/fun_rlkit/', pythonpath=True) # Code project folder
]

SWEEPER = DoodadSweeper(mounts=MOUNTS,
                        local_output_dir='/home/rishiv/Research/op3_exps',
)


def example_function(param1=0, param2='c'):
    print(param1, param2)

sweep_params = {
    'param1': [0,1,2],
    'param2': ['a', 'b'],
}

SWEEPER.run_sweep_serial(example_function, sweep_params, repeat=1)