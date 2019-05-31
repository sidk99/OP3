"""
Copy this file to config.py and modify as needed.
"""

# Change these things
CODE_DIRS_TO_MOUNT = [
    #'/home/user/python/module/one',
    # '/home/rishiv/Research/fun_rlkit/'
    # '/nfs/kun1/users/rishiv/Research/fun_rlkit/'
    # TODO: Add baseline SAVP package as well
]
DIR_AND_MOUNT_POINT_MAPPINGS = [
    # dict(
    #     local_dir='/home/rishiv/.mujoco/',
    #     mount_point='/root/.mujoco',
    # ),
    dict(
        # local_dir='/home/rishiv/Research/fun_rlkit/',
       local_dir='/nfs/kun1/users/rishiv/Research/fun_rlkit/',
        mount_point='/home/ubuntu/Research/fun_rlkit/',
       # mount_point='/nfs/kun1/users/rishiv/Research/fun_rlkit/',
       filter_dir=['output', 'data'],
       pythonpath=True,
   )
]
LOCAL_LOG_DIR = '/nfs/kun1/users/rishiv/Research/op3_exps'
# LOCAL_LOG_DIR = '/home/rishiv/Research/op3_exps'

"""
********************************************************************************
********************************************************************************
********************************************************************************

You probably don't need to set all of the configurations below this line,
unless you use AWS, GCP, Slurm, and/or Slurm on a remote server. I recommend
ignoring most of these things and only using them on an as-needed basis.

********************************************************************************
********************************************************************************
********************************************************************************
"""
RUN_DOODAD_EXPERIMENT_SCRIPT_PATH = (
# '/home/rishiv/Research/fun_rlkit/scripts/run_experiment_from_doodad.py'
    '/nfs/kun1/users/rishiv/Research/fun_rlkit/scripts/run_experiment_from_doodad.py'
)

"""
AWS Settings
"""
# If not set, default will be chosen by doodad
AWS_S3_PATH = 's3://op3.rlkit.data/'

# You probably don't need to change things below
# Specifically, the docker image is looked up on dockerhub.com.
DOODAD_DOCKER_IMAGE = "'rishiv/rv_ray_docker'" #"'jcoreyes/op3-ray'"
INSTANCE_TYPE = 'c1.medium' #
SPOT_PRICE = 0.035

GPU_DOODAD_DOCKER_IMAGE = "'rishiv/rv_ray_docker'" #'jcoreyes/op3-ray'
GPU_INSTANCE_TYPE = 'g3.8xlarge' #g2.2xlarge, p3.8xlarge
GPU_SPOT_PRICE = 0.75
# These AMI images have the docker images already installed.
REGION_TO_GPU_AWS_IMAGE_ID = {
    'us-west-2': 'ami-076347b8649dddb00'
}

REGION_TO_GPU_AWS_AVAIL_ZONE = {
    'us-west-2': 'us-west-2b'
}

# This really shouldn't matter and in theory could be whatever
OUTPUT_DIR_FOR_DOODAD_TARGET = '/tmp/doodad-output/'


"""
Slurm Settings
"""
SINGULARITY_IMAGE = '/home/PATH/TO/IMAGE.img'
SINGULARITY_PRE_CMDS = [
    'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro150/bin'
]
SLURM_CPU_CONFIG = dict(
    account_name='TODO',
    partition='savio',
    nodes=1,
    n_tasks=1,
    n_gpus=1,
)
SLURM_GPU_CONFIG = dict(
    account_name='TODO',
    partition='savio2_1080ti',
    nodes=1,
    n_tasks=1,
    n_gpus=1,
)


"""
Slurm Script Settings

These are basically the same settings as above, but for the remote machine
where you will be running the generated script.
"""
SSS_CODE_DIRS_TO_MOUNT = [
]
SSS_DIR_AND_MOUNT_POINT_MAPPINGS = [
    dict(
        local_dir='/global/home/users/USERNAME/.mujoco',
        mount_point='/root/.mujoco',
    ),
]
SSS_LOG_DIR = '/global/scratch/USERNAME/doodad-log'

SSS_IMAGE = '/global/scratch/USERNAME/TODO.img'
SSS_RUN_DOODAD_EXPERIMENT_SCRIPT_PATH = (
    '/global/home/users/USERNAME/path/to/rlkit/scripts'
    '/run_experiment_from_doodad.py'
)
SSS_PRE_CMDS = [
    'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/home/users/USERNAME'
    '/.mujoco/mjpro150/bin'
]

"""
GCP Settings
"""
GCP_IMAGE_NAME = 'TODO'
GCP_GPU_IMAGE_NAME = 'TODO'
GCP_BUCKET_NAME = 'TODO'

GCP_DEFAULT_KWARGS = dict(
    zone='us-west2-c',
    instance_type='n1-standard-4',
    image_project='TODO',
    terminate=True,
    preemptible=True,
    gpu_kwargs=dict(
        gpu_model='nvidia-tesla-p4',
        num_gpu=1,
    )
)