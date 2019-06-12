"""
Each item in this list specifies a file_mount.

local_dir: directory/file on the local machine.
remote_dir: directory/file name to copy to on the remote instance.
mount_point: where the directory/file will be mounted in the docker instance.
"""
# DIR_AND_MOUNT_POINT_MAPPINGS = [
#     dict(
#         local_dir='/home/jcoreyes/.mujoco',
#         remote_dir='/home/ubuntu/.mujoco',
#         mount_point='/root/.mujoco',
#     ),
#     dict(
#         local_dir='/home/jcoreyes/objects/rlkit',
#         remote_dir='/home/ubuntu/objects/rlkit',
#         mount_point='/home/jcoreyes/objects/rlkit',
#     ),
#
#     dict(
#         local_dir='/home/jcoreyes/.aws',
#         remote_dir='/home/ubuntu/.aws',
#         mount_point='/home/ubuntu/.aws',
#     ),
#     dict(
#         local_dir='/tmp/local_exp.pkl',
#         remote_dir='/home/ubuntu/local_exp.pkl',
#         mount_point='/tmp/local_exp.pkl',
#     ),
# ]

# dict(
#         # local_dir='/home/rishiv/Research/fun_rlkit/',
#        local_dir='/nfs/kun1/users/rishiv/Research/fun_rlkit/',
#         mount_point='/home/ubuntu/Research/fun_rlkit/',
#        # mount_point='/nfs/kun1/users/rishiv/Research/fun_rlkit/',
#        filter_dir=['output', 'data'],
#        pythonpath=True,
#    )

DIR_AND_MOUNT_POINT_MAPPINGS = [
    # dict(
    #     local_dir='/home/rishiv/.mujoco',
    #     remote_dir='/home/ubuntu/.mujoco',
    #     mount_point='/root/.mujoco',
    # ),
    dict(
        local_dir='/home/rishiv/Research/fun_rlkit/',
        remote_dir='/home/ubuntu/Research/fun_rlkit/', #EC2 server
        mount_point='/home/rishiv/Research/fun_rlkit/', #Docker
        filter_dir=['output', 'data']
    ),

    dict(
        local_dir='/home/rishiv/.aws',
        remote_dir='/home/ubuntu/.aws',
        mount_point='/home/ubuntu/.aws',
    ),
    dict(
        local_dir='/tmp/local_exp.pkl',
        remote_dir='/home/ubuntu/local_exp.pkl',
        mount_point='/tmp/local_exp.pkl',
    ),
]


# This can basically be anything. Used for launching on instances. The
# local launch parameters (exo_func, exp_variant, etc) are saved at this location
# on the local machine and then transfered to the remote machine.
EXPERIMENT_INFO_PKL_FILEPATH = '/tmp/local_exp.pkl'
# Again, can be anything. The Ray autoscaler yaml file is saved to this location
# before launching.
LAUNCH_FILEPATH = '/tmp/autoscaler_launch.yaml'

# TODO:Steven remove this. File syncing from s3 still uses this.
LOCAL_LOG_DIR = '/home/rishiv/Research/fun_rlkit/s3_logs'

AWS_CONFIG_NO_GPU=dict(
    REGION='us-west-2',
    INSTANCE_TYPE = 'c5.xlarge',
    SPOT_PRICE = 0.1,
    REGION_TO_AWS_IMAGE_ID = {
        'us-west-2': 'ami-01a4e5be5f289dd12'
    },
    REGION_TO_AWS_AVAIL_ZONE = {
        'us-west-2': 'us-west-2a,us-west-2b'
    },

)

gpu_instance_to_price = {
    'g3.16xlarge': 1.4, #4 GPU, 32 GB, 1 limit
    'p2.8xlarge': 2.3, #8 GPU, 96 GB, 0 limit
    'p2.16xlarge': 4.5, #16 GPU, 192 GB, 0 limit
    'p3.8xlarge': 3.8, #4 GPU, 64 GB, 5 limit
    'p3.16xlarge': 7.5, #8 GPU, 128 GB, 0 limit
    'p3dn.24xlarge': 9.5, #8 GPU, 256, 0 limit
}
which_gpu = 'p3.8xlarge' #g3.16xlarge

AWS_CONFIG_GPU = dict(
    REGION='us-west-2',
    INSTANCE_TYPE = which_gpu,
    SPOT_PRICE = gpu_instance_to_price[which_gpu],
    REGION_TO_AWS_IMAGE_ID = {
        'us-west-2': 'ami-076347b8649dddb00'
    },
    REGION_TO_AWS_AVAIL_ZONE = {
        'us-west-2': 'us-west-2a'
        # 'us-west-2': 'us-west-2a,us-west-2b'
    },
)

GCP_CONFIG_GPU = dict(
    REGION='us-west1',
    INSTANCE_TYPE='n1-highmem-8',
    SOURCE_IMAGE='https://www.googleapis.com/compute/v1/projects/railrl-private-gcp/global/images/railrl-private-ray',
    PROJECT_ID='railrl-private-gcp',
    gpu_kwargs=dict(
        num_gpu=1,
    ),
    REGION_TO_GCP_AVAIL_ZONE = {
        'us-west1': "us-west1-b",
    },

)


AWS_CONFIG = {
    True: AWS_CONFIG_GPU,
    False: AWS_CONFIG_NO_GPU,
}
GCP_CONFIG = {
    True: GCP_CONFIG_GPU,
    False: GCP_CONFIG_GPU,
}

# GPU_DOCKER_IMAGE = 'stevenlin598/ray_railrl'
# GPU_DOCKER_IMAGE = 'jcoreyes/op3-ray'
GPU_DOCKER_IMAGE = 'rishiv/rv_ray_docker'
"""
DOCKER_IMAGE['use_gpu']
I think for our case, the GPU docker image can be used as the non-gpu image as well.
"""
DOCKER_IMAGE = {
    True: GPU_DOCKER_IMAGE,
    False: GPU_DOCKER_IMAGE
}
LOG_BUCKET = 's3://op3.rlkit.data'

try:
    from rlkit.launchers.ray_config_personal import *
except ImportError:
    print("No personal ray_config.py found")
