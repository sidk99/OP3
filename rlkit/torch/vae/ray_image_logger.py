import ray.tune.logger as logger
from torchvision.utils import save_image
class ImageLogger(logger.Logger):
    def on_result(self, result):

        epoch = result['train/epoch']
        train_image = result['train_image']

        #save_dir =
        import pdb; pdb.set_trace()