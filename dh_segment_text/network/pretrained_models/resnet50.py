from tensorflow.contrib import slim, layers
import tensorflow as tf
from tensorflow.contrib.slim import nets
from ..model import Encoder
import os
import tarfile
from ...utils.misc import get_data_folder, download_file
from .vgg16 import mean_substraction


class ResnetV1_50(Encoder):
    """ResNet-50 implementation

    :ivar train_batchnorm: Option to use batch norm
    :ivar blocks: number of blocks (resnet blocks)
    :ivar weight_decay: value of weight decay
    :ivar batch_renorm: Option to use batch renorm
    :ivar corrected_version: option to use the original resnet implementation (True) but less efficient than \
    `slim`'s implementation
    :ivar pretrained_file: path to the file (.ckpt) containing the pretrained weights
    """
    def __init__(self, train_batchnorm: bool=False, blocks: int=4, weight_decay: float=0.0001,
                 batch_renorm: bool=True, corrected_version: bool=False):
        self.train_batchnorm = train_batchnorm
        self.blocks = blocks
        self.weight_decay = weight_decay
        self.batch_renorm = batch_renorm
        self.corrected_version = corrected_version
        self.pretrained_file = os.path.join(get_data_folder(), 'resnet_v1_50.ckpt')
        if not os.path.exists(self.pretrained_file):
            print("Could not find pre-trained file {}, downloading it!".format(self.pretrained_file))
            tar_filename = os.path.join(get_data_folder(), 'resnet_v1_50.tar.gz')
            download_file('http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz', tar_filename)
            tar = tarfile.open(tar_filename)
            tar.extractall(path=get_data_folder())
            tar.close()
            os.remove(tar_filename)
            assert os.path.exists(self.pretrained_file)
            print('Pre-trained weights downloaded!')

    def pretrained_information(self):
        return self.pretrained_file, [v for v in tf.global_variables()
                                      if 'resnet_v1_50' in v.name
                                      and 'renorm' not in v.name]

    def __call__(self, images: tf.Tensor, is_training=False):
        outputs = []

        with slim.arg_scope(nets.resnet_v1.resnet_arg_scope(weight_decay=self.weight_decay, batch_norm_decay=0.999)), \
             slim.arg_scope([layers.batch_norm], renorm_decay=0.95, renorm=self.batch_renorm):
            mean_substracted_tensor = mean_substraction(images)
            assert 0 < self.blocks <= 4

            if self.corrected_version:
                def corrected_resnet_v1_block(scope: str, base_depth: int, num_units: int, stride: int) -> tf.Tensor:
                    """
                    Helper function for creating a resnet_v1 bottleneck block.

                    :param scope: The scope of the block.
                    :param base_depth: The depth of the bottleneck layer for each unit.
                    :param num_units: The number of units in the block.
                    :param stride: The stride of the block, implemented as a stride in the last unit.
                                   All other units have stride=1.
                    :return: A resnet_v1 bottleneck block.
                    """
                    return nets.resnet_utils.Block(scope, nets.resnet_v1.bottleneck, [{
                        'depth': base_depth * 4,
                        'depth_bottleneck': base_depth,
                        'stride': stride
                    }] + [{
                        'depth': base_depth * 4,
                        'depth_bottleneck': base_depth,
                        'stride': 1
                    }] * (num_units - 1))

                blocks_list = [
                    corrected_resnet_v1_block('block1', base_depth=64, num_units=3, stride=1),
                    corrected_resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                    corrected_resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
                    corrected_resnet_v1_block('block4', base_depth=512, num_units=3, stride=2),
                ]
                desired_endpoints = [
                    'resnet_v1_50/conv1',
                    'resnet_v1_50/block1/unit_3/bottleneck_v1',
                    'resnet_v1_50/block2/unit_4/bottleneck_v1',
                    'resnet_v1_50/block3/unit_6/bottleneck_v1',
                    'resnet_v1_50/block4/unit_3/bottleneck_v1'
                ]
            else:
                blocks_list = [
                    nets.resnet_v1.resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                    nets.resnet_v1.resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                    nets.resnet_v1.resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
                    nets.resnet_v1.resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
                ]
                desired_endpoints = [
                    'resnet_v1_50/conv1',
                    'resnet_v1_50/block1/unit_2/bottleneck_v1',
                    'resnet_v1_50/block2/unit_3/bottleneck_v1',
                    'resnet_v1_50/block3/unit_5/bottleneck_v1',
                    'resnet_v1_50/block4/unit_3/bottleneck_v1'
                ]

            net, endpoints = nets.resnet_v1.resnet_v1(mean_substracted_tensor,
                                                      blocks=blocks_list[:self.blocks],
                                                      num_classes=None,
                                                      is_training=self.train_batchnorm and is_training,
                                                      global_pool=False,
                                                      output_stride=None,
                                                      include_root_block=True,
                                                      reuse=None,
                                                      scope='resnet_v1_50')

            # Add standardized original images
            outputs.append(mean_substracted_tensor/127.0)

            for d in desired_endpoints[:self.blocks + 1]:
                outputs.append(endpoints[d])

            return outputs
