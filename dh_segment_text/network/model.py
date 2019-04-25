#!/usr/bin/env python

import tensorflow as tf
from abc import ABC, abstractmethod
from typing import List, Union, Tuple, Optional, Dict


class Encoder(ABC):
    @abstractmethod
    def __call__(self, images: tf.Tensor, is_training=False) -> List[tf.Tensor]:
        """

        :param images: [NxHxWx3] float32 [0..255] input images
        :return: a list of the feature maps in decreasing spatial resolution (first element is most likely the input \
        image itself, then the output of the first pooling op, etc...)
        """
        pass

    def pretrained_information(self) -> Tuple[Optional[str], Union[None, List, Dict]]:
        """

        :return: The filename of the pretrained checkpoint and the corresponding variables (List of Dict mapping) \
        or `None` if no-pretraining is done
        """
        return None, None


class Decoder(ABC):
    @abstractmethod
    def __call__(self, feature_maps: List[tf.Tensor], num_classes: int, is_training=False) -> tf.Tensor:
        """

        :param feature_maps: list of feature maps, in decreasing spatial resolution, first one being at the original \
        resolution
        :return: [N,H,W,num_classes] float32 tensor of logit scores
        """
        pass
