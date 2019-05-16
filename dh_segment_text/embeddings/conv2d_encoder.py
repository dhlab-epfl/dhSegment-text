from .encoder import EmbeddingsEncoder
from .embeddings_utils import batch_resize_and_gather
import tensorflow as tf
import numpy as np

class Conv2dEncoder(EmbeddingsEncoder):
    def __init__(self, target_dim: int, starting_dim: int=256, max_conv: int=-1):
        self.target_dim = target_dim
        self.starting_dim = starting_dim
        max_power = int(np.round(np.log2(self.starting_dim)))
        min_power = int(np.floor(np.log2(self.target_dim)))
        if max_conv == -1:
            max_conv = (max_power-min_power)+1
        self.conv_sizes = np.logspace(min_power,max_power,max_conv, base=2).astype(int)[::-1][:-1]


    def __call__(self, embeddings: tf.Tensor, embeddings_map: tf.Tensor, target_shape: tf.Tensor) -> tf.Tensor:
        with tf.variable_scope("Conv2D_encoder"):
            embeddings_feature_map = batch_resize_and_gather(embeddings_map,
                                                             target_shape,
                                                             embeddings)
            with tf.variable_scope("Encoder"):
                if self.target_dim >= self.starting_dim:
                    raise IndexError(f"Target dim was bigger than {self.starting_dim}, got {self.target_dim}")
                reduced_embeddings = embeddings
                for i, conv_size in enumerate(self.conv_sizes):
                    reduced_embeddings = tf.contrib.layers.conv1d(reduced_embeddings, conv_size, (1), scope='conv_%01d'%i)
                reduced_embeddings = tf.contrib.layers.conv1d(reduced_embeddings, self.target_dim, (1), scope='conv_final')
            embeddings_feature_map = batch_resize_and_gather(embeddings_map,
                                                             target_shape,
                                                             reduced_embeddings)
            embeddings_feature_map.set_shape([None, None, None, self.target_dim])
        return embeddings_feature_map

