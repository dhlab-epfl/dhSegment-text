from .encoder import EmbeddingsEncoder
from .embeddings_utils import batch_resize_and_gather
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.slim import arg_scope
import numpy as np

class Conv1dEncoder(EmbeddingsEncoder):
    def __init__(self, target_dim: int, starting_dim: int=256, max_conv: int=-1, renorm=False, weight_decay: float=0.):
        self.target_dim = target_dim
        self.starting_dim = starting_dim
        max_power = int(np.round(np.log2(self.starting_dim)))
        min_power = int(np.floor(np.log2(self.target_dim)))
        if max_conv == -1:
            max_conv = (max_power-min_power)+1
        self.conv_sizes = np.logspace(min_power,max_power,max_conv, base=2).astype(int)[::-1][:-1]

        self.batch_norm_params = {
            "renorm": renorm,
            "renorm_clipping": {'rmax': 100, 'rmin': 0.1, 'dmax': 1},
            "renorm_momentum": 0.98
        }
        self.weight_decay = weight_decay



    def __call__(self, embeddings: tf.Tensor, embeddings_map: tf.Tensor, target_shape: tf.Tensor, is_training=False) -> tf.Tensor:

        batch_norm_fn = lambda x: tf.layers.batch_normalization(x, axis=-1, training=is_training,
                                                                name='batch_norm', **self.batch_norm_params)

        with tf.variable_scope("Conv1D_encoder"):
            with tf.variable_scope("Encoder"):
                with arg_scope([layers.conv1d],
                               normalizer_fn=batch_norm_fn,
                               weights_regularizer=layers.l2_regularizer(self.weight_decay)):
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
            embeddings_feature_map_first_dims = embeddings_feature_map[:,:,:,:3]
            embeddings_feature_map_first_dims = tf.div(
                tf.subtract(
                    embeddings_feature_map_first_dims, 
                    tf.reduce_min(embeddings_feature_map_first_dims)
                ), 
                tf.subtract(
                    tf.reduce_max(embeddings_feature_map_first_dims), 
                    tf.reduce_min(embeddings_feature_map_first_dims)
                )
            )
            tf.summary.image('summary/embeddings_encoded', embeddings_feature_map_first_dims, max_outputs=1)

        return embeddings_feature_map

