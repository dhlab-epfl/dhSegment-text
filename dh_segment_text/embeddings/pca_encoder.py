from .encoder import EmbeddingsEncoder
from .embeddings_utils import batch_resize_and_gather
import tensorflow as tf
import numpy as np

class PCAEncoder(EmbeddingsEncoder):
    def __init__(self, pca_mean_path: str, pca_components_path: str, target_dim: int):
        self.pca_mean = tf.constant(np.load(pca_mean_path), dtype=tf.float32)
        self.pca_components = tf.constant(np.load(pca_components_path), dtype=tf.float32)
        self.target_dim = target_dim

    def __call__(self, embeddings: tf.Tensor, embeddings_map: tf.Tensor, target_shape: tf.Tensor, is_training: bool=False) -> tf.Tensor:
        with tf.variable_scope("PCAEncoder"):
            reduced_components = tf.transpose(self.pca_components[:self.target_dim])
            reduced_embeddings = tf.einsum('aij,jk->aik', (embeddings-self.pca_mean), reduced_components)
            embeddings_feature_map = batch_resize_and_gather(embeddings_map,
                                                             target_shape,
                                                             reduced_embeddings)
            embeddings_feature_map.set_shape([None, None, None, self.target_dim])
        return embeddings_feature_map

