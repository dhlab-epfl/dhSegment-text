from .encoder import EmbeddingsEncoder
import tensorflow as tf
import numpy as np

class PCAEncoder(EmbeddingsEncoder):
    def __init__(self, pca_mean_path: str, pca_components_path: str, target_dim: int):
        self.pca_mean = tf.constant(np.load(pca_mean_path), dtype=tf.float32)
        self.pca_components = tf.constant(np.load(pca_components_path), dtype=tf.float32)
        self.target_dim = target_dim

    def __call__(self, embeddings: tf.Tensor, embeddings_map: tf.Tensor, target_shape: tf.Tensor) -> tf.Tensor:
        with tf.variable_scope("PCAEncoder"):
            reduced_components = tf.transpose(self.pca_components[:self.target_dim])
            reduced_embeddings = tf.einsum('aij,jk->aik', (embeddings-self.pca_mean), reduced_components)
            embeddings_map_reduced = tf.squeeze(
                tf.image.resize_nearest_neighbor(
                    tf.expand_dims(embeddings_map, axis=-1),
                    target_shape
                ), axis=-1)
            with tf.variable_scope("BatchGather"):
                b = tf.shape(embeddings_map_reduced)[0]
                x = tf.shape(embeddings_map_reduced)[1]
                y = tf.shape(embeddings_map_reduced)[2]
                batches_range = tf.expand_dims(tf.expand_dims(tf.range(b), axis=-1), axis=-1)
                batch_indices = tf.tile(batches_range, (1, x, y))
                embeddings_map_indices = tf.stack([batch_indices, embeddings_map_reduced], axis=-1)
                embeddings_feature_map = tf.gather_nd(reduced_embeddings, embeddings_map_indices)
                embeddings_feature_map.set_shape([None, None, None, self.target_dim])
        return embeddings_feature_map

