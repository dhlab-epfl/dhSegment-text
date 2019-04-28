import tensorflow as tf

def batch_resize_maps(batch_maps: tf.Tensor, target_shape: tf.Tensor, scope: str="ResizeMaps"):
    with tf.variable_scope(scope):
        maps_resized = tf.squeeze(
            tf.image.resize_nearest_neighbor(
                tf.expand_dims(batch_maps, axis=-1),
                target_shape
            ), axis=-1)
    return maps_resized

def batch_gather(batch_maps: tf.Tensor, embeddings: tf.Tensor, scope: str="BatchGather"):
    with tf.variable_scope(scope):
        b = tf.shape(batch_maps)[0]
        x = tf.shape(batch_maps)[1]
        y = tf.shape(batch_maps)[2]
        batches_range = tf.expand_dims(tf.expand_dims(tf.range(b), axis=-1), axis=-1)
        batch_indices = tf.tile(batches_range, (1, x, y))
        batch_maps_indices = tf.stack([batch_indices, batch_maps], axis=-1)
        embeddings_feature_map = tf.gather_nd(embeddings, batch_maps_indices)
    return embeddings_feature_map

def batch_resize_and_gather(batch_maps: tf.Tensor,
                            target_shape: tf.Tensor,
                            embeddings: tf.Tensor,
                            scope: str="BatchResizeGather"):
    with tf.variable_scope(scope):
        batch_maps_resized = batch_resize_maps(batch_maps, target_shape)
        embeddings_feature_map = batch_gather(batch_maps_resized, embeddings)
    return embeddings_feature_map
