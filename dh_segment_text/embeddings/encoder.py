import tensorflow as tf
from abc import ABC, abstractmethod
from typing import List

class EmbeddingsEncoder(ABC):
    @abstractmethod
    def __call__(self, embeddings: tf.Tensor, embeddings_map: tf.Tensor, target_shape: tf.Tensor, target_dim: int) -> tf.Tensor:
        pass
