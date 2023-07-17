import tensorflow as tf
import numpy as np


def sieve(x, A, FP_rate=0.25):
    A = tf.squeeze(A, axis=-1)
    batch_size = A.shape[0]
    point_num = A.shape[1]
    FP_num = int(int(point_num) * FP_rate)
    for pc_num in range(batch_size):
        pc = x[pc_num]
        a = A[pc_num]
        indices = tf.nn.top_k(a, k=FP_num).indices
        feature_points = tf.gather(pc, indices, axis=0)
        feature_points = tf.expand_dims(feature_points, axis=0)
        feature_indices = tf.expand_dims(indices, axis=0)
        if pc_num == 0:
            batch_feature_points = feature_points
            batch_feature_indices = feature_indices
        else:
            batch_feature_points = tf.concat([batch_feature_points, feature_points], axis=0)
            batch_feature_indices = tf.concat([batch_feature_indices, feature_indices], axis=0)
    return batch_feature_points, batch_feature_indices
