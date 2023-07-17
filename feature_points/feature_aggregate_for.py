import tensorflow as tf
import numpy as np
from tqdm import trange


def feature_aggregate(N_feature, M_feature, indices):
    N_feature = tf.squeeze(N_feature, axis=2)
    M_feature = tf.squeeze(M_feature, axis=2)
    batch_size = N_feature.shape[0]
    point_num = N_feature.shape[1]
    growth_rate = N_feature.shape[-1]
    expand_feature = tf.zeros_like(N_feature[0])
    expand_feature = tf.expand_dims(expand_feature, axis=0)
    for pc_num in range(batch_size):
        m_feature = M_feature[pc_num]
        m_ = indices[pc_num]
        order = tf.argsort(m_)
        tmp = tf.gather(m_feature, order)
        order_ = tf.sort(m_)
        order_ = tf.expand_dims(order_, axis=-1)
        shape = tf.constant([int(point_num), int(growth_rate)])
        features = tf.scatter_nd(order_, tmp, shape)
        features = tf.expand_dims(features, axis=0)
        expand_feature = tf.concat([expand_feature, features], axis=0)

    expand_feature = expand_feature[1:]
    aggregate_feature = N_feature + expand_feature
    aggregate_feature = tf.expand_dims(aggregate_feature, axis=-2)
    return aggregate_feature
