import tensorflow as tf
import numpy as np
from tqdm import trange


def feature_aggregate(N_feature, M_feature, indices):
    N_feature = tf.squeeze(N_feature, axis=2)
    M_feature = tf.squeeze(M_feature, axis=2)
    batch_size = N_feature.shape[0]
    point_num = N_feature.shape[1]
    growth_rate = N_feature.shape[-1]
    feat_point_num = M_feature.shape[1]
    order = tf.argsort(indices)
    x = [i for i in range(batch_size)]
    x = tf.convert_to_tensor(x)
    x = tf.reshape(x, [batch_size, 1, 1])
    x = tf.tile(x, [1, feat_point_num, 1])
    order = tf.expand_dims(order, axis=-1)
    order = tf.concat([x, order], axis=-1)
    tmp = tf.gather_nd(M_feature, order)
    order_ = tf.sort(indices)
    order_ = tf.expand_dims(order_, axis=-1)
    shape = tf.constant([int(batch_size), int(point_num), int(growth_rate)])
    order_ = tf.concat([x, order_], axis=-1)
    expand_feature = tf.scatter_nd(order_, tmp, shape)

    aggregate_feature = N_feature + expand_feature
    aggregate_feature = tf.expand_dims(aggregate_feature, axis=-2)
    return aggregate_feature
