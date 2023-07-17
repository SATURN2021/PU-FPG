import tensorflow as tf
from tf_ops.grouping.tf_grouping import knn_point_2


def knn(x, k=16, self_loop=False):
    """Construct edge feature for each point
    Args:
        x: (batch_size, num_points, num_dims)
        k: int
        self_loop: include the key (center point) or not?
    Returns:
        edge idx: (batch_size, num_points, k, num_dims)
    """
    if len(x.get_shape())>3:
        x = tf.squeeze(x, axis=2)
    _, idx = knn_point_2(k + 1, x, x, unique=True, sort=True)
    # this is only a naive version of self_loop implementation.
    if not self_loop:
        idx = idx[:, :, 1:, :]
    else:
        idx = idx[:, :, 0:-1, :]
#     print(idx)
    return idx


def get_graph_features(x, idx, return_central=True, feature_indices=None):
    """
    get the features for the neighbors and center points from the x and inx
    :param x: input features
    :param idx: the index for the neighbors and center points
    :return: 
    """
    if feature_indices is None:
        if len(x.get_shape())>3:
            x = tf.squeeze(x, axis=2)
        pc_neighbors = tf.gather_nd(x, idx)
        if return_central:
            pc_central = tf.tile(tf.expand_dims(x, axis=-2), [1, 1, idx.shape[2], 1])
            return pc_central, pc_neighbors
        else:
            return pc_neighbors
    else:
        batch_size = x.shape[0]
        channel = x.shape[-1]
        if len(x.get_shape())>3:
            x = tf.squeeze(x, axis=2)
        pc_neighbors = tf.gather_nd(x, idx)
#         x = tf.squeeze(x, axis=-2)
        y = tf.constant([i for i in range(channel)])
        y = tf.expand_dims(y, axis=0)
        y = tf.tile(y, [batch_size, 1])
        y = tf.expand_dims(y, axis=-1)
        feature_indices = tf.expand_dims(feature_indices, axis=-1)
        feature_indices = tf.concat([y, feature_indices], axis=-1)
        pc_central = tf.gather_nd(x, feature_indices)
        feature_points = pc_central
        if return_central:
            pc_central = tf.tile(tf.expand_dims(pc_central, axis=-2), [1, 1, idx.shape[2], 1])
            return pc_central, pc_neighbors, feature_points
        else:
            return pc_neighbors


def dil_knn(x, k=16, d=1, use_fsd=False, N_points=None):
    # x: (64, 256, 1, 3)
    if N_points is None:
        if len(x.get_shape()) > 3:
            x = tf.squeeze(x, axis=2)
        idx = knn(x, k=k*d)
        if d > 1:
            if use_fsd:
                idx = idx[:, :, k*(d-1):k*d, :]
            else:
                idx = idx[:, :, ::d, :]
        return idx
    else:
        if len(x.get_shape()) > 3:
            x = tf.squeeze(x, axis=2)
        if len(N_points.get_shape()) > 3:
            N_points = tf.squeeze(N_points, axis=2)
        idx = knn_M(x, N_points, k=10)
        if d > 1:
            if use_fsd:
                idx = idx[:, :, k*(d-1):k*d, :]
            else:
                idx = idx[:, :, ::d, :]
        return idx


def dyn_dil_get_graph_feature(x, k=16, d=1, use_fsd=False, return_central=True, N_points=None):
    """
    dynamically get the feature of the dilated GCN
    :param x: input feature
    :param k: number of neighbors
    :param d: dilation rate
    :param use_fsd: farthest point sampling, default False. Use uniform sampling
    :return: central feature, neighbors feature, edge index
    """
    if N_points is None:
        idx = dil_knn(x, k, d, use_fsd)
        if return_central:
            central, neighbors = get_graph_features(x, idx, True)
            return central, neighbors, idx
        else:
            neighbors = get_graph_features(x, idx, False)
            return neighbors, idx
    else:
        idx = dil_knn(x, k, d, use_fsd, N_points)
        if return_central:
            central, neighbors = get_graph_features(x, idx, True)
            return central, neighbors, idx
        else:
            neighbors = get_graph_features(x, idx, False)
            return neighbors, idx


def knn_M(x, all_points, k=16, self_loop=False):
    """Construct edge feature for each point
    Args:
        x: (batch_size, num_points, num_dims)
        k: int
        self_loop: include the key (center point) or not?
    Returns:
        edge idx: (batch_size, num_points, k, num_dims)
    """
    if len(x.get_shape())>3:
        x = tf.squeeze(x, axis=2)
    _, idx = knn_point_2(k + 1, all_points, x, unique=True, sort=True)

    if not self_loop:
        idx = idx[:, :, 1:, :]
    else:
        idx = idx[:, :, 0:-1, :]
    return idx
