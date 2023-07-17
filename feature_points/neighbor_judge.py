import tensorflow as tf
import numpy as np


def batch_distance_matrix_general(A, B):
    r_A = tf.reduce_sum(A * A, axis=2, keepdims=True)
    r_B = tf.reduce_sum(B * B, axis=2, keepdims=True)
    m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
    D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
    return D


def find_duplicate_columns(A):
    N = A.shape[0]
    P = A.shape[1]
    indices_duplicated = np.ones((N, 1, P), dtype=np.int32)
    for idx in range(N):
        _, indices = np.unique(A[idx], return_index=True, axis=0)
        indices_duplicated[idx, :, indices] = 0
    return indices_duplicated


def prepare_for_unique_top_k(D, A):
    indices_duplicated = tf.py_func(find_duplicate_columns, [A], tf.int32)
    D += tf.reduce_max(D) * tf.cast(indices_duplicated, tf.float32)


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
    D, idx = knn_point_2(k + 1, x, x, unique=True, sort=True)
    # this is only a naive version of self_loop implementation.
    if not self_loop:
        idx = idx[:, :, 1:, :]
        D = D[:, :, 1:]
    else:
        idx = idx[:, :, 0:-1, :]
        D = D[:, :, 0:-1]
    return idx, D


def knn_point_2(k, points, queries, sort=True, unique=True):
    """
    points: dataset points (N, P0, C)
    queries: query points (N, P, C)
    return indices is (N, P, C, 2) used for tf.gather_nd(points, indices)
    distances (N, P, C)
    """
    with tf.name_scope("knn_point"):
        batch_size = tf.shape(queries)[0]
        point_num = tf.shape(queries)[1]

        D = batch_distance_matrix_general(queries, points)
        if unique:
            prepare_for_unique_top_k(D, points)
        distances, point_indices = tf.nn.top_k(-D, k=k, sorted=sort)  # (N, P, K)
        batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, point_num, k, 1))
        indices = tf.concat([batch_indices, tf.expand_dims(point_indices, axis=3)], axis=3)
        return -distances, indices

    
def neighbor_judge(coords, k):
    indices, D = knn(coords, k=k, self_loop=False)
    pc_neighbors = tf.gather_nd(coords, indices)
    pc_central = tf.tile(tf.expand_dims(coords, axis=-2), [1, 1, indices.shape[2], 1])
    pc_local = pc_neighbors - pc_central
    neighbor_judge = tf.reduce_sum(pc_local, axis=-2)
    neighbor_judge = neighbor_judge * neighbor_judge
    neighbor_judge = tf.reduce_sum(neighbor_judge, axis=-1, keep_dims=True)
    return neighbor_judge
   