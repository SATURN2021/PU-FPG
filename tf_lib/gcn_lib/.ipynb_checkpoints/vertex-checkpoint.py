import tensorflow as tf
from .edge import get_graph_features, dyn_dil_get_graph_feature, knn, dil_knn, knn_M
from ..common import conv2d
import numpy as np
from feature_points.feature_aggregate import feature_aggregate
from graph_utils import get_Af, GUM, DGCN, denseGCN_DGG
# from graph_utils import *
from tf_lib.common.tf_nn import conv2d as conv2d_


def edge_conv(x,
              out_channels=64,
              idx=None, k=16, d=1,
              n_layers=1,
              scope='edge_conv',
              feature_indices=None,
              **kwargs):
    """
    :param x: input features
    :param idx: edge index for neighbors and centers (if None, has to use KNN to build a graph at first)
    :param out_channels: output channel,
    :param n_layers: number of GCN layers.
    :param scope: the name
    :param kwargs:
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if feature_indices is None:
            if idx is None:
                idx = dil_knn(x, k, d)
            central, neighbors = get_graph_features(x, idx)
            message = conv2d(neighbors - central, out_channels, [1, 1], padding='VALID',
                             scope='message', use_bias=True, activation_fn=None, **kwargs)
            x_center = conv2d(x, out_channels, [1, 1], padding='VALID',
                              scope='central', use_bias=False, activation_fn=None, **kwargs)
            edge_features = x_center + message
            edge_features = tf.nn.relu(edge_features)
            for i in range(n_layers - 1):
                edge_features = conv2d(edge_features, out_channels, [1, 1], padding='VALID',
                                       scope='l%d' % i, **kwargs)
            y = tf.reduce_max(edge_features, axis=-2, keepdims=True)
        else:
            if idx is None:
                idx = dil_knn(x, k, d)
            central, neighbors, feature_points = get_graph_features(x, idx, feature_indices=feature_indices)
            message = conv2d(neighbors - central, out_channels, [1, 1], padding='VALID',
                             scope='message', use_bias=True, activation_fn=None, **kwargs)
            feature_points = tf.expand_dims(feature_points, axis=-2)
            x_center = conv2d(feature_points, out_channels, [1, 1], padding='VALID',
                              scope='central', use_bias=False, activation_fn=None, **kwargs)
            edge_features = x_center + message
            edge_features = tf.nn.relu(edge_features)
            for i in range(n_layers - 1):
                edge_features = conv2d(edge_features, out_channels, [1, 1], padding='VALID',
                                       scope='l%d' % i, **kwargs)
            y = tf.reduce_max(edge_features, axis=-2, keepdims=True)
    return y


def densegcn(x,
             idx=None,
             growth_rate=12, n_layers=3, k=16, d=1,
             return_idx=False,
             scope='densegcn',
             N_points=None,
             **kwargs):
    """
    :param x: input features
    :param idx: edge index for neighbors and centers (if None, has to use KNN to build a graph at first)
    :param growth_rate: output channel of each path,
    :param n_layers: number of GCN layers.
    :param k: the kernel size (num of neighbors used in each path)
    :param d: dilation rate of each path
    :param scope: the name
    :param kwargs:
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if N_points is None:
            if idx is None:
                central, neighbors, idx = dyn_dil_get_graph_feature(x, k, d)
            else:
                central, neighbors = get_graph_features(x, idx)

            message = conv2d(neighbors - central, growth_rate, [1, 1], padding='VALID',
                             scope='message', use_bias=True, activation_fn=None, **kwargs)
            x_center = conv2d(x, growth_rate, [1, 1], padding='VALID',
                              scope='central', use_bias=False, activation_fn=None, **kwargs)
            edge_features = x_center + message
            y = tf.nn.relu(edge_features)

            features = [y]
            for i in range(n_layers - 1):
                if i == 0:
                    y = conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l_%d' % i, **kwargs)
                else:
                    y = tf.concat(features, axis=-1)
                features.append(conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l_%d' % i, **kwargs))

            if n_layers > 1:
                y = tf.concat(features, axis=-1)
            y = tf.reduce_max(y, axis=-2, keepdims=True)

            if return_idx:
                return y, idx
            else:
                return y
        else:
            if idx is None:
                central, neighbors, idx = dyn_dil_get_graph_feature(x, k, d, N_points=N_points)
            else:
                central, neighbors = get_graph_features(x, idx)

            message = conv2d(neighbors - central, growth_rate, [1, 1], padding='VALID',
                             scope='message', use_bias=True, activation_fn=None, **kwargs)
            x_center = conv2d(x, growth_rate, [1, 1], padding='VALID',
                              scope='central', use_bias=False, activation_fn=None, **kwargs)
            edge_features = x_center + message
            y = tf.nn.relu(edge_features)

            features = [y]
            for i in range(n_layers - 1):
                if i == 0:
                    y = conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l_%d' % i, **kwargs)
                else:
                    y = tf.concat(features, axis=-1)
                features.append(conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l_%d' % i, **kwargs))

            if n_layers > 1:
                y = tf.concat(features, axis=-1)
            y = tf.reduce_max(y, axis=-2, keepdims=True)

            if return_idx:
                return y, idx
            else:
                return y


def gcn(x,
        idx=None,
        growth_rate=12, n_layers=3, k=16, d=1,
        return_idx=False,
        scope='densegcn',
        **kwargs):
    """
    :param x: input features
    :param idx: edge index for neighbors and centers (if None, has to use KNN to build a graph at first)
    :param growth_rate: output channel of each path,
    :param n_layers: number of GCN layers.
    :param k: the kernel size (num of neighbors used in each path)
    :param d: dilation rate of each path
    :param scope: the name
    :param kwargs:
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if idx is None:
            central, neighbors, idx = dyn_dil_get_graph_feature(x, k, d)
        else:
            central, neighbors = get_graph_features(x, idx)

        message = conv2d(neighbors - central, growth_rate, [1, 1], padding='VALID',
                         scope='message', use_bias=True, activation_fn=None, **kwargs)
        x_center = conv2d(x, growth_rate, [1, 1], padding='VALID',
                          scope='central', use_bias=False, activation_fn=None, **kwargs)
        edge_features = x_center + message
        y = tf.nn.relu(edge_features)

        y = conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l_%d' % 0, **kwargs)
        for i in range(n_layers - 1):
            y = conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l_%d' % i, **kwargs)
        y = tf.reduce_max(y, axis=-2, keepdims=True)

        if return_idx:
            return y, idx
        else:
            return y


def inception_densegcn(x,
                       growth_rate=12,
                       k=16, d=2, n_dense=3,
                       use_global_pooling=True,
                       use_residual=True,
                       use_dilation=True,
                       scope='inception_resgcn',
                       **kwargs):
    """
    :param x: input features
    :param growth_rate: output channel of each path,
    :param k: the kernel size (num of neighbors used in each path)
    :param d: dilation rate of each path (default: 2)
    :param n_dense: number of layers in each denseGCN block, default is 3
    :param scope: the name
    :param kwargs:
    :return:
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        idx = knn(x, k=k * d)  # [B N K 2]
        idx1 = idx[:, :, :20, :]
        if use_dilation:
            idx2 = idx[:, :, ::d, :]
        else:
            idx2 = idx
        inception_reduction = conv2d(x, growth_rate, 1,
                                     padding='VALID', scope='inception_1_reduction', **kwargs)
        inception_1 = densegcn(inception_reduction, idx1, growth_rate, n_layers=n_dense,
                               scope='inception_1', **kwargs)
        inception_2 = densegcn(inception_reduction, idx2, growth_rate, n_layers=n_dense,
                               scope='inception_2', **kwargs)

        if use_global_pooling:
            inception_3 = tf.reduce_max(x, axis=-1, keepdims=True)
            inception_out = tf.concat([inception_1, inception_2, inception_3], axis=-1)
        else:
            inception_out = tf.concat([inception_1, inception_2], axis=-1)

        if use_residual and x.get_shape()[-1] == inception_out.get_shape()[-1]:
            inception_out = inception_out + x
    return inception_out


def inception_1densegcn(x,
                        growth_rate=12,
                        k=16, d=2, n_dense=3,
                        use_global_pooling=True,
                        use_residual=True,
                        use_dilation=True,
                        scope='inception_resgcn',
                        **kwargs):
    """
    :param x: input features
    :param growth_rate: output channel of each path,
    :param k: the kernel size (num of neighbors used in each path)
    :param d: dilation rate of each path (default: 2)
    :param n_dense: number of layers in each denseGCN block, default is 3
    :param scope: the name
    :param kwargs:
    :return:
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        idx = knn(x, k=k)  # [B N K 2]

        inception_reduction = conv2d(x, growth_rate, 1,
                                     padding='VALID', scope='inception_1_reduction', **kwargs)
        inception_1 = densegcn(inception_reduction, idx, growth_rate, n_layers=n_dense,
                               scope='inception_1', **kwargs)

        if use_global_pooling:
            inception_3 = tf.reduce_max(x, axis=-1, keepdims=True)
            inception_out = tf.concat([inception_1, inception_3], axis=-1)
        else:
            inception_out = inception_1

        if use_residual and x.get_shape()[-1] == inception_out.get_shape()[-1]:
            inception_out = inception_out + x
    return inception_out


def inceptiongcn(x,
                 growth_rate=12,
                 k=16, d=2, n_dense=3,
                 use_global_pooling=True,
                 use_residual=True,
                 use_dilation=True,
                 scope='inception_resgcn',
                 **kwargs):
    """
    :param x: input features
    :param growth_rate: output channel of each path,
    :param k: the kernel size (num of neighbors used in each path)
    :param d: dilation rate of each path (default: 2)
    :param n_dense: number of layers in each denseGCN block, default is 3
    :param scope: the name
    :param kwargs:
    :return:
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        idx = knn(x, k=k * d)  # [B N K 2]
        idx1 = idx[:, :, :k, :]
        if use_dilation:
            idx2 = idx[:, :, ::d, :]
        else:
            idx2 = idx
        inception_reduction = conv2d(x, growth_rate, 1,
                                     padding='VALID', scope='inception_1_reduction', **kwargs)
        inception_1 = gcn(inception_reduction, idx1, growth_rate, n_layers=n_dense,
                          scope='inception_1', **kwargs)
        inception_2 = gcn(inception_reduction, idx2, growth_rate, n_layers=n_dense,
                          scope='inception_2', **kwargs)

        if use_global_pooling:
            inception_3 = tf.reduce_max(x, axis=-1, keepdims=True)
            inception_out = tf.concat([inception_1, inception_2, inception_3], axis=-1)
        else:
            inception_out = tf.concat([inception_1, inception_2], axis=-1)

        if use_residual and x.get_shape()[-1] == inception_out.get_shape()[-1]:
            inception_out = inception_out + x
    return inception_out


def point_shuffler(inputs, scale=2):
    """
    Periodic shuffling layer for point cloud
    """
    outputs = tf.reshape(inputs, [tf.shape(inputs)[0], tf.shape(inputs)[1], 1, tf.shape(inputs)[3] // scale, scale])
    outputs = tf.transpose(outputs, [0, 1, 4, 3, 2])
    outputs = tf.reshape(outputs, [tf.shape(inputs)[0], tf.shape(inputs)[1] * scale, 1, tf.shape(inputs)[3] // scale])
    return outputs


def nodeshuffle(x=None,
                DGG=None, spat_feat=None, topo_feat=None,
                GUM_sigma=1.0,
                batch_size=24, 
                topo_channels=32, 
                dis_mat=None,
                lim_mat=None,
                scale=2,
                k=16, d=1, 
                channels=64,
                N_idx=None,
                M_idx_1=None,
                M_idx_2=None,
                M_idx_3=None,
                M_idx_4=None,
                scope='nodeshuffle',
                feature_indices_1=None,
                feature_indices_2=None,
                feature_indices_3=None,
                feature_indices_4=None,
                is_training=None,
                **kwargs):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):        
        features = conv2d(x, channels, 1,
                   padding='VALID', scope='up_reduction', **kwargs)
        y = edge_conv(features,
                      channels * scale,
                      idx=N_idx,
                      n_layers=1,
                      scope='edge_conv0',
                      **kwargs)
        if feature_indices_1 is None:
            if topo_feat is not None:
                topo_feat = DGCN(DGG, inputs=topo_feat, batch_size=batch_size, num_out_channels=topo_channels, scope='DGCN_no_FP', is_training=is_training)
                y = tf.concat([y, topo_feat], axis=-1)
            y = point_shuffler(y, scale)
            return y
        else:
            z_1 = edge_conv(features,
                            channels * scale,
                            idx=M_idx_1,
                            n_layers=1,
                            scope='edge_conv1',
                            feature_indices=feature_indices_1,
                            **kwargs)
            z_2 = edge_conv(features,
                            channels * scale,
                            idx=M_idx_2,
                            n_layers=1,
                            scope='edge_conv2',
                            feature_indices=feature_indices_2,
                            **kwargs)
            z_3 = edge_conv(features,
                            channels * scale,
                            idx=M_idx_3,
                            n_layers=1,
                            scope='edge_conv3',
                            feature_indices=feature_indices_3,
                            **kwargs)
            z_4 = edge_conv(features,
                            channels * scale,
                            idx=M_idx_4,
                            n_layers=1,
                            scope='edge_conv4',
                            feature_indices=feature_indices_4,
                            **kwargs)
            aggregate_feature = feature_aggregate(N_feature=y, M_feature=z_1, indices=feature_indices_1)
            aggregate_feature = feature_aggregate(N_feature=aggregate_feature, M_feature=z_2, indices=feature_indices_2)
            aggregate_feature = feature_aggregate(N_feature=aggregate_feature, M_feature=z_3, indices=feature_indices_3)
            aggregate_feature = feature_aggregate(N_feature=aggregate_feature, M_feature=z_4, indices=feature_indices_4)
            if topo_feat is not None:
                topo_feat = DGCN(DGG, inputs=topo_feat, batch_size=batch_size, num_out_channels=topo_channels, scope='DGCN_no_FP', is_training=is_training)
                aggregate_feature = tf.concat([aggregate_feature, topo_feat], axis=-1)
            aggregate_feature = point_shuffler(aggregate_feature, scale)
            return aggregate_feature


def mlpshuffle(x, scale=2,
               channels=64,
               scope='mlpshuffle', **kwargs):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        y = conv2d(x, channels, 1,
                   padding='VALID', scope='up_reduction', **kwargs)

        # MLPs
        y = conv2d(y, channels, 1,
                   padding='VALID', scope='conv1', **kwargs)
        y = conv2d(y, channels, 1,
                   padding='VALID', scope='conv2', **kwargs)
        y = conv2d(y, channels, 1,
                   padding='VALID', scope='conv3', **kwargs)
        y = conv2d(y, channels, 1,
                   padding='VALID', scope='conv4', **kwargs)
        y = conv2d(y, channels * scale, 1,
                   padding='VALID', scope='conv5', **kwargs)

        y = point_shuffler(y, scale)
        return y


def multi_cnn(x, scale=2, scope='multi_cnn',
              **kwargs):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        new_points_list = []
        n_channels = x.get_shape().as_list()[-1]
        for i in range(scale):
            branch_feat = conv2d(x, n_channels, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 use_bn=False,
                                 scope='branch_%d' % (i), **kwargs)
            new_points_list.append(branch_feat)
        out = tf.concat(new_points_list, axis=1)
    return out


def gen_grid(num_grid_point):
    """
    generate unique indicator for duplication based upsampling module.
    output [num_grid_point, 2]
    """
    x = tf.lin_space(-0.2, 0.2, num_grid_point)
    x, y = tf.meshgrid(x, x)
    grid = tf.reshape(tf.stack([x, y], axis=-1), [-1, 2])  # [2, 2, 2] -> [4, 2]
    return grid


def duplicate(x, scale, scope='duplicate',
              **kwargs):
    """
    Our implementation of duplicate-based upsampling module used in PU-Net Paper
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        batch_size, n_points, _, n_channels = x.get_shape().as_list()

        grid = gen_grid(np.round(np.sqrt(scale)).astype(np.int32))
        grid = tf.tile(tf.expand_dims(grid, 0),
                       [batch_size, n_points, 1])
        grid = tf.expand_dims(grid, axis=-2)

        x = tf.reshape(
            tf.tile(tf.expand_dims(x, 2), [1, 1, scale, 1, 1]),
            [batch_size, n_points * scale, 1, n_channels])
        x = tf.concat([x, grid], axis=-1)
        x = conv2d(x, 128, [1, 1],
                   padding='VALID', stride=[1, 1],
                   scope='up_layer1',
                   **kwargs)
        x = conv2d(x, 128, [1, 1],
                   padding='VALID', stride=[1, 1],
                   scope='up_layer2',
                   **kwargs)
    return x
