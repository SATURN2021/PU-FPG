# Author: Haochen Wang

import tensorflow as tf
from feature_points.neighbor_judge import knn
from tf_lib.common.tf_nn import fully_connected as fc
from tf_lib.common.tf_nn import conv2d


def batch_distance_matrix_general(A, B):
    r_A = tf.reduce_sum(A * A, axis=2, keepdims=True)
    r_B = tf.reduce_sum(B * B, axis=2, keepdims=True)
    m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
    D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
    return tf.sqrt(D)


def get_adjacency_matrix_from_knn(x, k4, feature_indices=None, M_idx=None, k3=10, fp_num=64, self_loop=True):
    batch_size = x.shape[0]
    point_num = x.shape[1]
    channels = x.shape[2]
    x = tf.cast(x, dtype=tf.float32)
    idx, _ = knn(x, k4)
    shape = tf.constant([int(batch_size), int(point_num), int(point_num), 1])
    order0 = tf.expand_dims(idx[:, :, :, 0], axis=-1)
    order2 = tf.expand_dims(idx[:, :, :, 1], axis=-1)
    order1 = [[[i for j in range(k4)] for i in range(point_num)] for m in range(batch_size)]
    order1 = tf.constant(order1)
    order1 = tf.reshape(order1, [batch_size, point_num, k4, 1])
    order = tf.concat([order0, order1, order2], axis=-1)
    tmp = [1 for i in range(batch_size * point_num * k4)]
    tmp = tf.constant(tmp)
    tmp = tf.reshape(tmp, [batch_size, point_num, k4, 1])
    A = tf.scatter_nd(order, tmp, shape)  # A: (batch_size, point_num, point_num, 1)
    if feature_indices is not None:
        f_i = tf.expand_dims(feature_indices, axis=-1)
        f_i = tf.tile(f_i, [1, 1, k3])
        f_i = tf.expand_dims(f_i, axis=-1)
        # M_idx: (64, 64, 10, 2)
        order0 = tf.expand_dims(M_idx[:, :, :, 0], axis=-1)
        order2 = tf.expand_dims(M_idx[:, :, :, 1], axis=-1)
        order1 = f_i
        order = tf.concat([order0, order1, order2], axis=-1)
        tmp = [1 for i in range(batch_size * fp_num * k3)]
        tmp = tf.constant(tmp)
        tmp = tf.reshape(tmp, [batch_size, fp_num, k3, 1])
        shape = tf.constant([int(batch_size), int(point_num), int(point_num), 1])
        fp_A = tf.scatter_nd(order, tmp, shape)
        A = A + fp_A
    A = tf.squeeze(A, axis=-1)  # A: (batch_size, point_num, point_num)
    A = tf.transpose(A, [0, 2, 1])
    A = tf.cast(A, dtype=tf.float32)
    if self_loop:
        diagonal = [1. for i in range(batch_size * point_num)]
        diagonal = tf.constant(diagonal)
        A = A + tf.matrix_diag(diagonal)
    return A


def get_adjacency_matrix_from_MLknn(point_num=256, feature_indices=None, M_idx=None, k3=10, self_loop=True):
    batch_size = feature_indices.shape[0]
    fp_num = feature_indices.shape[1]
    f_i = tf.expand_dims(feature_indices, axis=-1)
    f_i = tf.tile(f_i, [1, 1, k3])
    f_i = tf.expand_dims(f_i, axis=-1)
    # M_idx: (64, 64, 10, 2)
    order0 = tf.expand_dims(M_idx[:, :, :, 0], axis=-1)
    order2 = tf.expand_dims(M_idx[:, :, :, 1], axis=-1)
    order1 = f_i
    order = tf.concat([order0, order1, order2], axis=-1)
    tmp = [1 for i in range(batch_size * fp_num * k3)]
    tmp = tf.constant(tmp)
    tmp = tf.reshape(tmp, [batch_size, fp_num, k3, 1])
    shape = tf.constant([int(batch_size), int(point_num), int(point_num), 1])
    fp_A = tf.scatter_nd(order, tmp, shape)
    A = tf.squeeze(fp_A, axis=-1)
    A = tf.transpose(A, [0, 2, 1])
    A = tf.cast(A, dtype=tf.float32)
    if self_loop:
        diagonal = [1. for i in range(point_num)]
        diagonal = tf.constant(diagonal)
        diagonal = tf.matrix_diag(diagonal)
        diagonal = tf.expand_dims(diagonal, axis=0)
        diagonal = tf.tile(diagonal, [batch_size, 1, 1])
        A = A + diagonal
    return A


def get_Af(A, use_special=False):
    # A: (B, point_num, point_num)
    batch_size = A.shape[0]
    point_num = A.shape[1]
    A = tf.cast(A, dtype=tf.float32)
    A_ = tf.transpose(A, [0, 2, 1])
    A = A + A_
    if use_special:
        max_number = tf.reduce_max(tf.reduce_max(A, axis=-1, keepdims=True), axis=-2, keepdims=True)
        max_matrix = tf.tile(max_number, [1, point_num, point_num])
        special_pos = A - max_matrix
        special_pos = tf.cast(special_pos, dtype=tf.bool)
        special_pos = tf.cast(special_pos, dtype=tf.int32)
        special_mat = tf.constant([1])
        special_mat = tf.reshape(special_mat, [1, 1, 1])
        special_mat = tf.tile(special_mat, [batch_size, point_num, point_num])
        special_pos = special_pos - special_mat
        special_pos = tf.cast(special_pos, dtype=tf.bool)
        general_pos = tf.cast(A, dtype=tf.bool)
        zeros = tf.zeros_like(A)
        ones = tf.ones(general_pos.get_shape())
        Af0 = tf.where(general_pos, ones, zeros)
        Af1 = tf.where(special_pos, ones, zeros)
        Af = Af0 + Af1
    else:
        Af = A
    return Af


def get_ASin(A):
    batch_size = A.shape[0]
    point_num = A.shape[1]
    A = tf.cast(A, dtype=tf.float32)
    denominator = tf.reduce_sum(A, axis=-1)
    denominator = tf.expand_dims(denominator, axis=1)
    denominator = tf.expand_dims(denominator, axis=1)
    numerator0 = tf.transpose(A, [0, 2, 1])
    numerator0 = tf.expand_dims(numerator0, axis=-1)
    numerator1 = tf.expand_dims(A, axis=1)
    numerator = numerator0 * numerator1
    numerator = tf.transpose(numerator, [0, 1, 3, 2])
    denominator = tf.cast(denominator, dtype=tf.float32)
    denominator = denominator + tf.constant(0.0001)
    ASin = numerator / denominator
    ASin = tf.reduce_sum(ASin, axis=-1)
    return ASin


def get_ASout(A):
    batch_size = A.shape[0]
    point_num = A.shape[1]
    A = tf.cast(A, dtype=tf.float32)
    denominator = tf.transpose(A, [0, 2, 1])
    denominator = tf.reduce_sum(denominator, axis=-1)
    denominator = tf.expand_dims(denominator, axis=1)
    denominator = tf.expand_dims(denominator, axis=1)
    numerator0 = tf.expand_dims(A, axis=-1)
    numerator1 = tf.transpose(A, [0, 2, 1])
    numerator1 = tf.expand_dims(numerator1, axis=1)
    numerator = numerator0 * numerator1
    numerator = tf.transpose(numerator, [0, 1, 3, 2])
    denominator = tf.cast(denominator, dtype=tf.float32)
    denominator = denominator + tf.constant(0.0001)
    ASout = numerator / denominator
    ASout = tf.reduce_sum(ASout, axis=-1)
    return ASout


def get_proximity_matrix(input_matrix):
    batch_size = input_matrix.shape[0]
    point_num = input_matrix.shape[1]
    input_matrix = tf.cast(input_matrix, dtype=tf.float32)
    self_loop_A = input_matrix
    degrees = tf.reduce_sum(self_loop_A, axis=-1)
    ones = tf.ones(degrees.get_shape())
    degrees = degrees + 0.0001 * ones
    degrees = tf.sqrt(degrees)
    degrees = ones / degrees
    former = tf.expand_dims(degrees, axis=-1)
    later = tf.expand_dims(degrees, axis=-2)
    proximity_matrix = former * later * self_loop_A
    return proximity_matrix


def proximity_conv(PM_f, PM_Sin, PM_Sout, batch_size, features, num_out_channels=32, scope=None, is_training=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as sc:
        batch_size = PM_f.shape[0]
        point_num = PM_f.shape[1]
        channels = features.shape[2]
        PM_f = tf.cast(PM_f, dtype=tf.float32)
        PM_Sin = tf.cast(PM_Sin, dtype=tf.float32)
        PM_Sout = tf.cast(PM_Sout, dtype=tf.float32)
        PM_f = make_it_sparse(PM_f)
        PM_Sin = make_it_sparse(PM_Sin)
        PM_Sout = make_it_sparse(PM_Sout)
        f = tf.matmul(PM_f, features)
        Sin = tf.matmul(PM_Sin, features)
        Sout = tf.matmul(PM_Sout, features)
        f = tf.expand_dims(f, axis=-2)
        Sin = tf.expand_dims(Sin, axis=-2)
        Sout = tf.expand_dims(Sout, axis=-2)
        Zf = conv2d(f, num_out_channels, [1, 1], scope=scope, padding='VALID', is_training=is_training)
        ZSin = conv2d(Sin, num_out_channels, [1, 1], scope=scope, padding='VALID', 
                      is_training=is_training)
        ZSout = conv2d(Sout, num_out_channels, [1, 1], scope=scope, padding='VALID', 
                       is_training=is_training)
        Zf = tf.reshape(Zf, [batch_size, point_num, num_out_channels])
        ZSin = tf.reshape(ZSin, [batch_size, point_num, num_out_channels])
        ZSout = tf.reshape(ZSout, [batch_size, point_num, num_out_channels])
        output = tf.concat([Zf, ZSin, ZSout], axis=-1)
        return output


def adjacency_normalization(A):
    A = tf.cast(A, dtype=tf.float32)
    batch_size = A.shape[0]
    point_num = A.shape[1]
    M = tf.reduce_max(tf.reduce_max(A, axis=-1), axis=-1)
    m = tf.reduce_min(tf.reduce_min(A, axis=-1), axis=-1)
    k = tf.cast(M - m, dtype=tf.float32) + tf.constant(0.0001)
    b = tf.cast(m, dtype=tf.float32)
    k = tf.reshape(k, [batch_size, 1, 1])
    b = tf.reshape(b, [batch_size, 1, 1])
    res = (A - b) / k
    res = tf.cast(res, dtype=tf.float32)
    return res


def GUM(features, A, GUM_sigma, Af, scope=None, is_training=None, dis_mat=None, lim_mat=None, use_lim_mat=True, return_lim_mat=True, exp_upper=6.0, exp_lower=-15.0,
        RAM_w=1.0):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as sc:
        A = tf.cast(A, dtype=tf.float32)
        features = tf.cast(features, dtype=tf.float32)
        GUM_sigma = tf.cast(GUM_sigma, dtype=tf.float32)
        batch_size = A.shape[0]
        point_num = A.shape[1]
        channels = features.shape[2]

        D = batch_distance_matrix_general(features, features)

        sigma = GUM_sigma
        G = -D / (2 * sigma ** 2)

        b = exp_upper
        k = -(exp_upper - exp_lower) / tf.reduce_min(tf.reduce_min(G, axis=-1), axis=-1)
        k = tf.reshape(k, [-1, 1, 1])
        G = k * G + b

        G = tf.exp(G)
        residual_A = adjacency_normalization(G)
        if use_lim_mat:
            if lim_mat is None:
                A_ = tf.cast(A, dtype=tf.bool)
                A_ = tf.cast(A_, dtype=tf.float32)
                Af_ = tf.cast(Af, dtype=tf.bool)
                Af_ = tf.cast(Af_, dtype=tf.float32)
                lim_mat = Af_ - A_
                ones = tf.ones(shape=lim_mat.get_shape(), dtype=tf.float32)
                lim_mat = ones - lim_mat
        else:
            lim_mat = 1.
        res = A + RAM_w * residual_A * lim_mat
        res = adjacency_normalization(res)
        if return_lim_mat:
            return res, lim_mat, D
        else:
            return res, D


def DGCN(DGG, inputs, batch_size, num_out_channels=32, scope=None, is_training=None):
    if len(inputs.get_shape()) > 3:
        inputs = tf.squeeze(inputs, axis=-2)
    DGG = tf.cast(DGG, dtype=tf.float32)
    inputs = tf.cast(inputs, dtype=tf.float32)
    Af = get_Af(DGG)
    ASin = get_ASin(DGG)
    ASout = get_ASout(DGG)
    PM_f = get_proximity_matrix(Af)
    PM_Sin = get_proximity_matrix(ASin)
    PM_Sout = get_proximity_matrix(ASout)
    topo_feat = proximity_conv(PM_f, PM_Sin, PM_Sout, batch_size, features=inputs, 
                               num_out_channels=num_out_channels, scope=scope, is_training=is_training)
    topo_feat = tf.expand_dims(topo_feat, axis=-2)
    return topo_feat


def denseGCN_DGG(DGG, inputs, growth_rate=24, scope=None, scope_m=None, scope_c=None, is_training=None, dense=False, n_layers=3):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        DGG = tf.cast(DGG, dtype=tf.float32)
        inputs = tf.cast(inputs, dtype=tf.float32)
        point_num = inputs.shape[1]
        channels = inputs.shape[2]
        neighbors = tf.expand_dims(inputs, axis=1)
        central = tf.expand_dims(inputs, axis=2)
        message = neighbors - central
        message = tf.transpose(message, [0, 2, 1, 3])
        DGG = tf.expand_dims(DGG, axis=-1)
        message = message * DGG
        message = tf.reduce_sum(message, axis=1)
        message = tf.expand_dims(message, axis=2)
        message = conv2d(message, growth_rate, [1, 1], padding='VALID',
                         scope=scope_m, use_bias=True, activation_fn=None, is_training=is_training)
        inputs = tf.expand_dims(inputs, axis=-2)
        x_center = conv2d(inputs, growth_rate, [1, 1], padding='VALID',
                          scope=scope_c, use_bias=False, activation_fn=None, is_training=is_training)
        edge_features = x_center + message
        y = tf.nn.relu(edge_features)
        if dense:
            features = [y]
            for i in range(n_layers - 1):
                if i == 0:
                    y = conv2d(y, growth_rate, [1, 1], padding='VALID',
                               scope='l_%d' % i, is_training=is_training)
                else:
                    y = tf.concat(features, axis=-1)
                features.append(conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l_%d' % i, is_training=is_training))

            if n_layers > 1:
                y = tf.concat(features, axis=-1)
            y = tf.reduce_max(y, axis=-2, keepdims=True)
        return y
    
     
def inception_DuoGCN(input_spat_feat, 
                     input_topo_feat, 
                     DGG, 
                     batch_size,
                     growth_rate=32, 
                     topo_channels=16,
                     k4=10,
                     GUM_sigma=2,
                     scope=None,
                     is_training=None, 
                     use_residual=False, 
                     dis_mat=None, 
                     lim_mat=None,
                     use_lim_mat=True, 
                     return_lim_mat=True, 
                     exp_upper=6.0,
                     exp_lower=-15.0,
                     RAM_w=1.0):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        from tf_lib.gcn_lib.vertex import densegcn as denseGCN_ORI
        DGG = tf.cast(DGG, dtype=tf.float32)
        input_spat_feat = tf.cast(input_spat_feat, dtype=tf.float32)
        input_topo_feat = tf.cast(input_topo_feat, dtype=tf.float32)
        compressed_spat_feat = conv2d(input_spat_feat, growth_rate, 1, padding='VALID', scope='compressed_spat_feat_'+str(scope[-1]), is_training=is_training)
        compressed_topo_feat = conv2d(input_topo_feat, growth_rate, 1, padding='VALID', scope='compressed_topo_feat_'+str(scope[-1]), is_training=is_training)
        Af = get_Af(DGG)
        hybrid_feat = tf.concat([compressed_spat_feat, compressed_topo_feat], axis=-1)
        hybrid_feat = tf.squeeze(hybrid_feat, axis=-2)
        if lim_mat is None:
            DGG, lim_mat, D = GUM(hybrid_feat, A=DGG, GUM_sigma=GUM_sigma, Af=Af, scope='inception_GUM_'+str(scope[-1]), 
                                  is_training=is_training, dis_mat=dis_mat, lim_mat=lim_mat, use_lim_mat=use_lim_mat, return_lim_mat=True,
                                  exp_upper=exp_upper, exp_lower=exp_lower, RAM_w=RAM_w)    # DGG_updated
        else:
            DGG, D = GUM(hybrid_feat, A=DGG, GUM_sigma=GUM_sigma, Af=Af, scope='inception_GUM_'+str(scope[-1]), 
                         is_training=is_training, dis_mat=dis_mat, lim_mat=lim_mat, use_lim_mat=use_lim_mat, return_lim_mat=False, 
                         exp_upper=exp_upper, exp_lower=exp_lower, RAM_w=RAM_w)    # DGG_updated
        compressed_topo_feat = tf.squeeze(compressed_topo_feat, axis=-2)
        output_topo_feat = DGCN(DGG, inputs=compressed_topo_feat, batch_size=batch_size, num_out_channels=topo_channels, scope='DGCN_'+str(scope[-1]), is_training=is_training)
        compressed_spat_feat = tf.squeeze(compressed_spat_feat, axis=-2)
        output_spat_feat_0 = denseGCN_DGG(DGG, inputs=compressed_spat_feat, growth_rate=growth_rate, 
                                          scope='sf_'+str(scope[-1])+'0', scope_m='m'+str(scope[-1])+'0', 
                                          scope_c='c'+str(scope[-1])+'0', is_training=is_training, dense=True, n_layers=3)
        compressed_spat_feat = tf.expand_dims(compressed_spat_feat, axis=-2)
        output_spat_feat_1 = denseGCN_ORI(compressed_spat_feat, idx=None, growth_rate=growth_rate, n_layers=3, 
                                          k=k4, d=1, return_idx=False, scope='sf_'+str(scope[-1])+'1', N_points=None, is_training=is_training)
        output_spat_feat_2 = tf.reduce_max(input_spat_feat, axis=-1, keepdims=True)
        output_spat_feat = tf.concat([output_spat_feat_0, output_spat_feat_1, output_spat_feat_2], axis=-1)
        if use_residual:
            output_spat_feat = output_spat_feat + input_spat_feat
        if return_lim_mat:
            return DGG, output_spat_feat, output_topo_feat, lim_mat, D
        else:
            return DGG, output_spat_feat, output_topo_feat, D
        
        
def sparse_matmul(inputs, features, batch_size):
    x = tf.reshape(inputs, [batch_size, -1])
    y = tf.sort(x, direction='DESCENDING')
    t = y[:, 64]
    t = tf.reshape(t, [batch_size, 1, 1])
    z = inputs - t
    z = tf.nn.relu(z)
    for i in range(batch_size):
        former = z[i]
        later = features[i]
        res = tf.matmul(former, later, a_is_sparse=True)
        res = tf.expand_dims(res, axis=0)
        if i == 0:
            ans = res
        else:
            ans = tf.concat([ans, res], axis=0)
    return ans
    

def make_it_sparse(inputs):
    batch_size = inputs.shape[0]
    x = tf.reshape(inputs, [batch_size, -1])
    y = tf.sort(x, direction='DESCENDING')
    t = y[:, 64]
    t = tf.reshape(t, [batch_size, 1, 1])
    z = inputs - t
    z = tf.nn.relu(z)
    return z
    
    