# -*- coding: utf-8 -*-
# @Time        : 16/1/2019 5:04 PM
# @Description :
# @Author      : li rui hui & Guocheng Qian
# @Email       : ruihuili@gmail.com
# Modified by Haochen Wang

import tensorflow as tf
from Upsampling.discriminator import Discriminator
from Common.visu_utils import plot_pcd_three_views, point_cloud_three_views
from Common.ops import add_scalar_summary, add_hist_summary
from Upsampling.data_loader import Fetcher
from Common import model_utils
from Common import pc_util
from Common.loss_utils import pc_distance, get_uniform_loss, get_repulsion_loss, discriminator_loss, generator_loss
from tf_ops.sampling.tf_sampling import farthest_point_sample
import logging
import os
from tqdm import tqdm, trange
from glob import glob
import math
from time import time
from termcolor import colored
import numpy as np
from Common.model_utils import get_model_cls
from feature_points.compute_normals import compute_normals


class Model(object):
    def __init__(self, opts, sess):
        self.sess = sess
        self.opts = opts

    def allocate_placeholders(self):
        self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.input_x = tf.placeholder(tf.float32, shape=[self.opts.batch_size, self.opts.num_point, 3])
        self.input_y = tf.placeholder(tf.float32, shape=[self.opts.batch_size,
                                                         int(self.opts.up_ratio * self.opts.num_point), 3])
        self.pc_radius = tf.placeholder(tf.float32, shape=[self.opts.batch_size])
        self.normals = tf.placeholder(tf.float32, shape=[self.opts.batch_size, self.opts.num_point, 3])
        
    def build_model(self):
        logging.info("========== Building Model ==========")
        model_cls = get_model_cls(self.opts.model)
        self.G = model_cls(self.opts, self.is_training, name='generator')

        if self.opts.use_gan:
            self.D = Discriminator(self.opts, self.is_training, name='discriminator')

        # X -> Y
        if self.opts.model.lower() == 'punet':
            self.G_y = self.G(self.input_x, self.pc_radius)
        elif self.opts.model.lower() == 'pufpg':
            self.G_y = self.G(self.input_x, self.normals)
        else:
            self.G_y = self.G(self.input_x)
        
        self.dis_loss = self.opts.fidelity_w * pc_distance(self.G_y, 
                                                           self.input_y, 
                                                           radius=self.pc_radius, 
                                                           threshold=self.opts.cd_threshold, 
                                                           dis_type=self.opts.loss_type)

        self.pu_loss = self.dis_loss
        if self.opts.repulse:
            self.repulsion_loss = self.opts.repulsion_w * get_repulsion_loss(self.G_y)
            self.pu_loss += self.repulsion_loss

        if self.opts.uniform:
            self.uniform_loss = self.opts.uniform_w * get_uniform_loss(self.G_y)
            self.pu_loss += self.uniform_loss

        if self.opts.reg:
            self.pu_loss += tf.losses.get_regularization_loss()

        if self.opts.use_gan:
            self.G_gan_loss = self.opts.gan_w * generator_loss(self.D, self.G_y)
            self.total_gen_loss = self.G_gan_loss + self.pu_loss
            self.D_loss = discriminator_loss(self.D, self.input_y, self.G_y)
        else:
            self.total_gen_loss = self.pu_loss

        self.setup_optimizer()
        self.summary_all()

        if self.opts.vis:
            self.visualize_ops = [self.input_x[0], self.G_y[0], self.input_y[0]]
            self.visualize_titles = ['input_x', 'fake_y', 'real_y']

        GNet_param_num = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables('generator')])
        logging.info("===number of parameters in generator: {:.4f} K === ".format(float(GNet_param_num / 1e3)))
        if self.opts.use_gan:
            DNet_param_num = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables('discriminator')])
            logging.info("===number of parameters in discriminator: {:.4f} K ===".format(float(DNet_param_num / 1e3)))
            logging.info(
                "===total number of parameters: {:.4f} K === \n\n".format(float((GNet_param_num + DNet_param_num) / 1e3)))
        logging.info("========== Finish Model Building ========== \n")

    def summary_all(self):
        # summary
        if self.opts.use_gan:
            add_scalar_summary('loss/G_loss', self.G_gan_loss, collection='gen')
        add_scalar_summary('loss/dis_loss', self.dis_loss, collection='gen')
        if self.opts.repulse:
            add_scalar_summary('loss/repulsion_loss', self.repulsion_loss, collection='gen')
        if self.opts.uniform:
            add_scalar_summary('loss/uniform_loss', self.uniform_loss, collection='gen')
        add_scalar_summary('loss/total_gen_loss', self.total_gen_loss, collection='gen')
        self.g_summary_op = tf.summary.merge_all('gen')

        if self.opts.use_gan:
            add_hist_summary('D/true', self.D(self.input_y), collection='dis')
            add_hist_summary('D/fake', self.D(self.G_y), collection='dis')
            add_scalar_summary('loss/D_Y', self.D_loss, collection='dis')
            self.d_summary_op = tf.summary.merge_all('dis')

        if self.opts.vis:
            self.visualize_x_titles = ['input_x', 'fake_y', 'real_y']
            self.visualize_x_ops = [self.input_x[0], self.G_y[0], self.input_y[0]]
            self.image_x_merged = tf.placeholder(tf.float32, shape=[None, 1500, 1500, 1])
            self.image_x_summary = tf.summary.image('Upsampling', self.image_x_merged, max_outputs=1)

    def setup_optimizer(self):
        learning_rate_g = tf.where(
            tf.greater_equal(self.global_step, self.opts.start_decay_step),
            tf.train.exponential_decay(self.opts.base_lr_g, self.global_step - self.opts.start_decay_step,
                                       self.opts.lr_decay_steps, self.opts.lr_decay_rate, staircase=True),
            self.opts.base_lr_g
        )
        learning_rate_g = tf.maximum(learning_rate_g, self.opts.lr_clip)
        add_scalar_summary('learning_rate/learning_rate_g', learning_rate_g, collection='gen')

        # create pre-generator ops
        gen_update_ops = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if op.name.startswith("generator")]
        gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]

        with tf.control_dependencies(gen_update_ops):
            self.G_optimizers = tf.train.AdamOptimizer(learning_rate_g, beta1=self.opts.beta).minimize(
                self.total_gen_loss, var_list=gen_tvars,
                colocate_gradients_with_ops=True,
                global_step=self.global_step)

        if self.opts.use_gan:
            learning_rate_d = tf.where(
                tf.greater_equal(self.global_step, self.opts.start_decay_step),
                tf.train.exponential_decay(self.opts.base_lr_d, self.global_step - self.opts.start_decay_step,
                                           self.opts.lr_decay_steps, self.opts.lr_decay_rate, staircase=True),
                self.opts.base_lr_d
            )
            learning_rate_d = tf.maximum(learning_rate_d, self.opts.lr_clip)
            add_scalar_summary('learning_rate/learning_rate_d', learning_rate_d, collection='dis')

            self.D_optimizers = tf.train.AdamOptimizer(learning_rate_d,
                                                       beta1=self.opts.beta).minimize(self.D_loss,
                                                                                      self.global_step,
                                                                                      var_list=self.D.variables,
                                                                                      name='Adam_D_X')

    def train(self):
        self.allocate_placeholders()
        self.build_model()

        self.sess.run(tf.global_variables_initializer())

        fetchworker = Fetcher(self.opts)
        fetchworker.start()

        self.saver = tf.train.Saver(max_to_keep=None)
        self.writer = tf.summary.FileWriter(self.opts.log_dir, self.sess.graph)

        restore_epoch = 1
        if self.opts.restore:
            restore_epoch, checkpoint_path = model_utils.pre_load_checkpoint(self.opts.log_dir)
            self.saver.restore(self.sess, checkpoint_path)
            tf.assign(self.global_step, restore_epoch * fetchworker.num_batches).eval()
            restore_epoch += 1

        else:
            os.makedirs(os.path.join(self.opts.log_dir, 'plots'))

        with open(os.path.join(self.opts.log_dir, 'args.txt'), 'w') as log:
            for arg in sorted(vars(self.opts)):
                log.write(arg + ': ' + str(getattr(self.opts, arg)) + '\n')  # log of arguments

        step = self.sess.run(self.global_step)
        start = time()
        for epoch in range(restore_epoch, self.opts.max_epochs+1):
            logging.info('**** EPOCH %03d ****\t' % (epoch))
            self.k = epoch / self.opts.max_epochs * 3
            for batch_idx in range(fetchworker.num_batches):

                batch_input_x, batch_input_y, batch_radius, batch_normals = fetchworker.fetch()

                feed_dict = {self.input_x: batch_input_x,
                             self.input_y: batch_input_y,
                             self.pc_radius: batch_radius,
                             self.is_training: True,
                             self.normals: batch_normals}

                # Update D network
                if self.opts.use_gan:
                    _, d_loss, d_summary = self.sess.run([self.D_optimizers, self.D_loss, self.d_summary_op],
                                                         feed_dict=feed_dict)
                    self.writer.add_summary(d_summary, step)

                # Update G network
                for i in range(self.opts.gen_update):
                    _, g_total_loss, summary = self.sess.run(
                        [self.G_optimizers, self.total_gen_loss, self.g_summary_op], feed_dict=feed_dict)
                    self.writer.add_summary(summary, step)

                if step % self.opts.steps_per_print == 0:
                    logging.info('-----------EPOCH %d Step %d:-------------' % (epoch, step))
                    logging.info('  G_loss   : {}'.format(g_total_loss))
                    if self.opts.use_gan:
                        logging.info('  D_loss   : {}'.format(d_loss))
                    logging.info(' Time Cost : {}'.format(time() - start))
                    start = time()

                step += 1
            if (epoch % self.opts.epoch_per_save) == 0:
                self.saver.save(self.sess, os.path.join(self.opts.log_dir, 'model'), epoch)
                print(colored('Model saved at %s' % self.opts.log_dir, 'white', 'on_blue'))

        fetchworker.shutdown()

    def patch_prediction(self, patch_point, normals):
        # normalize the point clouds
        patch_point, centroid, furthest_distance = pc_util.normalize_point_cloud(patch_point)
        patch_point = np.expand_dims(patch_point, axis=0)    # (1, 256, 3)
        normals = np.expand_dims(normals, axis=0)    # (1, 256, 3)
        pred = self.sess.run([self.pred_pc], feed_dict={self.inputs: patch_point, self.normals: normals})
        pred = np.squeeze(centroid + pred * furthest_distance, axis=0)
        return pred

    def pc_prediction(self, pc, normals):    # pc: (2048, 3), normals: (1, 2048, 3)
        # get patch seed from farthestsampling
        points = tf.convert_to_tensor(np.expand_dims(pc, axis=0), dtype=tf.float32)
        start = time()
        print('------------------patch_num_point:', self.opts.patch_num_point)
        seed1_num = int(pc.shape[0] / self.opts.patch_num_point * self.opts.patch_num_ratio)

        # FPS sampling
        seed = farthest_point_sample(seed1_num, points).eval()[0]
        seed_list = seed[:seed1_num]
        print("farthest distance sampling cost", time() - start)
        print("number of patches: %d" % len(seed_list))
        input_list = []
        up_point_list = []

        patches, normal_patches = pc_util.extract_knn_patch(pc[np.asarray(seed_list), :], pc, self.opts.patch_num_point, np.squeeze(normals, axis=0))
        patch_time = 0.

        for i in trange(patches.shape[0]):
            point = patches[i]
            normal = normal_patches[i]
            start = time()
            up_point = self.patch_prediction(point, normal)
            end = time()
            patch_time += end-start

            up_point = np.squeeze(up_point, axis=0)
            input_list.append(point)
            up_point_list.append(up_point)
        return input_list, up_point_list, patch_time/len(patches)

    def test(self):
        self.inputs = tf.placeholder(tf.float32, shape=[1, self.opts.patch_num_point, 3])
        self.normals = tf.placeholder(tf.float32, shape=[1, self.opts.patch_num_point, 3])

        is_training = tf.placeholder_with_default(False, shape=[], name='is_training')
        self.pc_radius = tf.ones(self.opts.batch_size)
        model_cls = get_model_cls(self.opts.model)
        Gen = model_cls(self.opts, is_training, name='generator')

        if self.opts.model == 'punet':
            self.pred_pc = Gen(self.inputs, self.pc_radius)
        elif self.opts.model == 'pufpg':
            self.pred_pc = Gen(self.inputs, self.normals)
        else:
            self.pred_pc = Gen(self.inputs)
        for i in range(round(math.pow(self.opts.up_ratio, 1 / 4)) - 1):
            if self.opts.model =='punet':
                self.pred_pc = Gen(self.pred_pc, self.pc_radius)
            else:
                self.pred_pc = Gen(self.pred_pc)

        saver = tf.train.Saver()
        restore_epoch, checkpoint_path = model_utils.pre_load_checkpoint(self.opts.log_dir)
        print(checkpoint_path)
        saver.restore(self.sess, checkpoint_path)
        
        samples = glob(self.opts.test_data)
        point = pc_util.load(samples[0])

        self.opts.num_point = point.shape[0]
        out_point_num = int(self.opts.num_point * self.opts.up_ratio)

        total_time = 0.
        for point_path in samples:
            pc = pc_util.load(point_path)[:, :3]

            pc, centroid, furthest_distance = pc_util.normalize_point_cloud(pc)

            if self.opts.test_jitter:
                pc = pc_util.jitter_perturbation_point_cloud(pc[np.newaxis, ...], sigma=self.opts.jitter_sigma,
                                                             clip=self.opts.jitter_max)
                pc = pc[0, ...]
            
            normals = compute_normals(pc, self.opts.k1)    # (1, 2048, 3)
            input_list, pred_list, avg_patch_time = self.pc_prediction(pc, normals)

            total_time += avg_patch_time

            pred_pc = np.concatenate(pred_list, axis=0)
            pred_pc = (pred_pc * furthest_distance) + centroid

            pred_pc = np.reshape(pred_pc, [-1, 3])
            path = os.path.join(self.opts.out_folder, point_path.split('/')[-1][:-4] + '.ply')
            idx = farthest_point_sample(out_point_num, pred_pc[np.newaxis, ...]).eval()[0]
            pred_pc = pred_pc[idx, 0:3]

            np.savetxt(path[:-4] + '.xyz', pred_pc, fmt='%.6f')

        logging.info('Average Inference Time: {} ms'.format(total_time / len(samples) * 1000.))
        
    def test_semantic3d(self):
        self.inputs = tf.placeholder(tf.float32, shape=[1, self.opts.patch_num_point, 3])
        self.normals = tf.placeholder(tf.float32, shape=[1, self.opts.patch_num_point, 3])
        is_training = tf.placeholder_with_default(False, shape=[], name='is_training')
        self.pc_radius = tf.ones(self.opts.batch_size)
        model_cls = get_model_cls(self.opts.model)
        Gen = model_cls(self.opts, is_training, name='generator')

        if self.opts.model == 'punet':
            self.pred_pc = Gen(self.inputs, self.pc_radius)
        elif self.opts.model == 'pufpg':
            self.pred_pc = Gen(self.inputs, self.normals)
        else:
            self.pred_pc = Gen(self.inputs)
        for i in range(round(math.pow(self.opts.up_ratio, 1 / 4)) - 1):
            if self.opts.model =='punet':
                self.pred_pc = Gen(self.pred_pc, self.pc_radius)
            else:
                self.pred_pc = Gen(self.pred_pc)

        saver = tf.train.Saver()
        restore_epoch, checkpoint_path = model_utils.pre_load_checkpoint(self.opts.log_dir)
        print(checkpoint_path)
        saver.restore(self.sess, checkpoint_path)
        
        samples = glob(self.opts.test_data)
        
        logging.info('Loading data.')
        
        point_gt = pc_util.load(samples[0])
        point_gt = point_gt[:, :3]
        gt_point_num = point_gt.shape[0]
        input_point_num = math.floor(gt_point_num / self.opts.up_ratio)
        point = pc_util.downsample_points(point_gt, input_point_num)
        self.opts.num_point = point.shape[0]
        out_point_num = int(self.opts.num_point * self.opts.up_ratio)

        total_time = 0.
        
        pc = point
        point_path = samples[0]
        
        logging.info('Normalizing point cloud.')
        pc, centroid, furthest_distance = pc_util.normalize_point_cloud(pc)
        if self.opts.test_jitter:
            pc = pc_util.jitter_perturbation_point_cloud(pc[np.newaxis, ...], sigma=self.opts.jitter_sigma,
                                                         clip=self.opts.jitter_max)
            pc = pc[0, ...]
            
        logging.info('Computing normals.')
        normals = compute_normals(pc, self.opts.k1)    # (1, 2048, 3)
        logging.info('Predicting.')
        input_list, pred_list, avg_patch_time = self.pc_prediction(pc, normals)
        total_time += avg_patch_time
        pred_pc = np.concatenate(pred_list, axis=0)
        pred_pc = (pred_pc * furthest_distance) + centroid
        pred_pc = np.reshape(pred_pc, [-1, 3])
        
        path = os.path.join(self.opts.out_folder, point_path.split('/')[-1][:-4] + '.ply')
        idx = farthest_point_sample(out_point_num, pred_pc[np.newaxis, ...]).eval()[0]
        pred_pc = pred_pc[idx, 0:3]
        color = np.asarray([255 for i in range(pred_pc.shape[0])])
        color = np.expand_dims(color, axis=-1)
        pred_pc = np.concatenate([pred_pc, color], axis=-1)
        np.savetxt(path[:-4] + '.txt', pred_pc, fmt='%.6f')

        logging.info('Average Inference Time: {} ms'.format(total_time / len(samples) * 1000.))
