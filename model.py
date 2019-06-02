import os
import random

from glob import glob
from time import time
from typing import Optional
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from .utils import ImageDataLoader

from .tfutils import dense
from .tfutils import conv2d
from .tfutils import sub_pixel_conv2d

from .losses import generator_loss
from .losses import discriminator_loss

from .vgg19 import VGG19


class ESRGAN:
    def __init__(self,
                 sess: tf.Session,
                 dataset_path: str = "D://DataSet/SR/DIV2K/",
                 dataset_type: str = "DIV2K",
                 input_shape: Tuple = (128, 128, 3),
                 batch_size: int = 1,
                 patch_size: int = 16,
                 n_iter: int = int(5e5),
                 n_warm_up_iter: int = int(5e4),
                 n_feats: int = 64,
                 n_res_blocks: int = 23,
                 use_sn: bool = True,
                 gan_type: str = "lsgan",
                 use_ra: bool = True,
                 lambda_gp: float = 10.,
                 weight_rec_loss: float = 1e-2,
                 weight_adv_loss: float = 5e-3,
                 weight_perceptual_loss: float = 1e0,
                 use_perceptual_loss: bool = True,
                 vgg19_model_path: str = "./imagenet-vgg-verydeep-19.mat",
                 n_critic: int = 1,
                 d_lr: float = 1e-4,  # 2e-4
                 g_lr: float = 1e-4,  # 5e-5
                 lr_schedule_steps: tuple = (int(5e4), int(1e5), int(2e5), int(3e5)),
                 lr_decay_factor: float = .5,
                 beta1: float = .9,
                 beta2: float = .999,
                 grad_clip_norm: float = 1.,
                 save_freq: int = int(1e3),
                 log_freq: int = int(5e2),
                 checkpoint_dir: str = "./checkpoint/",
                 model_name: str = "ESRGAN",
                 n_threads: int = 8,
                 seed: int = 13371337):
        self.sess = sess
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.n_iter = n_iter
        self.n_warm_up_iter = n_warm_up_iter

        assert self.input_shape[0] == self.input_shape[1]
        assert self.input_shape[0] in [96, 128, 192]
        assert self.patch_size in [4, 16]

        self.n_feats = n_feats
        self.n_res_blocks = n_res_blocks
        self.use_sn = use_sn
        self.gan_type = gan_type
        self.use_ra = use_ra
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.d_lr = d_lr
        self.g_lr = g_lr
        self.lr_schedule_steps = lr_schedule_steps
        self.lr_decay_factor = lr_decay_factor
        self.beta1 = beta1
        self.beta2 = beta2
        self.grad_clip_norm = grad_clip_norm

        self.save_freq = save_freq
        self.log_freq = log_freq
        self.checkpoint_dir = checkpoint_dir

        self.model_name = model_name
        self.n_threads = n_threads
        self.seed = seed

        self.data = None
        self.inputs = None
        self.iterators = None
        self.n_samples: int = 800
        self.data_loader: Optional[ImageDataLoader] = None
        self.d_opt = None
        self.g_opt = None
        self.g_rec_opt = None

        self.d_adv_loss = 0.
        self.g_adv_loss = 0.
        self.d_loss = 0.
        self.g_loss = 0.
        self.rec_loss = 0.
        self.perceptual_loss = 0.
        self.weight_rec_loss = weight_rec_loss
        self.weight_adv_loss = weight_adv_loss

        self.vgg19 = None
        self.vgg_mean = [103.939, 116.779, 123.68]

        self.vgg19_model_path = vgg19_model_path
        self.use_perceptual_loss = use_perceptual_loss
        self.weight_perceptual_loss = weight_perceptual_loss

        self.saver = None
        self.best_saver = None

        self.merged = None
        self.writer = None
        self.generated_image = None

        # reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        # setup directories
        self.setup()

        # building the architecture
        self.build_model()

    def setup(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.dataset_type == "DIV2K":
            _lr_path = self.dataset_path + "/DIV2K_train_LR_bicubic/X4/"
            _hr_path = self.dataset_path + "/DIV2K_train_HR/"
        elif self.dataset_type == "Flickr2K":
            _lr_path = self.dataset_path + "/Flickr2K_LR_bicubic/X4/"
            _hr_path = self.dataset_path + "/Flickr2K_HR/"
        else:
            raise NotImplementedError("[-] not supported dataset yet :(")

        _lr_paths = sorted(glob(os.path.join(_lr_path, "*.png")))
        _hr_paths = sorted(glob(os.path.join(_hr_path, "*.png")))
        self.data = [(lr_p, hr_p) for lr_p, hr_p in zip(_lr_paths, _hr_paths)]

        print("[*] total {} images".format(len(self.data)))

    @staticmethod
    def summary():
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if "discriminator" in var.name]
        g_vars = [var for var in t_vars if "generator" in var.name]

        slim.model_analyzer.analyze_vars(d_vars, print_info=True)
        slim.model_analyzer.analyze_vars(g_vars, print_info=True)

    @staticmethod
    def res_block(x_init, ch: int, kernel: int = 3, pad: int = 1,
                  use_bias: bool = True, sn: bool = True,
                  scope: str = "res_block"):
        with tf.variable_scope(scope):
            x = conv2d(x_init, ch, kernel=kernel, stride=1, pad=pad,
                       use_bias=use_bias, sn=sn,
                       scope="conv2d_1")
            x = tf.nn.leaky_relu(x, alpha=.2)

            x = conv2d(x, ch, kernel=kernel, stride=1, pad=pad,
                       use_bias=use_bias, sn=sn,
                       scope="conv2d_2")
            return x + x_init

    @staticmethod
    def dense_block(x_init, ch: int, n_blocks: int = 4,
                    use_bias: bool = True, sn: bool = True,
                    scope: str = "dense_block"):
        with tf.variable_scope(scope):
            x_concat = [x_init]

            for i in range(n_blocks):
                x = tf.concat(x_concat, axis=-1)

                x = conv2d(x, ch // 2, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn,
                           scope="conv2d_{}".format(i))
                x = tf.nn.leaky_relu(x, alpha=.2)
                x_concat.append(x)

            x = tf.concat(x_concat, axis=-1)

            x = conv2d(x, ch, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn,
                       scope="conv2d_proj".format(i))
            return x

    def residual_dense_block(self, x_init, ch: int, n_blocks: int = 3, beta: float = .2,
                             use_bias: bool = True, sn: bool = True,
                             scope: str = "residual_dense_block"):
        with tf.variable_scope(scope):
            x = x_init
            for i in range(n_blocks):
                x = x + self.dense_block(x, ch, use_bias=use_bias, sn=sn,
                                         scope="dense_block_{}".format(i + 1)) * beta
            x = x_init + x * beta
            return x

    @staticmethod
    def res_block_down(x_init, ch: int,
                       use_bias: bool = True, sn: bool = True,
                       scope: str = "res_block_down"):
        with tf.variable_scope(scope):
            with tf.variable_scope("residual_1"):
                x = conv2d(x_init, ch, kernel=3, stride=2, pad=1, use_bias=use_bias, sn=sn)
                x = tf.nn.leaky_relu(x, alpha=.2)

            with tf.variable_scope("residual_2"):
                x = conv2d(x, ch, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
                x = tf.nn.leaky_relu(x, alpha=.2)

            with tf.variable_scope("skip_conn"):
                x_init = conv2d(x_init, ch, kernel=3, stride=2, pad=1, use_bias=use_bias, sn=sn)

            return x + x_init

    def generator(self, x_lr, reuse: bool = False):
        ch: int = self.n_feats

        with tf.variable_scope("generator_v3", reuse=reuse):
            x = conv2d(x_lr, self.input_shape[-1], kernel=3, stride=1, pad=1,
                       use_bias=False, sn=self.use_sn, scope="conv2d_init")
            x_lsc = x

            for nb in range(self.n_res_blocks):
                x = self.residual_dense_block(x, ch=ch, use_bias=True, sn=self.use_sn,
                                              scope="RRDB-{}".format(nb + 1))

            x = conv2d(x, ch, kernel=3, stride=1, pad=1, use_bias=True, scope="conv2d_trunk")

            x += x_lsc

            # up-sampling
            scale: int = int(np.log2(self.patch_size))
            for i in range(scale):
                x = conv2d(x, ch * 4, kernel=3, stride=1, pad=1, scope="conv2d_up_{}".format(i))
                x = sub_pixel_conv2d(x, f=None, s=scale)
                x = tf.nn.leaky_relu(x, alpha=.2)

            x = conv2d(x, ch, kernel=3, stride=1, pad=1,
                       use_bias=True, sn=self.use_sn, scope="conv2d_hr")
            x = tf.nn.leaky_relu(x, alpha=.2)

            x = conv2d(x, self.input_shape[-1], kernel=3, stride=1, pad=1,
                       use_bias=True, sn=self.use_sn, scope="conv2d_last")
            x = tf.sigmoid(x)
            return x

    def discriminator(self, x_hr, reuse: bool = False):
        """ original paper maybe used VGG-like model, but i just custom-ed it"""
        with tf.variable_scope("discriminator", reuse=reuse):
            ch: int = self.n_feats

            x = x_hr

            x = self.res_block_down(x, ch * 1, use_bias=True, sn=self.use_sn,
                                    scope="res_block_down_1")

            x = self.res_block_down(x, ch * 2, use_bias=True, sn=self.use_sn,
                                    scope="res_block_down_2")

            x = self.res_block_down(x, ch * 4, use_bias=True, sn=self.use_sn,
                                    scope="res_block_down_3")

            x = self.res_block_down(x, ch * 8, use_bias=True, sn=self.use_sn,
                                    scope="res_block_down_4")

            x = self.res_block_down(x, ch * 16, use_bias=True, sn=self.use_sn,
                                    scope="res_block_down_5")

            x = self.res_block(x, ch * 16, use_bias=False, sn=self.use_sn, scope="res_block")
            x = tf.nn.leaky_relu(x, alpha=.2)

            x = tf.layers.flatten(x)

            x = dense(x, units=128, sn=self.use_sn, scope="dense_1")
            x = tf.nn.leaky_relu(x, alpha=.2)

            x = dense(x, units=1, sn=self.use_sn, scope="disc")
            return x

    def gradient_penalty(self, real, fake):
        if self.gan_type.__contains__("dragan"):
            _eps = tf.random.uniform(shape=tf.shape(real), minval=0., maxval=1.)

            _, _var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            _std = tf.sqrt(_var)

            fake = real + .5 * _std * _eps

        alpha = tf.random_uniform(shape=(self.batch_size, 1, 1, 1), minval=0., maxval=1.)
        interp = real + alpha * (fake - real)

        d_interp = self.discriminator(interp, reuse=True)

        grad = tf.gradients(d_interp, interp)[0]
        grad_norm = tf.norm(tf.layers.flatten(grad), axis=1)

        gp: float = 0.
        if self.gan_type == "wgan-lp":
            gp = self.lambda_gp * tf.reduce_mean(tf.square(tf.maximum(0., grad_norm - 1.)))
        elif self.gan_type == "wgan-gp" or self.gan_type == "dragan":
            gp = self.lambda_gp * tf.reduce_mean(tf.square(grad_norm - 1.))
        return gp

    def build_data_loader(self):
        self.data_loader = ImageDataLoader(patch_shape=self.input_shape,
                                           patch_size=self.patch_size)

        inputs = tf.data.Dataset.from_tensor_slices(self.data)
        inputs = inputs. \
            apply(tf.data.experimental.shuffle_and_repeat(buffer_size=self.n_samples, seed=self.seed)). \
            apply(tf.data.experimental.map_and_batch(self.data_loader.pre_processing,
                                                     batch_size=self.batch_size,
                                                     num_parallel_batches=self.n_threads,
                                                     drop_remainder=True)). \
            apply(tf.data.experimental.prefetch_to_device(device='/gpu:0', buffer_size=self.patch_size))

        self.iterators = inputs.make_one_shot_iterator()
        self.inputs = self.iterators.get_next()

    def build_vgg19_model(self, x, reuse: bool = False):
        with tf.variable_scope("vgg19", reuse=reuse):
            x = tf.cast(x * 255., dtype=tf.float32)

            r, g, b = tf.split(x, 3, 3)
            bgr = tf.concat([b - self.vgg_mean[0],
                             g - self.vgg_mean[1],
                             r - self.vgg_mean[2]], axis=3)
            self.vgg19 = VGG19(bgr)

            net = self.vgg19.vgg19_net['conv5_4']
            return net

    def build_model(self):
        self.build_data_loader()

        x_lr, x_hr = self.inputs

        g_fake = self.generator(x_lr)

        # PatchGAN-wise
        d_fake = self.discriminator(g_fake)
        d_real = self.discriminator(x_hr, reuse=True)

        # losses
        self.d_adv_loss = discriminator_loss(self.gan_type, d_real, d_fake, use_ra=self.use_ra)
        self.g_adv_loss = generator_loss(self.gan_type, d_real, d_fake, use_ra=self.use_ra)

        gp = 0.
        if self.gan_type.__contains__('wgan') or self.gan_type == 'dragan':
            gp = self.gradient_penalty(real=x_hr, fake=g_fake)

        self.d_loss = self.d_adv_loss + gp
        self.rec_loss = tf.reduce_mean(tf.abs(g_fake - x_hr))
        self.g_loss = self.weight_adv_loss * self.g_adv_loss + self.weight_rec_loss * self.rec_loss

        if self.use_perceptual_loss:
            x_real = tf.image.resize_images(x_hr, size=(224, 224), align_corners=False)
            x_fake = tf.image.resize_images(g_fake, size=(224, 224), align_corners=False)

            vgg19_real = self.build_vgg19_model(x_real)
            vgg19_fake = self.build_vgg19_model(x_fake, reuse=True)

            self.perceptual_loss = tf.reduce_mean(tf.square(vgg19_real - vgg19_fake))

            self.g_loss += self.weight_perceptual_loss * self.perceptual_loss

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if "discriminator" in var.name]
        g_vars = [var for var in t_vars if "generator" in var.name]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            d_opt = tf.train.AdamOptimizer(self.d_lr, beta1=self.beta1, beta2=self.beta2)
            d_grads, d_vars = zip(*d_opt.compute_gradients(self.d_loss, var_list=d_vars))
            d_grads = [tf.clip_by_norm(grad, self.grad_clip_norm) for grad in d_grads]
            self.d_opt = d_opt.apply_gradients(zip(d_grads, d_vars))

            g_opt = tf.train.AdamOptimizer(self.g_lr, beta1=self.beta1, beta2=self.beta2)
            g_grads, g_vars = zip(*g_opt.compute_gradients(self.g_loss, var_list=g_vars))
            g_grads = [tf.clip_by_norm(grad, self.grad_clip_norm) for grad in g_grads]
            self.g_opt = g_opt.apply_gradients(zip(g_grads, g_vars))

            g_rec_opt = tf.train.AdamOptimizer(self.g_lr, beta1=self.beta1, beta2=self.beta2)
            g_rec_grads, g_rec_vars = zip(*g_rec_opt.compute_gradients(self.rec_loss, var_list=g_vars))
            g_rec_grads = [tf.clip_by_norm(grad, self.grad_clip_norm) for grad in g_rec_grads]
            self.g_rec_opt = g_rec_opt.apply_gradients(zip(g_rec_grads, g_rec_vars))

        # summaries
        tf.summary.scalar("loss/d_adv_loss", self.d_adv_loss)
        tf.summary.scalar("loss/g_adv_loss", self.g_adv_loss)
        tf.summary.scalar("loss/rec_loss", self.rec_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)
        if self.use_perceptual_loss:
            tf.summary.scalar("loss/perceptual_loss", self.perceptual_loss)

        tf.summary.image("real/x_lr", x_lr, max_outputs=1)
        tf.summary.image("real/x_hr", x_hr, max_outputs=1)
        tf.summary.image("fake/gen", g_fake, max_outputs=1)

        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=5)
        self.best_saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter(self.checkpoint_dir, self.sess.graph)

    def load_ckpt(self) -> int:
        global_step: int = 1
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            print("[+] global step : {} successfully loaded!".format(global_step))
        else:
            print('[-] No checkpoint file found')
        return global_step

    def train(self):
        self.sess.run(tf.global_variables_initializer())

        start_time = time()

        global_step = self.load_ckpt()

        _g_loss: float = -1.
        rec_loss: float = 0.
        best_rec_loss: float = 5e-2
        for idx in range(global_step, self.n_iter + 1):
            if idx > self.n_warm_up_iter:
                # update D network
                _, d_loss = self.sess.run([self.d_opt, self.d_loss])

                # update G network
                if idx % self.n_critic == 0:
                    _, rec_loss, g_loss = self.sess.run([self.g_opt, self.rec_loss, self.g_loss])
                    _g_loss = g_loss

                print("[*] Iter [{:05d}/{:05d}] time : {:.4f}, "
                      "d_loss : {:.6f}, rec_loss: {:.6f}, g_loss : {:.6f}".
                      format(idx, self.n_iter, time() - start_time, d_loss, rec_loss, _g_loss))
            else:
                # update G network
                _, rec_loss = self.sess.run([self.g_rec_opt, self.rec_loss])

                print("[*] Iter [{:05d}/{:05d}] time : {:.4f}, rec_loss : {:.6f}".
                      format(idx, self.n_iter, time() - start_time, rec_loss))

            if idx % self.log_freq == 0:
                summary = self.sess.run(self.merged)
                self.writer.add_summary(summary, idx)

            if idx % self.save_freq == 0:
                self.saver.save(self.sess, self.checkpoint_dir, idx)

                if rec_loss < best_rec_loss and idx <= self.n_warm_up_iter:
                    print("[*] rec loss is improved from {:.6f} to {:.6f}".format(best_rec_loss, rec_loss))
                    self.best_saver.save(self.sess, self.checkpoint_dir + "/best_rec_loss", idx)
                    best_rec_loss = rec_loss

        print("[*] took {}s".format(time() - start_time))

        self.saver.save(self.sess, self.checkpoint_dir, idx)

    def test(self):
        self.sess.run(tf.global_variables_initializer())

    def __str__(self) -> str:
        return self.model_name
