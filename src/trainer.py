import os
import time

import cv2
import numpy as np
import tensorflow as tf

from .data_generator import DataGenerator
from .sagan_models import create_discriminator, create_generator


class Trainer(object):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.epoch = config.epoch
        self.epoch_start = 0
        self.batch_size = config.batch_size
        self.print_freq = config.print_freq
        self.save_freq = config.save_freq
        self.gpl = config.gpl
        self.sample_num = config.sample_num
        self.data_path = config.data_path
        self.checkpoint_dir = config.checkpoint_dir
        self.result_dir = config.result_dir
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.image_size = config.image_size
        self.z_dim = config.z_dim
        self.g_pretrained_model = config.g_pretrained_model
        self.d_pretrained_model = config.d_pretrained_model

        # initial models
        self.g = create_generator(
            image_size=self.image_size,
            z_dim=self.z_dim,
            filters=config.g_conv_filters,
            kernel_size=config.g_conv_kernel_size,
        )

        self.d = create_discriminator(
            image_size=self.image_size, filters=config.d_conv_filters, kernel_size=config.d_conv_kernel_size
        )

        # initial optimizers
        self.g_opt = tf.optimizers.get(config.g_opt)
        self.g_opt.learning_rate = config.g_lr
        if isinstance(self.g_opt, tf.optimizers.Adam):
            self.g_opt.beta_1 = config.beta1
            self.g_opt.beta_2 = config.beta2

        self.d_opt = tf.optimizers.get(config.d_opt)
        self.d_opt.learning_rate = config.d_lr
        if isinstance(self.g_opt, tf.optimizers.Adam):
            self.d_opt.beta_1 = config.beta1
            self.d_opt.beta_2 = config.beta2

        if config.restore_model:
            self.restore_model()
        else:
            self.load_pretrained_model()

        self.data_generator = self.get_data_generator(config.category_file)

    @staticmethod
    def w_loss(y_true, y_pred):
        return tf.reduce_mean(y_true * y_pred)

    def gradient_penalty(self, real, fake, label):
        alpha = tf.random.uniform(shape=[len(real), 1, 1, 1], minval=0.0, maxval=1.0)
        interpolated = alpha * real + (1 - alpha) * fake

        with tf.GradientTape() as tape_p:
            tape_p.watch(interpolated)
            logit = self.d([interpolated, label])

        grad = tape_p.gradient(logit, interpolated)
        grad_norm = tf.norm(tf.reshape(grad, (self.batch_size, -1)), axis=1)

        return self.gpl * tf.reduce_mean(tf.square(grad_norm - 1.0))

    def get_label_map(self, path, exclude_categories=["0"], image_extention_flag=True):
        new_data = {}
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = f.readlines()
                    for row in data:
                        row = row.replace("\n", "")
                        c = row.split(",")
                        if c[1] not in exclude_categories:
                            if image_extention_flag:
                                new_data[c[0]] = int(c[1])
                            else:
                                new_data[c[0][:-4]] = int(c[1])
            else:
                print("file " + str(path) + " does not exist")
        except Exception as e:
            print(e)
        return new_data

    def get_data_generator(self, category_file):
        label_map = self.get_label_map(category_file)
        images = []
        labels = []
        cat_count = {}
        for dirname, dirnames, filenames in os.walk(self.data_path):
            for f in filenames:
                image_path = os.path.join(dirname, f)
                image_label = label_map[f]
                if image_label in cat_count.keys() and cat_count[image_label] >= 100:
                    continue
                else:
                    images.append(image_path)
                    labels.append(image_label)
                    if image_label not in cat_count.keys():
                        cat_count[image_label] = 1
                    else:
                        cat_count[image_label] += 1

        print(len(images))
        self.nbatch = int(np.ceil(len(images) / self.batch_size))
        return DataGenerator(images, labels, image_size=self.image_size, batch_size=self.batch_size)

    def load_pretrained_model(self):
        if self.g_pretrained_model:
            self.g.load_weights(self.g_pretrained_model)
        if self.d_pretrained_model:
            self.d.load_weights(self.d_pretrained_model)

    def restore_model(self):
        g_latest = tf.train.latest_checkpoint(os.path.join(self.checkpoint_dir, "g"))
        self.g.load_weights(g_latest)
        d_latest = tf.train.latest_checkpoint(os.path.join(self.checkpoint_dir, "d"))
        self.d.load_weights(d_latest)
        g_latest = g_latest.replace("\\", "/")
        self.epoch_start = int(g_latest.split("/")[-1][3:].split(".")[0])

    def save_models(self, epoch):
        self.g.save_weights(os.path.join(self.checkpoint_dir, "g", "cp-{:06d}.ckpt".format(epoch)))
        self.d.save_weights(os.path.join(self.checkpoint_dir, "d", "cp-{:06d}.ckpt".format(epoch)))

    def sample(self):
        z = tf.random.truncated_normal(shape=(self.sample_num, self.z_dim), dtype=tf.float32)
        # c = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.int32)
        # c = tf.constant([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=tf.int32)
        # c = tf.constant([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=tf.int32)
        c = tf.constant([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], dtype=tf.int32)
        # c = tf.constant([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], dtype=tf.int32)
        # c = tf.constant([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=tf.int32)
        c = tf.reshape(c, [self.sample_num, 1])
        return self.g([z, c])[0].numpy()

    def save_samples(self, epoch):
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        img = np.reshape(self.sample(), (-1, self.image_size, 3))

        cv2.imwrite(os.path.join(self.sample_dir, "sp-{:06d}.jpg".format(epoch)), ((img + 1) * 127).astype(np.uint8))

    def train_discriminator_step(self, real_img, noise_z):
        with tf.GradientTape() as tape_d:

            fake_img, g_attn1, g_attn2 = self.g([noise_z, real_img[1]], training=False)
            real_pred, r_attn1, r_attn2 = self.d(real_img, training=True)
            fake_pred, f_attn1, f_attn2 = self.d([fake_img, real_img[1]], training=True)

            y_true = tf.ones(shape=tf.shape(real_pred), dtype=tf.float32)

            real_loss = self.w_loss(-y_true, real_pred)
            fake_loss = self.w_loss(y_true, fake_pred)

            gp = self.gradient_penalty(real_img[0], fake_img, real_img[1])

            total_loss = real_loss + fake_loss + gp

        gradients = tape_d.gradient(total_loss, self.d.trainable_variables)
        self.d_opt.apply_gradients(zip(gradients, self.d.trainable_variables))

        return total_loss, gp

    def train_generator_step(self, noise_z, noise_c):
        with tf.GradientTape() as tape_g:
            fake_img, g_att1, g_att2 = self.g([noise_z, noise_c], training=True)
            fake_pred, d_att1, d_att2 = self.d([fake_img, noise_c], training=False)

            g_loss = self.w_loss(fake_pred, -tf.ones(shape=tf.shape(fake_pred), dtype=tf.float32))

            gradients = tape_g.gradient(g_loss, self.g.trainable_variables)
            self.g_opt.apply_gradients(zip(gradients, self.g.trainable_variables))

        return g_loss

    def train(self):
        print("Start Training")
        print("epoch: {}".format(self.epoch))

        for epoch in range(self.epoch_start, self.epoch_start + self.epoch):
            epoch_start_time = time.time()

            for image_batch in self.data_generator.dataset:
                z = tf.random.truncated_normal(shape=(self.batch_size, self.z_dim), dtype=tf.float32)
                c = tf.random.uniform([self.batch_size, 1], minval=0, maxval=4, dtype=tf.dtypes.int32)
                d_loss, gp_loss = self.train_discriminator_step(image_batch, z)
                g_loss = self.train_generator_step(z, c)

            if (epoch % self.print_freq) == 0:
                print(
                    "epoch {}/{} ({:.2f} sec):, d_loss {:.4f}, gp_loss {:.4f}, g_loss {:.4f}".format(
                        epoch,
                        self.epoch_start + self.epoch,
                        time.time() - epoch_start_time,
                        d_loss.numpy(),
                        gp_loss.numpy(),
                        g_loss.numpy(),
                    )
                )

            if (epoch % self.save_freq) == 0:
                self.save_models(epoch)
                self.save_samples(epoch)

    def test(self):
        for j in range(5):
            sameples = self.sample()

            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)

            for i, s in enumerate(sameples):
                image1 = cv2.resize(
                    ((s[:, :, ::-1] + 1) * 127).astype(np.uint8), (360, 576), interpolation=cv2.INTER_CUBIC
                )
                cv2.imwrite(os.path.join(self.result_dir, "result-{:d}-{:02d}.jpg".format(j, i)), image1)
