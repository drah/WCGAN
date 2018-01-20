from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from collections import namedtuple
import json

from module import *
from utils import *

class gan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_h = 64
        self.image_w = 64
        self.input_c_dim = 3
        self.output_c_dim = 3
        self.ngf = 32
        self.ndf = 64
        self.L1_lambda = args.L1_lambda
        self.xent_lambda = args.xent_lambda
        self.dataset_dir = args.dataset_dir
        self.max_seq_len = args.max_seq_len
        self.dataset_dir = args.dataset_dir
        self.max_seq_len = args.max_seq_len

        self.discriminator_i = discriminator_i
        self.discriminator_t = discriminator_t
        self.generator_it = generator_it
        self.generator_ti = generator_ti

        self.data_layer = dataLayer(
                os.path.join(args.dataset_dir, 'Images/'),
                os.path.join(args.dataset_dir, 'list_attr_celeba.txt'),
                os.path.join(args.dataset_dir, 'vocab.txt'),
                self.dataset_dir,
                self.image_h,
                self.image_w,
                self.batch_size,
                n_attributes=self.max_seq_len-2,
                pre_load_images=True)

        OPTIONS = namedtuple('OPTIONS', 'batch_size fine_h fine_w \
                              gf_dim df_dim output_c_dim max_seq_len vocab_size')

        self.options = OPTIONS._make((self.batch_size, self.image_h, self.image_w,
                                      self.ngf, self.ndf, self.output_c_dim,
                                      self.max_seq_len, self.data_layer.vocab_size))

        self._build_model()
        self.saver = tf.train.Saver()

    def _build_model(self):
        opt = self.options

        ### Generator ###

        # placeholder for real image and text
        # (batch_size, 64, 64, 3)
        self.real_image = tf.placeholder(tf.float32,
                [opt.batch_size, self.image_h, self.image_w, self.input_c_dim],
                name='real_image')
        # (batch_size, max_seq_len)
        self.real_text = tf.placeholder(tf.int32,
                [opt.batch_size, opt.max_seq_len],
                name='real_text') # BOS...EOS

        # (batch_size, max_seq_len, vocab_size)
        self.real_text_one_hot = tf.one_hot(self.real_text, opt.vocab_size)

        self.seq_len = None # None = not use

        # [<BOS>, ..]
        # t' = GT(i), (batch_size, len, vocab_size), act=softmax
        self.it_text = self.generator_it(self.real_image, self.real_text_one_hot[:,:-1,:], self.seq_len, opt)

        # i'' = GI(GT(i)), (batch_size, 64, 64, 3), act=tanh
        self.it_ti_image = self.generator_ti(self.it_text, self.seq_len, opt) # ...EOS

        # [.., <EOS>]
        # i' = GI(t), (batch_size, 64, 64, 3), act=tanh
        self.ti_image = self.generator_ti(self.real_text_one_hot[:,1:,:], self.seq_len, opt, reuse=True)

        # t'' = GT(GI(t)), (batch_size, len, vocab_size), act=softmax
        self.ti_it_text = self.generator_it(
                self.ti_image, self.real_text_one_hot[:,:-1,:], self.seq_len, opt, reuse=True) # BOS...

        # DI(i'), scores, (batch_size, 2, 2, 1)
        self.di_fake = self.discriminator_i(self.ti_image, opt)

        # DT(t'), scores, (batch_size, 8, 1, 1)
        self.dt_fake = self.discriminator_t(self.it_text, self.seq_len, opt)

        # adversarial loss with cycle-consistency loss
        # L_G_T
        self.g_loss_it = -tf.reduce_mean(tf.reduce_sum(self.dt_fake, [1,2,3])) \
                          + self.L1_lambda * abs_criterion(self.real_image, self.it_ti_image) \
                          + self.xent_lambda * xent_criterion(self.ti_it_text, self.real_text_one_hot[:,1:,:])

        # L_G_I
        self.g_loss_ti = -tf.reduce_mean(tf.reduce_sum(self.di_fake, [1,2,3])) \
                          + self.L1_lambda * abs_criterion(self.real_image, self.it_ti_image) \
                          + self.xent_lambda * xent_criterion(self.ti_it_text, self.real_text_one_hot[:,1:,:])

        ### Discriminator ###

        # placeholder for generated image and text
        self.fake_image = tf.placeholder(tf.float32,
                [opt.batch_size, self.image_h, self.image_w, self.input_c_dim],
                name='fake_image')

        self.fake_text = tf.placeholder(tf.float32,
                [opt.batch_size, opt.max_seq_len-1, opt.vocab_size],
                name='fake_text') # ...EOS

        self.fake_text_len = None # None = not use

        # DI(i)
        self.di_real = self.discriminator_i(self.real_image, opt, reuse=True)
        # DT(t)
        self.dt_real = self.discriminator_t(self.real_text_one_hot[:,1:,:], self.seq_len, opt, reuse=True)
        # DI(i')
        self.di_fake_sample = self.discriminator_i(self.fake_image, opt, reuse=True)
        # DT(t')
        self.dt_fake_sample = self.discriminator_t(self.fake_text, self.fake_text_len, opt, reuse=True)

        # gradient penalty for DI
        eps = tf.random_uniform([], 0, 1)
        interpolation = self.fake_image * eps + self.real_image * (1.-eps)
        score_interpolation = self.discriminator_i(interpolation, opt, reuse=True)
        interpolation_grad = tf.gradients(score_interpolation, interpolation)
        interpolation_grad = tf.reshape(interpolation_grad, [opt.batch_size, -1])
        gradient_penalty = tf.squared_difference(tf.sqrt(tf.reduce_sum(tf.square(interpolation_grad),1)), 1)

        self.di_loss_real = -tf.reduce_mean(self.di_real)
        self.di_loss_fake = tf.reduce_mean(self.di_fake_sample)

        # DI loss with gradient penalty
        self.di_loss = self.di_loss_real + self.di_loss_fake + 10.0 * tf.reduce_mean(gradient_penalty)

        # gradient penalty for DT
        eps = tf.random_uniform([], 0, 1)
        interpolation = self.fake_text * eps + self.real_text_one_hot[:,1:,:] * (1-eps)
        score_interpolation = self.discriminator_t(interpolation, self.seq_len, opt, reuse=True)
        interpolation_grad = tf.gradients(score_interpolation, interpolation)
        interpolation_grad = tf.reshape(interpolation_grad, [opt.batch_size, -1])
        gradient_penalty = tf.squared_difference(tf.sqrt(tf.reduce_sum(tf.square(interpolation_grad),1)), 1)

        self.dt_loss_real = -tf.reduce_mean(self.dt_real)
        self.dt_loss_fake = tf.reduce_mean(self.dt_fake_sample)

        # DT loss with gradient penalty
        self.dt_loss = self.dt_loss_real + self.dt_loss_fake + 10.0 * tf.reduce_mean(gradient_penalty)


        ### the followings are for testing ###

        self.test_i = tf.placeholder(tf.float32,
                [self.batch_size, self.image_h, self.image_w, self.input_c_dim],
                name='test_i')
        self.test_t = tf.placeholder(tf.int32,
                [self.batch_size, opt.max_seq_len],
                name='test_t')
        self.test_t_one_hot = tf.one_hot(self.test_t, opt.vocab_size)

        self.test_it = self.generator_it(self.test_i, self.test_t_one_hot[:,:-1,:], None, opt, reuse=True)
        self.test_ti = self.generator_ti(self.test_t_one_hot[:,1:,:], self.seq_len, opt, reuse=True)


        # trainable
        t_vars = tf.trainable_variables()

        # summarize
        self.g_it_sum = tf.summary.scalar("g_loss_it", self.g_loss_it)
        self.g_ti_sum = tf.summary.scalar("g_loss_ti", self.g_loss_ti)

        self.di_loss_sum = tf.summary.scalar("di_loss", self.di_loss)
        self.dt_loss_sum = tf.summary.scalar("dt_loss", self.dt_loss)

        self.di_loss_real_sum = tf.summary.scalar("di_loss_real", self.di_loss_real)
        self.di_loss_fake_sum = tf.summary.scalar("di_loss_fake", self.di_loss_fake)

        self.dt_loss_real_sum = tf.summary.scalar("dt_loss_real", self.dt_loss_real)
        self.dt_loss_fake_sum = tf.summary.scalar("dt_loss_fake", self.dt_loss_fake)

        self.di_sum = tf.summary.merge(
            [self.di_loss_sum, self.di_loss_real_sum, self.di_loss_fake_sum]
        )
        self.dt_sum = tf.summary.merge(
            [self.dt_loss_sum, self.dt_loss_real_sum, self.dt_loss_fake_sum]
        )

        self.di_vars = [var for var in t_vars if 'discriminator_i' in var.name]
        self.dt_vars = [var for var in t_vars if 'discriminator_t' in var.name]
        self.g_vars_it = [var for var in t_vars if 'generator_it' in var.name]
        self.g_vars_ti = [var for var in t_vars if 'generator_ti' in var.name]
        self.di_vars_sum = [tf.summary.histogram(v.name, v) for v in self.di_vars]
        self.dt_vars_sum = [tf.summary.histogram(v.name, v) for v in self.dt_vars]
        self.g_vars_ti_sum = [tf.summary.histogram(v.name, v) for v in self.g_vars_ti]
        self.g_vars_it_sum = [tf.summary.histogram(v.name, v) for v in self.g_vars_it]

        self.g_it_sum = tf.summary.merge([self.g_it_sum]+self.g_vars_it_sum)
        self.g_ti_sum = tf.summary.merge([self.g_ti_sum]+self.g_vars_ti_sum+[tf.summary.image('ti_image', self.ti_image), tf.summary.image('real_image', self.real_image)])
        self.di_sum = tf.summary.merge([self.di_sum]+self.di_vars_sum)
        self.dt_sum = tf.summary.merge([self.dt_sum]+self.dt_vars_sum)
        for var in t_vars: print(var.name)

    def train(self, args):
        # optimizer graph
        optimizer = tf.train.AdamOptimizer(args.lr, beta1=args.beta1)
        gvs = optimizer.compute_gradients(self.di_loss, self.di_vars)
        gvs = [(tf.clip_by_value(g, -5.0, 5.0), v) for g,v in gvs]
        self.di_optim = optimizer.apply_gradients(gvs)

        optimizer = tf.train.AdamOptimizer(args.lr, beta1=args.beta1)
        gvs = optimizer.compute_gradients(self.dt_loss, self.dt_vars)
        gvs = [(tf.clip_by_value(g, -5.0, 5.0), v) for g,v in gvs]
        self.dt_optim = optimizer.apply_gradients(gvs)

        optimizer = tf.train.AdamOptimizer(args.lr, beta1=args.beta1)
        gvs = optimizer.compute_gradients(self.g_loss_it, self.g_vars_it)
        gvs = [(tf.clip_by_value(g, -5.0, 5.0), v) for g,v in gvs if g is not None]
        self.g_it_optim = optimizer.apply_gradients(gvs)

        optimizer = tf.train.AdamOptimizer(args.lr, beta1=args.beta1)
        gvs = optimizer.compute_gradients(self.g_loss_ti, self.g_vars_ti)
        gvs = [(tf.clip_by_value(g, -5.0, 5.0), v) for g,v in gvs]
        self.g_ti_optim = optimizer.apply_gradients(gvs)

        # initialize
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        batch_idxs = self.data_layer.train_size // self.batch_size
        for epoch in range(args.from_epoch, args.epoch):

            for idx in range(0, batch_idxs):
                images, texts = self.data_layer.get_batch(self.batch_size)

                # Forward G network
                fake_image, fake_text = self.sess.run([self.ti_image, self.it_text],
                        feed_dict={
                            self.real_text: texts,
                            self.real_image: images})

                # Update G network: GT
                _, summary_str = self.sess.run([self.g_it_optim, self.g_it_sum],
                        feed_dict={
                            self.real_text: texts,
                            self.real_image: images})
                self.writer.add_summary(summary_str, counter)

                # Update D network: DT
                _, summary_str = self.sess.run([self.dt_optim, self.dt_sum],
                        feed_dict={
                            self.real_text: texts,
                            self.real_image: images,
                            self.fake_text: fake_text})
                self.writer.add_summary(summary_str, counter)

                # Update G network: GI
                _, summary_str = self.sess.run([self.g_ti_optim, self.g_ti_sum],
                        feed_dict={
                            self.real_text: texts,
                            self.real_image: images})
                self.writer.add_summary(summary_str, counter)

                # Update D network: DI
                _, summary_str = self.sess.run([self.di_optim, self.di_sum],
                        feed_dict={
                            self.real_text: texts,
                            self.real_image: images,
                            self.fake_image: fake_image})
                self.writer.add_summary(summary_str, counter)

                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" \
                       % (epoch, idx, batch_idxs, time.time() - start_time)))

                if np.mod(counter, 100) == 1:
                    self.sample_model(args.sample_dir, epoch, idx)

                if np.mod(counter, 1000) == 2:
                    self.save(args.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name = "cyclegan.model"
        model_dir = "%s_%s_%s" % (self.dataset_dir, self.image_h, self.image_w)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s_%s" % (self.dataset_dir, self.image_h, self.image_w)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, sample_dir, epoch, idx):
        images, texts = self.data_layer.get_val_batch(self.batch_size)
        fake_image, fake_text = self.sess.run([self.ti_image, self.it_text],
                feed_dict={
                    self.real_image: images,
                    self.real_text: texts})
        save_images(fake_image, [self.batch_size, 1],
                    './{}/{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
        save_texts(fake_text, self.data_layer.vocab,
                    './{}/{:02d}_{:04d}'.format(sample_dir, epoch, idx))
        try:
            get_accuracy('./{}/{:02d}_{:04d}'.format(sample_dir, epoch, idx),
                    self.data_layer.val_texts_save_path)
        except:
            print("val_texts not found")

    def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        index_path = os.path.join(args.test_dir, '{0}_index.html'.format(args.which_direction))
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        accuracy_list = []
        n_test = self.data_layer.test_size // self.batch_size
        for i in range(n_test):
            images, texts = self.data_layer.get_test_batch(self.batch_size)
            fake_images, fake_texts = self.sess.run([self.test_ti, self.test_it],
                    {self.test_t: texts, self.test_i: images})

            real_text_path = os.path.join(args.test_dir, "real_%d_%d.txt" %
                    (i*self.batch_size+1,(i+1)*self.batch_size))
            fake_text_path = os.path.join(args.test_dir, "fake_%d_%d.txt" %
                    (i*self.batch_size+1,(i+1)*self.batch_size))

            save_texts_from_ids(texts, self.data_layer.vocab, real_text_path)
            save_texts(fake_texts, self.data_layer.vocab, fake_text_path)
            rc = open(real_text_path)
            fc = open(fake_text_path)

            if i < 2:
                for j, (ri, fi) in enumerate(zip(images, fake_images)):
                    real_image_path = "real_%d_%d.jpg" % (i,j)
                    fake_image_path = "fake_%d_%d.jpg" % (i,j)

                    save_images(
                            np.array(ri[np.newaxis,:,:,:]),
                            [1, 1],
                            os.path.join(args.test_dir, real_image_path))
                    save_images(
                            fi[np.newaxis,:,:,:],
                            [1, 1],
                            os.path.join(args.test_dir, fake_image_path))

                    index.write("<td>%s</td>" % os.path.basename(real_image_path))
                    index.write("<td><img src='%s'></td>" % (
                            real_image_path if os.path.isabs(real_image_path) else (
                                    '.' + os.path.sep + real_image_path)))
                    index.write("<td>%s</td>" % fc.readline())
                    index.write("</tr>")

                    index.write("<td>real_text_to_image</td>")
                    index.write("<td>%s</td>" % rc.readline())
                    index.write("<td><img src='%s'></td>" % (
                            fake_image_path if os.path.isabs(fake_image_path) else (
                                    '.' + os.path.sep + fake_image_path)))
                    index.write("</tr>")

                rc.close()
                fc.close()

            acc = get_accuracy(fake_text_path, real_text_path)
            accuracy_list.append(acc)
            index.write("<td>accuracy of texts in batch %d: %f</td>" % (i, acc))
            index.write("</tr>")

        index.write("<td>accuracy of all texts: %f</td>" % np.mean(accuracy_list))
        index.write("</tr>")

