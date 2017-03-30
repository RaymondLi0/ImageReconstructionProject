import glob
import os
import time

import PIL.Image as Image
import numpy as np
import params
import tensorflow as tf
from termcolor import cprint
from tqdm import tqdm

from utils import weight_variable, bias_variable, conv2d, uconv2d, max_pool_2x2, create_dir, avg_pool_2x2


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        tf.summary.histogram('histogram', var)


class ContextEncoder_adv(object):
    def __init__(self, batch_size, nb_epochs, batch_index=0, mask=None, mscoco=params.DATA_PATH,
                 train_path=params.TRAIN_PATH,
                 valid_path=params.VALID_PATH, caption_path=params.CAPTION_PATH,
                 experiment_path=params.EXPERIMENT_PATH,
                 lambda_adversarial=params.LAMBDA_ADVERSARIAL):
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self._train_batch_index = batch_index
        self._valid_batch_index = batch_index
        self.mscoco = mscoco
        self.train_path = train_path
        self.valid_path = valid_path
        self.caption_path = caption_path
        self.experiment_path = experiment_path
        self.save_path = os.path.join(self.experiment_path, "model/")
        self.logs_path = os.path.join(self.experiment_path, "logs")
        self.lambda_adversarial = lambda_adversarial
        create_dir(self.save_path)
        create_dir(self.logs_path)

        self.nb_bw_img = 0

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        if mask is not None:
            self.np_mask = mask
        else:
            self.np_mask = np.zeros((1, 64, 64, 1))
            self.np_mask[:, 16:48, 16:48, :] = 1

        self._get_dataset_characteristics()

        self._sess = tf.Session()

    def _get_dataset_characteristics(self):
        train_path = os.path.join(self.mscoco, self.train_path)
        valid_path = os.path.join(self.mscoco, self.valid_path)
        # caption_path = os.path.join(mscoco, caption_path)
        # with open(caption_path) as fd:
        #     caption_dict = pkl.load(fd)

        self.train_imgs = glob.glob(train_path + "/*.jpg")
        self.valid_imgs = glob.glob(valid_path + "/*.jpg")

    def build_model(self):
        # x : input
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, 64, 64, 3])
        self.mask = tf.placeholder(tf.float32, shape=[1, 64, 64, 1])
        self.x_masked = self.x * (1 - self.mask)

        self._encode()
        self._channel_wise()
        self._decode()
        self._generate_image()
        self._compute_loss()

        # adversarial loss
        self._init_discriminator_variables()
        self._adversarial_loss()

        self._optimize()
        self.merged_summary = tf.summary.merge_all()

    def _encode(self):
        with tf.name_scope("encode"):
            with tf.name_scope('weights'):
                self._W_conv1 = weight_variable([5, 5, 3, 128])
                self._W_conv2 = weight_variable([5, 5, 128, 256])
                self._W_conv3 = weight_variable([5, 5, 256, 512])
                self._W_conv4 = weight_variable([5, 5, 512, 512])
                self._W_conv5 = weight_variable([3, 3, 512, 512])
                variable_summaries(self._W_conv1)
                variable_summaries(self._W_conv2)
                variable_summaries(self._W_conv3)
                variable_summaries(self._W_conv4)
                variable_summaries(self._W_conv5)
            with tf.name_scope('biases'):
                self._b_conv1 = bias_variable([128])
                self._b_conv2 = bias_variable([256])
                self._b_conv3 = bias_variable([512])
                self._b_conv4 = bias_variable([512])
                self._b_conv5 = bias_variable([512])
                variable_summaries(self._b_conv1)
                variable_summaries(self._b_conv2)
                variable_summaries(self._b_conv3)
                variable_summaries(self._b_conv4)
                variable_summaries(self._b_conv5)

            # 64 64 3
            self.h_conv1 = tf.nn.relu(conv2d(self.x_masked, self._W_conv1, stride=1) + self._b_conv1)
            self.h_pool1 = avg_pool_2x2(self.h_conv1)

            # 32 32 128
            self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self._W_conv2, stride=1) + self._b_conv2)
            self.h_pool2 = avg_pool_2x2(self.h_conv2)

            # 16 16 256
            self.h_conv3 = tf.nn.relu(conv2d(self.h_pool2, self._W_conv3, stride=1) + self._b_conv3)
            self.h_pool3 = avg_pool_2x2(self.h_conv3)

            # 8 8 512
            self.h_conv4 = tf.nn.relu(conv2d(self.h_pool3, self._W_conv4, stride=1) + self._b_conv4)
            self.h_pool4 = avg_pool_2x2(self.h_conv4)

            # 4 4 512
            self.h_conv5 = tf.nn.relu(conv2d(self.h_pool4, self._W_conv5, stride=1) + self._b_conv5)

            # 4 4 512

    def _channel_wise(self):
        with tf.name_scope('channel_wise'):
            with tf.name_scope('weights'):
                self._W_fc1 = weight_variable([512, 4 * 4, 4 * 4])
                variable_summaries(self._W_fc1)
            with tf.name_scope('biases'):
                self._b_fc1 = bias_variable([512])
                variable_summaries(self._b_fc1)
            self.h_conv5_flat_img = tf.reshape(self.h_conv5, [512, self.batch_size, 4 * 4])
            self.h_fc1 = tf.nn.relu(
                tf.reshape(tf.matmul(self.h_conv5_flat_img, self._W_fc1), [self.batch_size, 16, 512]) + self._b_fc1)

            self.h_fc1_img = tf.reshape(self.h_fc1, [self.batch_size, 4, 4, 512])

    def _decode(self):
        with tf.name_scope('decode'):
            with tf.name_scope('weights'):
                self._W_uconv1 = weight_variable([5, 5, 512, 512])
                self._W_uconv2 = weight_variable([5, 5, 256, 512])
                self._W_uconv3 = weight_variable([5, 5, 128, 256])
            with tf.name_scope('biases'):
                self._b_uconv1 = bias_variable([512])
                self._b_uconv2 = bias_variable([256])
                self._b_uconv3 = bias_variable([128])
            self.h_uconv1 = tf.nn.relu(
                uconv2d(self.h_fc1_img, self._W_uconv1, output_shape=[self.batch_size, 8, 8, 512],
                        stride=2) + self._b_uconv1)

            # 8 8 512
            self.h_uconv2 = tf.nn.relu(
                uconv2d(self.h_uconv1, self._W_uconv2, output_shape=[self.batch_size, 16, 16, 256],
                        stride=2) + self._b_uconv2)

            # 16 16 256
            self.h_uconv3 = tf.nn.relu(
                uconv2d(self.h_uconv2, self._W_uconv3, output_shape=[self.batch_size, 32, 32, 128],
                        stride=2) + self._b_uconv3)

            # 32 32 128

    def _generate_image(self):
        with tf.name_scope('generated_image'):
            self._W_uconv4 = weight_variable([5, 5, 3, 128])
            self._b_uconv4 = bias_variable([3])
            self.y = tf.nn.relu(
                uconv2d(self.h_uconv3, self._W_uconv4, output_shape=[self.batch_size, 32, 32, 3],
                        stride=1) + self._b_uconv4)
            # 32 32 3
            self.y_padded = tf.pad(self.y, [[0, 0], [16, 16], [16, 16], [0, 0]])
            # 64 64 3
            tf.summary.image("original_image", self.x, max_outputs=12)
            tf.summary.image("generated_image", self.y_padded + self.x_masked, max_outputs=12)

    def _compute_loss(self):
        with tf.name_scope('reconstruction_loss'):
            self._reconstruction_loss = tf.nn.l2_loss(self.mask * (self.x - self.y_padded)) / self.batch_size
            tf.summary.scalar('reconstruction_loss', self._reconstruction_loss)

    def _init_discriminator_variables(self):
        with tf.name_scope('discriminator'):
            with tf.name_scope('weights'):
                self._W_discr1 = weight_variable([5, 5, 3, 128])
                self._W_discr2 = weight_variable([5, 5, 128, 256])
                self._W_discr3 = weight_variable([5, 5, 256, 512])
                self._W_dfc = weight_variable([4 * 4 * 512, 1])

            with tf.name_scope('biases'):
                self._b_discr1 = bias_variable([128])
                self._b_discr2 = bias_variable([256])
                self._b_discr3 = bias_variable([512])
                self._b_dfc = bias_variable([1])

    def _discriminator_encoder(self, image):
        with tf.name_scope('discriminator_encoder'):
            # image is 32 32 3
            h_d1 = tf.nn.relu(conv2d(image, self._W_discr1, stride=1) + self._b_discr1)
            h_dpool1 = avg_pool_2x2(h_d1)

            # 16 16 128
            h_d2 = tf.nn.relu(conv2d(h_dpool1, self._W_discr2, stride=1) + self._b_discr2)
            h_dpool2 = avg_pool_2x2(h_d2)

            # 8 8 256
            h_d3 = tf.nn.relu(conv2d(h_dpool2, self._W_discr3, stride=1) + self._b_discr3)
            h_dpool3 = avg_pool_2x2(h_d3)

            # 4 4 512
            h_dpool3_flat = tf.reshape(h_dpool3, [self.batch_size, 4 * 4 * 512])
            discr = tf.matmul(h_dpool3_flat, self._W_dfc) + self._b_dfc
            return discr

    def _adversarial_loss(self):
        with tf.name_scope('adversarial_loss'):
            self._discr_variables = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
            self._gen_variables = [v for v in tf.trainable_variables() if not v.name.startswith('discriminator')]
            print(len(self._discr_variables), "DISCR VARIABLES ", [v.name for v in self._discr_variables])
            print(len(self._gen_variables),"GEN VARIABLES", [v.name for v in self._gen_variables])

            # center of the image only
            self.real_img = tf.slice(self.x, [0, 16, 16, 0], [self.batch_size, 32, 32, 3])
            real_discr = self._discriminator_encoder(self.real_img)
            fake_discr = self._discriminator_encoder(self.y)

            real_discr_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=real_discr, labels=tf.ones_like(real_discr)))
            fake_discr_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_discr, labels=tf.zeros_like(fake_discr)))

            self._discr_adversarial_loss = (real_discr_loss + fake_discr_loss) / 2
            self._gen_adversarial_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_discr, labels=tf.ones_like(fake_discr)))

            self._discr_loss = self._discr_adversarial_loss
            self._gen_loss = self.lambda_adversarial * self._gen_adversarial_loss + (
                                                                                        1 - self.lambda_adversarial) * self._reconstruction_loss

            tf.summary.scalar("discr loss", self._discr_loss)
            tf.summary.scalar("gen full loss (adversarial and reconstruction)", self._gen_loss)
            tf.summary.scalar("gen adversarial loss", self._gen_adversarial_loss)

    def _optimize(self):
        with tf.variable_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer()
            grads_discr = optimizer.compute_gradients(loss=self._discr_loss, var_list=self._discr_variables)
            grads_gen = optimizer.compute_gradients(loss=self._gen_loss, var_list=self._gen_variables)

            self.train_discr = optimizer.apply_gradients(grads_discr, global_step=self.global_step)
            self.train_gen = optimizer.apply_gradients(grads_gen, global_step=self.global_step)

    def _load_train_batch(self):
        '''
        get next train batch
        '''
        # print("loading batch  index {} ".format(self._train_batch_index))
        start_time = time.time()
        batch = np.zeros((self.batch_size, 64, 64, 3))

        batch_imgs = self.train_imgs[
                     self._train_batch_index * self.batch_size:(self._train_batch_index + 1) * self.batch_size]

        for i, img_path in enumerate(batch_imgs):
            img = Image.open(img_path)
            if np.array(img).shape == (64, 64, 3):
                batch[i] = np.array(img)
            else:
                self.nb_bw_img += 1

        N_train_batch = len(self.train_imgs) // self.batch_size
        self._train_batch_index = (self._train_batch_index + 1) % N_train_batch

        # print("batch loaded in : ", time.time() - start_time)

        return batch

    def _load_valid_batch(self):
        '''
        get next valid batch
        TODO : merge load_batch functions and remove black images
        '''
        start_time = time.time()
        batch = np.zeros((self.batch_size, 64, 64, 3))

        batch_imgs = self.valid_imgs[
                     self._valid_batch_index * self.batch_size:(self._valid_batch_index + 1) * self.batch_size]

        for i, img_path in enumerate(batch_imgs):
            img = Image.open(img_path)
            if np.array(img).shape == (64, 64, 3):
                batch[i] = np.array(img)
            else:
                self.nb_bw_img += 1

        N_valid_batch = len(self.valid_imgs) // self.batch_size
        self._valid_batch_index = (self._valid_batch_index + 1) % N_valid_batch

        # print("batch loaded in : ", time.time() - start_time)

        return batch

    def _load_valid_data(self):
        '''
        get valid data
        :return:
        '''
        n_validation = len(self.valid_imgs) // 5
        val_imgs = self.valid_imgs[:n_validation]

        for i, img_path in enumerate(val_imgs):
            img = Image.open(img_path)
            if np.array(img).shape == (64, 64, 3):
                val_imgs[i] = np.array(img)

        return val_imgs

    def _restore(self):
        """
        Retrieve last model saved if possible
        Create a main Saver object
        Create a SummaryWriter object
        Init variables
        :param save_name: string (default : model)
            Name of the model
        :return:
        """
        saver = tf.train.Saver(max_to_keep=2)
        # Try to restore an old model
        last_saved_model = tf.train.latest_checkpoint(self.save_path)

        self._sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(os.path.join(self.logs_path, "train"),
                                             graph=self._sess.graph,
                                             flush_secs=20)
        val_writer = tf.summary.FileWriter(os.path.join(self.logs_path, "val"),
                                           graph=self._sess.graph,
                                           flush_secs=20)

        if last_saved_model is not None:
            saver.restore(self._sess, last_saved_model)
            print("[*] Restoring model  {}".format(last_saved_model))
        else:
            tf.train.global_step(self._sess, self.global_step)
            print("[*] New model created")
        return saver, train_writer, val_writer

    def train(self):
        """
        Train the model
        :return:
        """
        # Retrieve a model or create a new
        saver, train_writer, val_writer = self._restore()

        epoch = 0
        n_train_batches = len(self.train_imgs) // self.batch_size
        n_val_batches = len(self.valid_imgs) // self.batch_size // 2

        # Retrieve current global step
        last_step = self._sess.run(self.global_step)
        epoch += last_step // n_train_batches
        last_iter = last_step - n_train_batches * epoch
        print("last iter {}".format(last_iter))
        print("last step {}".format(last_step))
        print("epocj {}".format(epoch))
        # Iterate over epochs

        is_not_restart = False
        while epoch < self.nb_epochs:

            for i in tqdm(range(n_train_batches)):
                if i < last_iter and not is_not_restart:
                    continue
                is_not_restart = True
                batch = self._load_train_batch()

                _ = self._sess.run(self.train_gen, feed_dict={self.x: batch, self.mask: self.np_mask})
                ops = [self.train_discr, self.global_step]
                if i % 200 == 0:
                    ops.append(self.merged_summary)
                output = self._sess.run(ops, feed_dict={self.x: batch, self.mask: self.np_mask})

                if i % 200 == 0:
                    # print("nb of black and white images so far : {}".format(self.nb_bw_img))
                    train_writer.add_summary(output[-1], global_step=output[1])

            saver.save(self._sess, global_step=output[1], save_path=self.save_path)
            val_loss = 0
            for i in tqdm(range(n_val_batches)):
                batch = self._load_valid_batch()
                loss, summary_str = self._sess.run([self._reconstruction_loss, self.merged_summary],
                                                   feed_dict={self.x: batch, self.mask: self.np_mask})
                val_loss += loss
            val_loss /= n_val_batches * self.batch_size
            val_writer.add_summary(summary_str, global_step=output[1])
            val_writer.add_summary(
                tf.Summary(value=[tf.Summary.Value(tag="val_loss", simple_value=val_loss), ]), global_step=output[1]
            )
            cprint("Epoch {}".format(epoch), color="yellow")

            epoch += 1

        cprint("Training done.", "green", attrs=["bold"])

        train_writer.flush()
        val_writer.flush()
        train_writer.close()
        val_writer.close()


if __name__ == '__main__':
    ce = ContextEncoder_adv(batch_size=32, nb_epochs=50)
    ce.build_model()
    ce.train()