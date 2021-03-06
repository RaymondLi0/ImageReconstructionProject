import glob
import os
import time

import PIL.Image as Image
import numpy as np
import params
import tensorflow as tf
from termcolor import cprint
from tqdm import tqdm

from ImageReconstructionProject.utils import weight_variable, bias_variable, conv2d, uconv2d, max_pool_2x2, create_dir


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        tf.summary.histogram('histogram', var)


class ContextEncoder(object):
    def __init__(self, batch_size, nb_epochs, batch_index=0, mask=None, mscoco=params.DATA_PATH,
                 train_path=params.TRAIN_PATH,
                 valid_path=params.VALID_PATH, caption_path=params.CAPTION_PATH,
                 experiment_path=params.EXPERIMENT_PATH):
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
            self.h_conv1 = tf.nn.relu(conv2d(self.x_masked, self._W_conv1, stride=1) + self._b_conv1)
            self.h_pool1 = max_pool_2x2(self.h_conv1)

            self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self._W_conv2, stride=1) + self._b_conv2)
            self.h_pool2 = max_pool_2x2(self.h_conv2)

            self.h_conv3 = tf.nn.relu(conv2d(self.h_pool2, self._W_conv3, stride=1) + self._b_conv3)
            self.h_pool3 = max_pool_2x2(self.h_conv3)

            self.h_conv4 = tf.nn.relu(conv2d(self.h_pool3, self._W_conv4, stride=1) + self._b_conv4)
            self.h_pool4 = max_pool_2x2(self.h_conv4)

            self.h_conv5 = tf.nn.relu(conv2d(self.h_pool4, self._W_conv5, stride=1) + self._b_conv5)

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

            self.h_uconv2 = tf.nn.relu(
                uconv2d(self.h_uconv1, self._W_uconv2, output_shape=[self.batch_size, 16, 16, 256],
                        stride=2) + self._b_uconv2)

            self.h_uconv3 = tf.nn.relu(
                uconv2d(self.h_uconv2, self._W_uconv3, output_shape=[self.batch_size, 32, 32, 128],
                        stride=2) + self._b_uconv3)

    def _generate_image(self):
        with tf.name_scope('generated_image'):
            self._W_uconv4 = weight_variable([5, 5, 3, 128])
            self._b_uconv4 = bias_variable([3])
            self.y = tf.nn.relu(
                uconv2d(self.h_uconv3, self._W_uconv4, output_shape=[self.batch_size, 32, 32, 3],
                        stride=1) + self._b_uconv4)
            self.y_padded = tf.pad(self.y, [[0, 0], [16, 16], [16, 16], [0, 0]])
            tf.summary.image("original_image", self.x, max_outputs=12)
            tf.summary.image("generated_image", self.y_padded + self.x_masked, max_outputs=12)

    def _compute_loss(self):
        with tf.name_scope('reconstruction_loss'):
            self._reconstruction_loss = tf.nn.l2_loss(self.mask * (self.x - self.y_padded)) / self.batch_size
            tf.summary.scalar('reconstruction_loss', self._reconstruction_loss)

    def _optimize(self):
        with tf.variable_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer()
            grads = optimizer.compute_gradients(self._reconstruction_loss)
            # capped_grads = [(tf.clip_by_norm(grad, 1), var) for grad, var in grads]
            self.train_fn = optimizer.apply_gradients(grads, global_step=self.global_step)

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

    def _save(self, saver, summary_writer, is_iter=True, extras=None):
        """
        All save operations (Graph, Weights, Curve, Embeddings)
        This function handles all writings on disk
        :param saver: tf.train.Saver
            The main Saver object to save graph object
        :param summary_writer: tf.summary.FileWriter
            The main FileWriter to write all summary operations for TensorBoard
        :param is_iter: boolean (default: True)
            Saving is different given its a saving iteration or a saving iteration operation
        :param extras: float or tf.summaries
            Use to save all summaries for iteration, and validation accuracy for epoch operations
        :return:
        """

        current_iter = self._sess.run(self.global_step)
        # Epoch saving (logs + model)
        if not is_iter:
            # Save validation_accuracy
            summary_writer.add_summary(extras, global_step=current_iter)

            # Save graph
            saver.save(self._sess, global_step=current_iter, save_path=self.save_path)

        # Iter saving (logs)
        else:
            summary_writer.add_summary(extras, global_step=current_iter)

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

                _, loss, summary_str, global_step = self._sess.run(
                    [self.train_fn, self._reconstruction_loss, self.merged_summary, self.global_step],
                    feed_dict={self.x: batch, self.mask: self.np_mask})

                if global_step % 200 == 0:
                    # print("nb of black and white images so far : {}".format(self.nb_bw_img))

                    self._save(saver, train_writer, is_iter=True, extras=summary_str)

            val_loss = 0
            for i in tqdm(range(n_val_batches)):
                batch = self._load_valid_batch()
                loss, summary_str = self._sess.run([self._reconstruction_loss, self.merged_summary], feed_dict={self.x: batch, self.mask: self.np_mask})
                val_loss += loss
            val_loss /= n_val_batches * self.batch_size
            self._save(saver, val_writer, is_iter=False, extras=summary_str)
            val_writer.add_summary(
                tf.Summary(value=[tf.Summary.Value(tag="val_loss", simple_value=val_loss), ]), global_step=global_step
            )
            cprint("Epoch {}".format(epoch), color="yellow")

            epoch += 1

        cprint("Training done.", "green", attrs=["bold"])

        train_writer.flush()
        val_writer.flush()
        train_writer.close()
        val_writer.close()


if __name__ == '__main__':
    ce = ContextEncoder(batch_size=32, nb_epochs=50)
    ce.build_model()
    ce.train()
