import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
from termcolor import cprint
import glob
import PIL.Image as Image
import time
import params


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        # with tf.name_scope('stddev'):
        #     stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # tf.summary.scalar('stddev', stddev)
        # tf.summary.scalar('max', tf.reduce_max(var))
        # tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def uconv2d(x, W, output_shape, stride):
    return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1, stride, stride, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class ContextEncoder(object):
    def __init__(self, batch_size, nb_epochs, batch_index=0, mask=None, mscoco=params.DATA_PATH, train_path=params.TRAIN_PATH,
                 valid_path=params.VALID_PATH, caption_path=params.CAPTION_PATH, experiment_path = params.EXPERIMENT_PATH):
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self._train_batch_index = batch_index
        self.valid_batch_index = batch_index
        self.mscoco = mscoco
        self.train_path = train_path
        self.valid_path = valid_path
        self.caption_path = caption_path
        self.experiment_path = experiment_path
        self.save_path = os.path.join(self.experiment_path, "model")
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
        self.x_masked = self.x * (1-self.mask)

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
                self._W_conv1 = weight_variable([5, 5, 3, 64])
                self._W_conv2 = weight_variable([5, 5, 64, 64])
                self._W_conv3 = weight_variable([5, 5, 64, 128])
                variable_summaries(self._W_conv1)
                variable_summaries(self._W_conv2)
                variable_summaries(self._W_conv3)
            with tf.name_scope('biases'):
                self._b_conv1 = bias_variable([64])
                self.b_conv2 = bias_variable([64])
                self.b_conv3 = bias_variable([128])
                variable_summaries(self._b_conv1)
                variable_summaries(self.b_conv2)
                variable_summaries(self.b_conv3)
            self.h_conv1 = tf.nn.relu(conv2d(self.x_masked, self._W_conv1, stride=1) + self._b_conv1)
            self.h_pool1 = max_pool_2x2(self.h_conv1)

            self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self._W_conv2, stride=1) + self.b_conv2)
            self.h_pool2 = max_pool_2x2(self.h_conv2)

            self.h_conv3 = tf.nn.relu(conv2d(self.h_pool2, self._W_conv3, stride=1) + self.b_conv3)
            self.h_pool3 = max_pool_2x2(self.h_conv3)

    def _channel_wise(self):
        with tf.name_scope('channel_wise'):
            with tf.name_scope('weights'):
                self.W_fc1 = weight_variable([128, 8 * 8, 8 * 8])
            with tf.name_scope('biases'):
                self.b_fc1 = bias_variable([128])
            self.h_pool3_flat_img = tf.reshape(self.h_pool3, [128, self.batch_size, 8 * 8])
            self.h_fc1 = tf.nn.relu(
                tf.reshape(tf.matmul(self.h_pool3_flat_img, self.W_fc1), [self.batch_size, 64, 128]) + self.b_fc1)

            self.h_fc1_img = tf.reshape(self.h_fc1, [-1, 8, 8, 128])

    def _decode(self):
        # print(self.h_fc1_img.get_shape().as_list())
        with tf.name_scope('decode'):
            with tf.name_scope('weights'):
                self.W_uconv1 = weight_variable([5, 5, 64, 128])
                self.W_uconv2 = weight_variable([5, 5, 64, 64])
            with tf.name_scope('biases'):
                self.b_uconv1 = bias_variable([64])
                self.b_uconv2 = bias_variable([64])
            self.tmp = uconv2d(self.h_fc1_img, self.W_uconv1, output_shape=[self.batch_size, 16, 16, 64], stride=2)
            # tf.Assert
            self.h_uconv1 = tf.nn.relu(
                uconv2d(self.h_fc1_img, self.W_uconv1, output_shape=[self.batch_size, 16, 16, 64],
                        stride=2) + self.b_uconv1)

            self.h_uconv2 = tf.nn.relu(
                uconv2d(self.h_uconv1, self.W_uconv2, output_shape=[self.batch_size, 32, 32, 64],
                        stride=2) + self.b_uconv2)

    def _generate_image(self):
        with tf.name_scope('generated_image'):
            self.W_uconv3 = weight_variable([5, 5, 3, 64])
            self.b_uconv3 = bias_variable([3])
            self.y = tf.nn.relu(
                uconv2d(self.h_uconv2, self.W_uconv3, output_shape=[self.batch_size, 32, 32, 3],
                        stride=1) + self.b_uconv3)
            self.y_padded = tf.pad(self.y, [[0, 0], [16, 16], [16, 16], [0, 0]])
            tf.summary.image("original_image", self.x, max_outputs=12)
            tf.summary.image("generated_image", self.y_padded + self.x_masked, max_outputs=12)

    def _compute_loss(self):
        with tf.name_scope('reconstruction_loss'):
            self._reconstruction_loss = tf.nn.l2_loss(self.mask * (self.x - self.y_padded))
            tf.summary.scalar('reconstruction_loss', self._reconstruction_loss)

    def _optimize(self):
        train_vars = [v for v in tf.trainable_variables() if v.name.startswith("generated_image")]
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

        self._train_batch_index = (self._train_batch_index + 1) % len(self.train_imgs)

        # print("batch loaded in : ", time.time() - start_time)

        return batch


    def _load_valid_batch(self):
        '''
        get next valid batch
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

        self._valid_batch_index = (self._valid_batch_index + 1) % len(self.train_imgs)

        # print("batch loaded in : ", time.time() - start_time)

        return batch

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

        summary_writer = tf.summary.FileWriter(self.logs_path,
                                               graph=self._sess.graph,
                                               flush_secs=20)
        if last_saved_model is not None:
            saver.restore(self._sess, last_saved_model)
            print("[*] Restoring model  {}".format(last_saved_model))
        else:
            tf.train.global_step(self._sess, self.global_step)
            print("[*] New model created")
        return saver, summary_writer

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
        # Epoch saving (model + embeddings)
        if not is_iter:
            # Save validation_accuracy
            summary_writer.add_summary(extras, global_step=current_iter)

            # Save graph
            saver.save(self._sess, global_step=current_iter, save_path=self.save_path)

        # Iter saving (write variable + loss)
        else:
            summary_writer.add_summary(extras, global_step=current_iter)

    def train(self):
        """
        Train the model
        :return:
        """
        # Retrieve a model or create a new
        saver, summary_writer = self._restore()

        epoch = 0
        n_train_batches = len(self.train_imgs) // self.batch_size


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
                if i < last_iter and not is_not_restart :
                    continue
                is_not_restart = True
                batch = self._load_train_batch()

                _, loss, summary_str, global_step = self._sess.run([self.train_fn, self._reconstruction_loss, self.merged_summary, self.global_step],
                                                      feed_dict={self.x: batch, self.mask: self.np_mask})
                # print(global_step)
                # global_step = self._sess.run(self.global_step)

                if global_step % 200 == 0:
                    print("nb of black and white images so far : {}".format(self.nb_bw_img))
                    self._save(saver, summary_writer, is_iter=False, extras=summary_str)
                    # self._save(saver, summary_writer, is_iter=False, extras=None)

            # self._save(saver, summary_writer, is_iter=False, extras=summary_str)
            cprint("Epoch {}".format(epoch), color="yellow")

            epoch += 1

        cprint("Training done.", "green", attrs=["bold"])

        summary_writer.flush()
        summary_writer.close()

if __name__ == '__main__':
    ce = ContextEncoder(batch_size=16, nb_epochs=30)
    ce.build_model()
    ce.train()
