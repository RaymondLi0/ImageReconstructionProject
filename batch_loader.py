import params
import os
import glob
import time
import numpy as np
import PIL.Image as Image
from tqdm import tqdm


class BatchLoader(object):
    def __init__(self, batch_size, train_batch_index=0, valid_batch_index=0, mscoco=params.DATA_PATH,
                 train_path=params.TRAIN_PATH,
                 valid_path=params.VALID_PATH, caption_path=params.CAPTION_PATH):
        self.batch_size = batch_size
        self.train_batch_index = train_batch_index
        self.valid_batch_index = valid_batch_index
        self.mscoco = mscoco
        self.train_path = train_path
        self.valid_path = valid_path
        self.caption_path = caption_path

        self._get_dataset_characteristics()

        self.train_batch_index = self.train_batch_index % self.n_train_batches
        self.valid_batch_index = self;valid_batch_index % self.n_valid_batches

    def _get_dataset_characteristics(self):
        train_path = os.path.join(self.mscoco, self.train_path)
        valid_path = os.path.join(self.mscoco, self.valid_path)
        # caption_path = os.path.join(mscoco, caption_path)
        # with open(caption_path) as fd:
        #     caption_dict = pkl.load(fd)

        self.train_imgs = glob.glob(train_path + "/*.jpg")
        self.valid_imgs = glob.glob(valid_path + "/*.jpg")

        self.n_train_batches = len(self.train_imgs) // self.batch_size
        self.n_valid_batches = len(self.valid_imgs) // self.batch_size

    def load_batch(self, train=True, verbose=False):
        '''
        get next train batch
        '''
        if verbose:
            print("loading {} batch index {} ".format("train" if train else "valid", self._train_batch_index))
        start_time = time.time()
        batch = np.zeros((self.batch_size, 64, 64, 3))

        if train:
            batch_imgs = self.train_imgs[
                         self.train_batch_index * self.batch_size:(self.train_batch_index + 1) * self.batch_size]
        else:
            batch_imgs = self.valid_imgs[
                         self.valid_batch_index * self.batch_size:(self.valid_batch_index + 1) * self.batch_size]

        for i, img_path in enumerate(batch_imgs):
            img = Image.open(img_path)
            if np.array(img).shape == (64, 64, 3):
                batch[i] = np.array(img)
            else:
                batch[i] = np.repeat(np.array(img).reshape((64, 64, 1)), 3, axis=2)

        if train:
            self.train_batch_index = (self.train_batch_index + 1) % self.n_train_batches
        else:
            self.valid_batch_index = (self.valid_batch_index + 1) % self.n_valid_batches

        if verbose:
            print("batch loaded in : ", time.time() - start_time)

        return batch


if __name__ == '__main__':
    start_time = time.time()
    bl = BatchLoader(64)
    for i in tqdm(range(100)):
        first_batch = bl.load_batch()
    print("TIME ", time.time()-start_time)
    # print(first_batch[0])
