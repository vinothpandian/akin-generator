import os

import cv2
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

class DataGenerator():
    def __init__(self, image_paths, labels, image_size, batch_size):
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size
        self.batch_size = batch_size
        self.generator = self._generator()
        self.dataset = self.get_semantic_images_for_gan_train(batch_size, True)

    def read_image(self, image_path, crop=False):
        image = cv2.imread(image_path)

        if crop:
            image = self.crop_center(image)

        image = cv2.resize(image, (self.image_size, self.image_size))
        return image[:, :, ::-1].astype(np.float32) # convert to RGB and float

    def crop_center(self, image):
        h_center, w_center, shift = image.shape[0] // 2, image.shape[1] // 2, self.image_size//2
        
        return image[
            int(h_center-(shift)):int(h_center-(shift)+self.image_size), 
            int(w_center-(shift)):int(w_center-(shift)+self.image_size)
        ]

    def _generator(self):
        data = np.empty((self.batch_size, self.image_size, self.image_size, 3))
        idxes = np.arange(len(self.image_paths))
        while True:
            np.random.shuffle(idxes)
            
            i = 0   
            while (i + 1) * self.batch_size <= len(self.image_paths):
                batch_paths = self.image_paths[i * self.batch_size:(i + 1) * self.batch_size]

                for j, path in enumerate(batch_paths):
                    img = self.read_image(path, crop=False)
                    data[j] = (img / 127.) - 1
                i += 1
                yield data.astype(np.float32)

    def get_semantic_images_for_gan_train(self, batch_size, cache=None):
        dataset_len = len(self.image_paths)

        list_ds = tf.data.Dataset.from_tensor_slices((self.image_paths, self.labels))

        train_dataset = list_ds.map(self.process_path_for_gan, num_parallel_calls=AUTOTUNE)

        if dataset_len % batch_size != 0:
            dataset_len = int(dataset_len / batch_size) * batch_size

        train_dataset = train_dataset.take(dataset_len)

        train_dataset = train_dataset.shuffle(buffer_size=dataset_len).batch(batch_size=self.batch_size)
        # Repeat forever

        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        if cache:
            if isinstance(cache, str):
                train_dataset = train_dataset.cache(cache)
            else:
                train_dataset = train_dataset.cache()

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

        return train_dataset

    def process_path_for_gan(self, image_path, label):
        # load the raw data from the file as a string
        img = tf.io.read_file(image_path)
        img = self.decode_img(img, self.image_size, self.image_size, convert_to_bgr=False)

        return img, [label - 1]

    def decode_img(self, img, img_width, img_height, convert_to_bgr=False):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # convert channels to bgr?
        if convert_to_bgr:
            img = img[..., ::-1]
        # resize the image to the desired size.
        img = (img * 2) - 1.0
        return tf.image.resize(img, [img_width, img_height])