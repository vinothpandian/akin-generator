import os

import tensorflow as tf

image_size = 64
num_of_categories = 6
z_dim = 100
input_layers = tf.keras.layers.Input((z_dim,))
z = tf.keras.layers.Reshape((1, 1, z_dim))(input_layers)

category_input = tf.keras.layers.Input((1,))
y = tf.keras.layers.Embedding(num_of_categories,10)(category_input)
y = tf.keras.layers.Reshape((1, 1, 10))(y)

x = tf.keras.layers.concatenate([z, y])
model = tf.keras.models.Model([input_layers, category_input], x)
model.summary()