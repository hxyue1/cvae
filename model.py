from tensorflow import keras as tfk
from tensorflow.python.keras import layers as tfkl
from tensorflow.python.keras.layers import Input, InputLayer, Lambda, Reshape, \
                                    Dropout, Flatten, Dense, Conv2D, \
                                    Conv2DTranspose
import tensorflow as tf

class EncoderZ_5(tfkl.Layer):
    # n_filter_base is a bogus argument for now
    def __init__(self, z_dim, n_filter_base, dtype, name='decoder', **kwargs):
        super(EncoderZ_5, self).__init__(name=name, **kwargs)

        self.conv_layers = tf.keras.Sequential(
          [
              tf.keras.layers.InputLayer(input_shape=(96, 96, 3)),
              tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu', dtype=dtype),
              tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu', dtype=dtype),
              tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2), activation='relu', dtype=dtype),
              tf.keras.layers.Flatten()
          ]
        )
        
        self.dense_mean = Dense(
              z_dim, 
              activation=None, 
              name='z_mean', 
              dtype=dtype
              )
        
        self.dense_raw_stddev = Dense(
            z_dim, 
            activation=None,
            name='z_raw_stddev', 
            dtype=dtype
            )
        
        self.sampler_z = Sampler_Z()
      
    def call(self, x_input):
      z = self.conv_layers(x_input)
      mean = self.dense_mean(z)
      logvar = self.dense_raw_stddev(z)
      z_sample = self.sampler_z((mean, logvar))
      return z_sample, mean, logvar

class DecoderX_5(tfkl.Layer):

    def __init__(self, z_dim, n_filter_base, dtype, name='decoder', **kwargs):
        super(DecoderX_5, self).__init__(name=name, **kwargs)

        self.deconv_layers = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(z_dim,)),
                tf.keras.layers.Dense(units=12*12*128, activation=tf.nn.relu, dtype=dtype),
                tf.keras.layers.Reshape(target_shape=(12, 12, 128)),
                tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation='relu', dtype=dtype),
                tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu', dtype=dtype),
                tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu', dtype=dtype),
                tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same', dtype=dtype),
            ]
        )
    

    def call(self, z):
      return self.deconv_layers(z)

class Sampler_Z(tfkl.Layer):   

    def call(self, inputs):
        mean, logvar = inputs
        # Reparameterize
        eps = tf.random.normal(shape=mean.shape)
        z_sample = eps * tf.exp(logvar * .5) + mean
        #tf.print(tf.math.reduce_mean(eps))
        return z_sample