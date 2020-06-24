import tensorflow as tf
from tensorflow import keras

class Encoder(keras.layers.Layer):
  def __init__(self, encoded_dim):
    super(Encoder, self).__init__()
    self.enocoder_layer = keras.layers.Dense(encoded_dim, activation='relu', activity_regularizer=keras.regularizers.l1(10e-5))

  def call(self, input):
    return self.enocoder_layer(input)

class Decoder(keras.layers.Layer):
  def __init__(self, encoded_dim, input_dim):
    super(Decoder, self).__init__()
    self.decoder_layer = keras.layers.Dense(input_dim, activation='sigmoid')

  def call(self, code):
    return self.decoder_layer(code)

class SparseAutoencoder(keras.Model):
  def __init__(self, encoded_dim, input_dim):
    super(SparseAutoencoder, self).__init__()
    self.encoder = Encoder(encoded_dim)
    self.decoder = Decoder(encoded_dim, input_dim)

  def call(self, input):
    code = self.encoder(input)
    reconstructed = self.decoder(code)
    return reconstructed
