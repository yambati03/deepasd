from sparse_ae import SparseAutoencoder
from parse_abide import *
import tensorflow as tf
from tensorflow import keras
import numpy as np

ids = get_subject_ids()
networks = get_saved_networks(ids)
features = vectorise_networks(networks)

x_train, x_test = prepare_data(features)

model = SparseAutoencoder(100, 741)

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, x_train,
                epochs=50,
                batch_size=1,
                shuffle=True,
                validation_data=(x_test, x_test))
