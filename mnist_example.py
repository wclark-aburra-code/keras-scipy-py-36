from __future__ import absolute_import, division, print_function, unicode_literals                                                              
import tensorflow as tf                                                 
from tensorflow import keras                                            
import numpy as np

dataset = keras.datasets.fashion_mnist
(training_data, training_categories), (test_data, test_categories) = dataset.load_data()
assert training_data.shape[0] == len(training_categories)
assert test_data.shape[0] == len(test_categories)

training_data = training_data / 255.0 # normalization; all values must be between zero and one
test_data = test_data / 255.0
sequential_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # 28 by 28 images need to be flattened to arrays 
    keras.layers.Dense(128, activation='relu'), 
    keras.layers.Dense(10, activation='softmax') 
])
sequential_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
sequential_model.fit(training_data, training_categories, epochs=10)
loss, accuracy = sequential_model.evaluate(test_data, test_categories)
print('\nTest accuracy:', accuracy) # 0.8854 in test