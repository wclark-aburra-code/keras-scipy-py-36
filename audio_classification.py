from __future__ import absolute_import, division, print_function, unicode_literals                                                              
import tensorflow as tf                                                 
from tensorflow import keras                                            
import numpy
import wavefile
import os

# this padding function that adds trailing zeroes is important so that all wave signals have the same length,
# and consequently the neural network can work with numpy arrays of consitent shape and dimensions
def padded(byte_array, output_length):
    return numpy.append(byte_array,numpy.zeros(output_length-len(byte_array))) # check - append, not concat

maximum_length = 180000 # this is about the maximum length
labels = {"snare" : 1, "kick" : 2, "hi_hat" : 3}

# load training data
training_labels = []
training_values = []

file_directory = './audio_dataset/train/hi_hat'
file_list = [f for f in os.listdir(file_directory) if  os.path.isfile(os.path.join(file_directory, f)) and (f != '.DS_Store')] 
for fname in file_list:
    imported_wave = wavefile.load(filename=file_directory + "/" + fname)
    mono_channel = imported_wave[1][0] # we want the left channel, or mono
    normalized_channel = (mono_channel + 1)/2 # normalization, to yield a value between zero and one
    training_labels.append(labels["hi_hat"])
    training_values.append(padded(normalized_channel, maximum_length))

file_directory = './audio_dataset/train/snare'
file_list = [f for f in os.listdir(file_directory) if  os.path.isfile(os.path.join(file_directory, f)) and (f != '.DS_Store')] 
for fname in file_list:
    imported_wave = wavefile.load(filename=file_directory + "/" + fname)
    mono_channel = imported_wave[1][0] # we want the left channel, or mono
    normalized_channel = (mono_channel + 1)/2 # normalization, to yield a value between zero and one
    training_labels.append(labels["snare"])
    training_values.append(padded(normalized_channel, maximum_length))

file_directory = './audio_dataset/train/kick_drum'
file_list = [f for f in os.listdir(file_directory) if  os.path.isfile(os.path.join(file_directory, f)) and (f != '.DS_Store')] 
for fname in file_list:
    imported_wave = wavefile.load(filename=file_directory + "/" + fname)
    mono_channel = imported_wave[1][0] # we want the left channel, or mono
    normalized_channel = (mono_channel + 1)/2 # normalization, to yield a value between zero and one
    training_labels.append(labels["kick"])
    training_values.append(padded(normalized_channel, maximum_length))

# define neural network model
sequential_model = keras.Sequential([
    keras.layers.Dense(32, input_shape=(maximum_length,)), 
    keras.layers.Dense(10, activation='softmax') 
])
sequential_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
# Note that we don't have to normalize the training values like with MNIST images.
# The wavfile library automatically yields floats between 0 and 1 for all sample amplitudes
sequential_model.fit(numpy.asarray(training_values), numpy.asarray(training_labels), epochs=10)

# load testing data
testing_labels = []
testing_values = []

file_directory = './audio_dataset/test/hi_hat'
file_list = [f for f in os.listdir(file_directory) if  os.path.isfile(os.path.join(file_directory, f)) and (f != '.DS_Store')] 
for fname in file_list:
    imported_wave = wavefile.load(filename=file_directory + "/" + fname)
    mono_channel = imported_wave[1][0] # we want the left channel, or mono
    normalized_channel = (mono_channel + 1)/2 # normalization, to yield a value between zero and one
    testing_labels.append(labels["hi_hat"])
    testing_values.append(padded(normalized_channel, maximum_length))

file_directory = './audio_dataset/test/snare'
file_list = [f for f in os.listdir(file_directory) if  os.path.isfile(os.path.join(file_directory, f)) and (f != '.DS_Store')] 
for fname in file_list:
    imported_wave = wavefile.load(filename=file_directory + "/" + fname)
    mono_channel = imported_wave[1][0] # we want the left channel, or mono
    normalized_channel = (mono_channel + 1)/2 # normalization, to yield a value between zero and one
    testing_labels.append(labels["snare"])
    testing_values.append(padded(normalized_channel, maximum_length))

file_directory = './audio_dataset/test/kick_drum'
file_list = [f for f in os.listdir(file_directory) if  os.path.isfile(os.path.join(file_directory, f)) and (f != '.DS_Store')] 
for fname in file_list:
    imported_wave = wavefile.load(filename=file_directory + "/" + fname)
    mono_channel = imported_wave[1][0] # we want the left channel, or mono
    normalized_channel = (mono_channel + 1)/2 # normalization, to yield a value between zero and one
    testing_labels.append(labels["kick"])
    testing_values.append(padded(normalized_channel, maximum_length))

loss, accuracy = sequential_model.evaluate(numpy.asarray(testing_values), numpy.asarray(testing_labels))
print('\nTest accuracy:', accuracy) # roughly 0.61764705 in test