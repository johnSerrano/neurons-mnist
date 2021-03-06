from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import random
import sys

num_iterations = 10
if (len(sys.argv) > 1):
    num_iterations = int(sys.argv[1])

nb_classes = 10

model = Sequential()

model.add(Convolution2D(25, 3, 3, border_mode="same", input_shape=(1, 28, 28)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, border_mode="same", input_shape=(1, 28, 28)))
model.add(Activation("relu"))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(output_dim=100, init="glorot_uniform"))
model.add(Activation("relu"))
model.add(Dropout(0.4))
model.add(Dense(output_dim=10, init="glorot_uniform"))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer='adadelta')

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

train_data = train_data.reshape(train_data.shape[0], 1, 28, 28)
test_data = test_data.reshape(test_data.shape[0], 1, 28, 28)

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
train_data /= 255
test_data /= 255
test_labels_copy = test_labels[:]
train_labels = np_utils.to_categorical(train_labels, nb_classes)
test_labels = np_utils.to_categorical(test_labels, nb_classes)

model.fit(train_data, train_labels, nb_epoch=num_iterations, batch_size=128, verbose=1, shuffle=True, show_accuracy=True)

result = model.evaluate(test_data, test_labels, batch_size=128, show_accuracy=True, verbose=0, sample_weight=None)
print('Test score:', result[0])
print('Test accuracy:', result[1])

results = model.predict_classes(test_data, batch_size=128, verbose=0)

test_data = test_data.reshape(10000, 28, 28)

#for _ in range(10):
for i in range(9):
    j = random.randint(0, 10000)
    plt.subplot(3,3,i+1)
    plt.imshow(test_data[j], cmap='gray', interpolation='none')
    plt.title("Result: {} ; {}".format(results[j], test_labels_copy[j]))

plt.show()
