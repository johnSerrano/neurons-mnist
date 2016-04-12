from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt


nb_classes = 10

model = Sequential()

model.add(Dense(input_dim=784, output_dim=300, init="glorot_uniform"))
model.add(Activation("relu"))
model.add(Dense(input_dim=300, output_dim=100, init="glorot_uniform"))
model.add(Activation("relu"))
model.add(Dense(input_dim=100, output_dim=10, init="glorot_uniform"))
model.add(Activation("softmax"))


model.compile(loss="mean_absolute_error", optimizer='adam')


(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(train_data[i], cmap='gray', interpolation='none')
    plt.title("Class {}".format(train_labels[i]))

train_data = train_data.reshape(60000, 784)
test_data = test_data.reshape(10000, 784)
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
train_data /= 255
test_data /= 255
train_labels = np_utils.to_categorical(train_labels, nb_classes)
test_labels = np_utils.to_categorical(test_labels, nb_classes)

model.fit(train_data, train_labels, nb_epoch=10, batch_size=128, verbose=1, shuffle=True, show_accuracy=True)

result = model.evaluate(test_data, test_labels, batch_size=10000, show_accuracy=True, verbose=1, sample_weight=None)
print('Test score:', result[0])
print('Test accuracy:', result[1])
