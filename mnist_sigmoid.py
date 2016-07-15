from keras.models import Sequential
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
from keras.layers import Dense, Activation
from keras.utils import np_utils

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28

train_counts = 10000
testing_counts = 1000


(X_train, y_train), (X_test, y_test) = mnist.load_data()

model = Sequential()

# feeding smaller dataset
X_train = np.delete(X_train, np.s_[train_counts:], axis=0)
y_train = np.delete(y_train, np.s_[train_counts:], axis=0)
X_test = np.delete(X_test, np.s_[testing_counts:], axis=0)
y_test = np.delete(y_test, np.s_[testing_counts:], axis=0)

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train.resize(train_counts, img_rows*img_cols)
Y_train.resize(train_counts, nb_classes)
X_test.resize(testing_counts, img_rows*img_cols)
Y_test.resize(testing_counts, nb_classes)

# three layers
model.add(Dense(input_dim=img_rows*img_cols, output_dim=500))
model.add(Activation("sigmoid"))
model.add(Dense(output_dim=500))
model.add(Activation("sigmoid"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))

model.compile(loss='mse',
	optimizer=SGD(lr=0.1),
	metrics=['accuracy'])
model.fit(X_train, Y_train, nb_epoch=20, batch_size=100, verbose=1)

score = model.evaluate(X_test, Y_test)
print('total loss %f' % score[0])
print('accuracy %f' % score[1])

result = model.predict(X_test)