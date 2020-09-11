# Import Libraries
from tensorflow import keras

from keras.models import Sequential
from keras.datasets import mnist

from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta

from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 15

# Import Data + Split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess Data (Current Dimension (60000, 28, 28) -> Should be (60000,28,28,1))

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

input_shape = (28, 28, 1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# CNN Model: Most fit for image classification problems


model = Sequential()

model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu' , input_shape = input_shape))
model.add(Conv2D(64, (3 , 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2 , 2)))

model.add(Dropout(0.25)) # helps prevent overfitting
model.add(Flatten()) # Flattens the input

model.add(Dense(256, activation='relu')) # implements the operation
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=categorical_crossentropy,
              optimizer=Adadelta(),
              metrics=['accuracy'])

# Train the Model
hist = model.fit(X_train,
                 y_train,
                 batch_size= batch_size,
                 epochs= epochs,
                 verbose= 1,
                 validation_data= (X_test, y_test) )

print('Training was successfull')

model.save('Mnist.h5')
print("Uploading to Mnist.h5")

score = model.evaluate(X_test, y_test, verbose = 0)