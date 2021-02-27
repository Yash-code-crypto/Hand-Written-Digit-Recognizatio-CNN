#libraries
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D,MaxPooling2D

#getting dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# expand the dimension for color channel

x_train=np.expand_dims(x_train,axis=3)
x_test=np.expand_dims(x_test,axis=3)

#one hot coding

y_train =to_categorical(y_train, 10)
y_test =to_categorical(y_test, 10)

#features scaling
x_train,x_test=x_train/255,x_test/255



def my_con():
  model=Sequential()
  model.add(Conv2D(32, (5, 5),activation='relu',input_shape=(28,28,1)))
  model.add(MaxPooling2D((2, 2)))

  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D((2, 2)))

  model.add(Flatten())
  model.add(Dense(100, activation='relu'))

  model.add(Dropout(0.5))
  model.add(Dense(10, activation='softmax'))

  model.compile(optimizer=Adam(lr=0.01),loss='categorical_crossentropy',metrics=['accuracy'])

  return model

model=my_con()

#traning the model
history = model.fit(x_train, y_train,batch_size=256,epochs=8,verbose=1,validation_split=0.2,shuffle=True)

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('CNN_model2.h5')























