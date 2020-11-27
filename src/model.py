from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D, Convolution2D
from keras.layers.pooling import MaxPooling2D

class Model:
    def __init__(self, image_h=160, image_w=320, image_d=3):
        model = Sequential()
        model.add(Lambda(lambda x: x/255, input_shape=(160, 320, 3)))
        model.add(Convolution2D(6, 5, 5, activation="relu"))
        model.add(MaxPooling2D())
        model.add(Convolution2D(6, 5, 5, activation="relu"))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(120))
        model.add(Dense(84))
        model.add(Dense(1))
        model.compile(loss="mse", optimizer="adam")
        self.model = model

    def fit_model(self, x, y, validation_split=0.2, shuffle=True):
        self.model.fit(x, y, validation_split=validation_split, shuffle=shuffle, epochs=10)
        self.model.save("model.h5")
