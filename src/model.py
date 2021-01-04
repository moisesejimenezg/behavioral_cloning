from keras.models import Sequential, load_model
from keras.layers import Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D, Convolution2D
from keras.layers.pooling import MaxPooling2D

class Model:
    def __init__(self, image_h=160, image_w=320, image_d=3):
        model = Sequential()
        ## Normalization
        model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(160, 320, 3)))
        ## Cropping of the images to focus on the valuable information
        model.add(Cropping2D(cropping=((70, 25), (0, 0))))
        model.add(Convolution2D(24, 5, 2, activation="relu"))
        model.add(Convolution2D(36, 5, 2, activation="relu"))
        model.add(Convolution2D(48, 5, 2, activation="relu"))
        model.add(Convolution2D(64, 3, activation="relu"))
        model.add(Convolution2D(64, 3, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(10))
        model.add(Dense(1))
        model.compile(loss="mse", optimizer="adam")
        self.model = model

    def fit_model(self, x, y, validation_split=0.2, shuffle=True):
        print("Fitting model with: " + str(len(x)) + " images.")
        self.model.fit(x, y, validation_split=validation_split, shuffle=shuffle, epochs=5)
        self.model.save("model.h5")

    def load_model(self, model = 'model.h5'):
        self.model = load_model(model)
        print(self.model.summary())
