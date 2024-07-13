import os
import numpy as np

from keras.models import load_model
import keras


#Setting up the layers of the neural Network
digit_model = keras.Sequential([
    keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu',padding = 'Same', input_shape=(28,28,1), ),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu', padding = 'Same', ),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10,activation='softmax')
])

model = load_model(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'my_model.h5'))


def predict_digit(image):
    """
    Predict the digit in the cell.
    """
    image = image.reshape(1, 28, 28, 1)
    image = image / 255.0
    output = model.predict(image)
    prediction = np.argmax(output)
    if output[0][prediction]<0.9:
        return 0
    return prediction

