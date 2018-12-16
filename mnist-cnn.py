import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
from PIL import Image


def load_training_data():
    training_data = pd.read_csv('datasets/train.csv')
    x_train = training_data.values[:, 1:]  # column 0 is label
    y_train = training_data.values[:, 0]
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_train = x_train / 255  # convert [0, 255] -> [0,1]
    # one hot encoding for labels
    y_train = to_categorical(y_train, num_classes=10)
    return x_train, y_train


def load_test_data():
    testing_data = pd.read_csv('datasets/test.csv')
    x_test = testing_data.values[:, :]  # return a numpy representation of pd DataFrame
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)  # reshape 784-> 28*28*1
    x_test = x_test / 255
    return x_test


def train():
    x_train, y_train = load_training_data()

    # Set the CNN model
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                     activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='acc', patience=2)
    model.fit(x_train, y_train, epochs=30, batch_size=3000, validation_split=0.1, callbacks=[early_stopping])

    model.save('model/mnist_cnn_model.h5')


def predict():
    x_test = load_test_data()
    selected_model = load_model('model/mnist_cnn_model.h5')
    result = selected_model.predict(x_test, batch_size=1000)
    result = np.argmax(result, axis=1)
    result = pd.Series(result, name='Label')
    return result


def export_result(result):
    # build submission DataFrame
    submission = pd.concat([pd.Series(range(1, 28001), name='ImageId'), result], axis=1)
    submission.to_csv('result/result.csv', index=False)


def predict_own_picture():
    filename = input("Please input your own picture's path(should be 28*28, black background):\n")
    im = Image.open(filename)
    im = im.convert('L')  # convert image to grey scale
    im = im.resize((28, 28), Image.ANTIALIAS)
    im.show()
    my_image = np.array(im)
    my_image = my_image.reshape(1, 28, 28, 1)
    selected_model = load_model('model/mnist_cnn_model.h5')
    my_image = my_image / 255
    result = selected_model.predict(my_image)
    result = np.argmax(result, axis=1)

    print("The number is: " + str(result) + "\n")


while 1:
    command = input("Please input a number:\n"
                    "1. train model\n"
                    "2. make predictions and export to csv\n"
                    "3. predict your own picture\n"
                    "4. quit\n")
    if command == "1" or command == "train":
        train()
    elif command == "2" or command == "predict":
        test_result = predict()
        export_result(test_result)
    elif command == "3" or command == "predict own":
        predict_own_picture()
    elif command == "4" or command == "quit":
        break
print("finish")
