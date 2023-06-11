import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix


class BinaryClassifier:

    def __init__(self, data, goal):
        self.data = data
        self.cnn = tf.keras.models.Sequential()
        self.trash = 'trash'
        self.goal = goal
        self.training_set = None
        self.test_set = None

    def data_processing(self):

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        self.training_set = train_datagen.flow_from_directory(
            f'{self.data}/training_set',
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary')

        test_datagen = ImageDataGenerator(rescale=1. / 255)
        self.test_set = test_datagen.flow_from_directory(
            f'{self.data}/test_set',
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary')

        # convolution layer 추가

    def layers(self):
        self.cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))  # 풀링 적용

        self.cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        self.cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        self.cnn.add(tf.keras.layers.Flatten())

        self.cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

        self.cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        self.cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def fit(self):
        history = self.cnn.fit(x=self.training_set, validation_data=self.test_set, epochs=25)
        import matplotlib.pyplot as plt

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'])
        plt.show()

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'])
        plt.show()

    def evaluate(self):

        y_true = self.test_set.classes

        y_pred_prob = self.cnn.predict(self.test_set)
        y_pred = np.round(y_pred_prob).flatten()

        cm = confusion_matrix(y_true, y_pred)

        print(cm)

    def single_prediction(self, test):
        test_image = image.load_img(self.data + test, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = self.cnn.predict(test_image / 255.0)
        if self.training_set.class_indices[self.goal] == 1:
            if result[0][0] > 0.5:
                prediction = self.goal
            else:
                prediction = self.trash
        else:
            if result[0][0] < 0.5:
                prediction = self.goal
            else:
                prediction = self.trash
        print(prediction)


cardboard = BinaryClassifier('/Users/cheonsemin/PycharmProjects/newly_made_codes/venv/dataset/cardboard','cardboard')
cardboard.data_processing()
cardboard.layers()
cardboard.fit()
cardboard.single_prediction('/single_prediction/guess_what.jpg')

glass = BinaryClassifier('/Users/cheonsemin/PycharmProjects/newly_made_codes/venv/dataset/glass', 'glass')
glass.data_processing()
glass.layers()
glass.fit()
glass.single_prediction('/single_prediction/guess_what.jpg')

metal = BinaryClassifier('/Users/cheonsemin/PycharmProjects/newly_made_codes/venv/dataset/metal', 'metal')
metal.data_processing()
metal.layers()
metal.fit()
metal.single_prediction('/single_prediction/guess_what.jpg')

paper = BinaryClassifier('/Users/cheonsemin/PycharmProjects/newly_made_codes/venv/dataset/paper', 'paper')
paper.data_processing()
paper.layers()
paper.fit()
paper.single_prediction('/single_prediction/guess_what.jpg')

plastic = BinaryClassifier('/Users/cheonsemin/PycharmProjects/newly_made_codes/venv/dataset/plastic', 'plastic')
plastic.data_processing()
plastic.layers()
plastic.fit()
plastic.single_prediction('/single_prediction/guess_what.jpg')

