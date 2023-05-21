import numpy as np
import os
import cv2

import matplotlib.pyplot as plt
from keras.datasets import mnist


# Spiral dataset
class Spiral:

    # generate spiral dataset
    @staticmethod
    def generate(points: int, classes: int) -> tuple:                                                  
            X = np.zeros((points*classes, 2))
            Y = np.zeros(points*classes, dtype="uint8")
            for class_number in range(classes):
                ix = range(points*class_number, points*(class_number+1))
                r = np.linspace(0.0, 1.0, points)
                t = np.linspace(class_number*4.0, (class_number+1.0)*4.0, points) + np.random.randn(points)*0.2
                X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
                Y[ix] = class_number
            return X, Y

    # display spiral dataset
    @staticmethod
    def display(X:list, Y:list) -> None:
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='brg')
        plt.title("Spiral-Datensatz")
        plt.show()

# Mnist dataset
class Mnist:

    # load / generate mnist dataset
    @staticmethod
    def generate(samples_training_data:int, samples_val_data:int) -> tuple:
        (X, Y), (X_val, Y_val) = mnist.load_data()
        X, X_val = X / 255, X_val / 255 
        X, Y = X[:samples_training_data], Y[:samples_training_data]
        X_val, Y_val = X_val[:samples_val_data], Y_val[:samples_val_data]
        return X, Y, X_val, Y_val

    # display mnist dataset
    @staticmethod
    def display(X: np.ndarray) -> None:
        for i in range(9):
            plt.subplot(331 + i)
            plt.imshow(X[i], cmap=plt.get_cmap("gray"))
        plt.title("Mnist-Datensatz")
        plt.show()


# Cat vs. Dogs dataset
class Cat_Dogs:

    @staticmethod
    def generate(samples_training_data:int, samples_val_data:int) -> tuple:
        TRAIN_DIR = "src/datasets/cats_dogs/train/"
        IMG_SIZE = 50
        X = []
        Y = []
        X_val = []
        Y_val = []

        # train data
        for img in os.listdir(path=TRAIN_DIR)[12500-int(samples_training_data/2):12500+int(samples_training_data/2)]:
            Y.append(0 if img.split(".")[-3] == "cat" else 1)
            path = os.path.join(TRAIN_DIR, img)
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
            X.append(np.array(img))

        # validation data
        for img in os.listdir(path=TRAIN_DIR)[:samples_val_data]:
            Y_val.append(0 if img.split(".")[-3] == "cat" else 1)
            path = os.path.join(TRAIN_DIR, img)
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
            X_val.append(np.array(img))

        X, X_val = np.array(X) / 255, np.array(X_val) / 255 
    
        return X, Y, X_val, Y_val

    @staticmethod
    def display(X: np.ndarray) -> None:
        for i in range(9):
            plt.subplot(331 + i)
            plt.imshow(X[i], cmap=plt.get_cmap("gray"))
        ("Katzen-Hunde-Datensatz")
        plt.show()


