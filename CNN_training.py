### Important : Unzip data in data folder before running this script.


import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# import tensorflow as tf
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import layers
from keras.layers import *
from keras.utils import np_utils
from tqdm import tqdm


train_df = pd.read_csv("Data/emnist.csv", header=None)
# train_df.head()
X_train = train_df.loc[:, 1:]
y_train = train_df.loc[:, 0]

label_map = pd.read_csv(
    "Data/label_mapping.txt",
    delimiter=" ",
    index_col=0,
    header=None,
    squeeze=True,
)


label_dictionary = {}
for index, label in enumerate(label_map):
    label_dictionary[index] = chr(label)


def reshape_and_rotate(image):
    W = 28
    H = 28
    image = image.reshape(W, H)
    image = np.fliplr(image)
    image = np.rot90(image)
    return image


X_train = np.apply_along_axis(reshape_and_rotate, 1, X_train.values)
X_train = X_train.astype("float32") / 255

number_of_classes = y_train.nunique()

y_train = np_utils.to_categorical(y_train, number_of_classes)

X_train = X_train.reshape(-1, 28, 28, 1)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=88
)

print(X_train.shape, y_train.shape)


model = Sequential()

model.add(
    layers.Conv2D(
        filters=32,
        kernel_size=(5, 5),
        padding="same",
        activation="relu",
        input_shape=(W, H, 1),
    )
)
model.add(layers.MaxPool2D(strides=2))
model.add(
    layers.Conv2D(filters=48, kernel_size=(5, 5), padding="valid", activation="relu")
)
model.add(layers.MaxPool2D(strides=2))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(84, activation="relu"))
model.add(layers.Dense(number_of_classes, activation="softmax"))

model.summary()


optimizer_name = "adam"

model.compile(
    loss="categorical_crossentropy", optimizer=optimizer_name, metrics=["accuracy"]
)

early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=1, mode="min")
mcp_save = ModelCheckpoint(
    "SavedModels/handwritten-cnn-trained.h5",
    save_best_only=True,
    monitor="val_loss",
    verbose=1,
    mode="auto",
)

model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    verbose=1,
    validation_split=0.1,
    callbacks=[early_stopping, mcp_save],
)
