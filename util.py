# homework 1

# package
%matplotlib inline
# We'll start with our library imports...
from __future__ import print_function

import numpy as np                 # to use numpy arrays
import pandas as pd
import tensorflow as tf            # to specify and run computation graphs
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping # for early stopping
from tensorflow.keras import regularizers            # for regularization


# load data
fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist

# Split the dataset into 90% training and 10% validation
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_full, y_train_full, test_size=0.1, random_state=42
)

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
print("x_valid shape:", x_valid.shape, "y_valid shape:", y_valid.shape)

# scale pixel intensities down to the 0-1 range by dividing 255.
x_train, x_valid, x_test = x_train / 255., x_valid / 255., x_test / 255.