import itertools
import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dropout, Softmax, Dense, GaussianNoise, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python import confusion_matrix
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.utils.np_utils import to_categorical

VERSION = "kp-hr-1"

mapping = {"0": "A", "1": "B", "2": "C", "3": "D", "4": "E", "5": "F", "6": "G", "7": "H", "8": "I", "9": "J",
           "10": "K", "11": "L", "12": "M", "13": "N", "14": "O", "15": "P", "16": "Q", "17": "R", "18": "S", "19": "T",
           "20": "U", "21": "V", "22": "W", "23": "X", "24": "Y", "25": "Z", "26": "del", "27": "nothing",
           "28": "space"}

data = np.load("key-points/X.npy")
labels = np.load("key-points/Y.npy")

clusters = [
    ["Y", "space", "nothing", "X"],
    ["A", "E", "T", "M", "N", "S", "I", "J"],
    ["B", "D", "F", "Z", "L"],
    ["U", "V", "W", "K", "R"],
    ["G", "H", "P", "Q", "del"],
    ["O", "C"],
]


def norm_data(in_data):
    ndata = in_data.reshape(-1, 21, 2)
    maximum = np.max(ndata, axis=1)
    minimum = np.min(ndata, axis=1)
    ranges = maximum - minimum
    ndata = (ndata - minimum[:, None, :]) / ranges[:, None, :]
    ndata = ndata.reshape(-1, 42)
    return ndata


nd = norm_data(data)


def block(size, dropout=0.2):
    def func(x):
        x = Dense(size, kernel_initializer="glorot_normal")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        if dropout > 0:
            x = Dropout(dropout)(x)
        return x

    return func


def create_high_model(num_classes=5):
    model_input = Input(shape=42)

    x = model_input
    x = GaussianNoise(0.02)(x)

    mem = x
    x = block(1024)(x)
    x = block(512)(x)
    mem = block(512)(mem)
    x = Add()([mem, x])
    x = block(256)(x)
    x = block(128, 0)(x)
    x = Dense(num_classes, kernel_initializer="glorot_normal")(x)
    x = Softmax()(x)

    model = Model(model_input, x)
    model.compile(optimizer=Adam(learning_rate=5e-4),
                  loss=CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])
    return model


def create_sub_model(num_classes=29):
    model_input = Input(shape=42)

    x = model_input
    x = GaussianNoise(0.02)(x)

    x = block(512)(x)
    x = block(256)(x)
    x = block(128, 0)(x)

    x = Dense(num_classes, kernel_initializer="glorot_normal")(x)
    x = Softmax()(x)

    model = Model(model_input, x)
    model.compile(optimizer=Adam(learning_rate=5e-4),
                  loss=CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])
    return model


def create_clustered_labels():
    inverse = {k: v for v, k in mapping.items()}
    classed_labels = np.zeros(labels.shape[0])
    classes_indices = []

    for i in range(len(clusters)):
        c = clusters[i]
        indices = np.isin(np.argmax(labels, axis=1), [int(inverse[l]) for l in c])
        classed_labels[indices] = np.repeat(i, labels[indices].shape[0])
        classes_indices.append(indices)
    return to_categorical(classed_labels), classes_indices


def predict(x, y, cy):
    cp = high_model(x)
    print(np.sum(np.argmax(cp, axis=1) == np.argmax(cy, axis=1)) / x.shape[0])
    p = np.zeros_like(y)
    for i in range(len(clusters)):
        indices = np.argmax(cp, axis=1) == i
        p[indices] = models[i]["model"](x[indices])
    print(np.sum(np.argmax(y, axis=1) == np.argmax(p, axis=1)) / x.shape[0])
    return p, cp


BATCH_SIZE = 64
EPOCHS = 150

# Train the large model
clustered_labels, clustered_indices = create_clustered_labels()
high_model = create_high_model(len(clusters))
high_model.summary()
X_train, X_test, y_train, y_test = train_test_split(nd, clustered_labels, test_size=0.2)

history = high_model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
                         validation_data=(X_test, y_test), shuffle=True,
                         callbacks=[ReduceLROnPlateau(patience=3, factor=0.9, min_lr=1e-9, verbose=1),
                                    EarlyStopping(patience=7, verbose=1)])

high_model.save(f"versions/{VERSION}/high")
with open(f"versions/{VERSION}/high/history.pickle", "wb") as f:
    pickle.dump(history.history, f)

models = {}
for i in range(len(clusters)):
    indices = clustered_indices[i]
    small_data = nd[indices]
    small_labels = labels[indices]
    m = create_sub_model()
    X_train, X_test, y_train, y_test = train_test_split(small_data, small_labels, test_size=0.2)
    history = m.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                    verbose=1, callbacks=[ReduceLROnPlateau(patience=3, factor=0.9, min_lr=1e-9, verbose=1),
                                          EarlyStopping(patience=5, verbose=1)],
                    validation_data=(X_test, y_test), shuffle=True)
    m.save(f"versions/{VERSION}/sub{i}")
    with open(f"versions/{VERSION}/sub{i}/history.pickle", "wb") as f:
        pickle.dump(history.history, f)
    models[i] = {"model": m, "x_test": X_test, "y_test": y_test, "hist": history}


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(16, 16))

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm / np.sum(cm, axis=1).reshape(-1, 1), decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.imshow(cm, interpolation='nearest', cmap="Reds")
    plt.title("Confusion matrix")
    plt.colorbar()
    plt.show()


predictions, predicted_clusters = predict(nd, labels, clustered_labels)

#     return figure
cluster_matrix = confusion_matrix(clustered_labels.argmax(axis=1), np.argmax(predicted_clusters, axis=1))
plot_confusion_matrix(cluster_matrix, clusters)
matrix = confusion_matrix(labels.argmax(axis=1), predictions.argmax(axis=1))
plot_confusion_matrix(matrix, mapping.values())
