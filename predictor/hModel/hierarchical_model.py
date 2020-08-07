from os import listdir

import numpy as np
from tensorflow.python.keras.saving.save import load_model




class HierarchicalModel:
    def __init__(self, basedir="hModel/kp-hr-1", num_classes=29):
        self.num_classes = num_classes
        self.high_model = load_model(f"{basedir}/high")
        print("Loaded high model")
        self.models = {}
        for d in listdir(basedir):
            if not d.startswith("sub"):
                continue
            index = d[-1]
            self.models[index] = load_model(f"{basedir}/{d}")
            print(f"Loaded {d} model")

        self.clusters = [
            ["Y", "space", "nothing", "X"],
            ["A", "E", "M", "N", "T", "S", "I", "J"],
            ["B", "D", "F", "Z", "L"],
            ["U", "V", "W", "K", "R"],
            ["G", "H", "P", "Q", "del"],
            ["O", "C"],
        ]
        self.mapping = {"0": "A", "1": "B", "2": "C", "3": "D", "4": "E", "5": "F", "6": "G", "7": "H", "8": "I",
                        "9": "J", "10": "K", "11": "L", "12": "M", "13": "N", "14": "O", "15": "P", "16": "Q",
                        "17": "R", "18": "S", "19": "T", "20": "U", "21": "V", "22": "W", "23": "X", "24": "Y",
                        "25": "Z", "26": "del", "27": "nothing", "28": "space"}

    def predict(self, X):
        cp = self.high_model(X)
        p = np.zeros((X.shape[0], self.num_classes))
        for i in self.models.keys():
            indices = np.argmax(cp, axis=1) == int(i)
            p[indices] = self.models[i](X[indices])
        return p, cp

    def __call__(self, X):
        return self.predict(X)


