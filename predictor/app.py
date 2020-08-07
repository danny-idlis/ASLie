from hModel import hierarchical_model
from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)

model = hierarchical_model.HierarchicalModel()

mapping = ["A", "B", "C", "D", "E", "F", "G", "H", "A", "B", "C", "D", "E", "F", "G", "H", "A", "B", "C", "D", "E", "F",
           "G", "H", "A", "B", "C", "D", "E", "F", "G", "H", "A", "B", "C", "D", "E", "F", "G", "H"]


@app.route("/", methods=["POST"])
def get_model_prediction():
    data = request.json
    model_input = format_keypoints(data)
    output = handle_model_prediction(model(model_input))
    return jsonify(output)


def handle_model_prediction(prediction):
    index = np.argmax(prediction[0])
    print("Letter:", model.mapping[str(index)], "Score:", np.max(prediction[0]))
    return {"letter": model.mapping[str(index)], "score": np.max(prediction[0])}


def format_keypoints(keypoints, norm=True, suffix=""):
    if norm:
        keypoints = np.asarray(keypoints)
        nd = keypoints.reshape(-1, 21, 2)
        maximum = np.max(nd, axis=1)
        minimum = np.min(nd, axis=1)
        ranges = maximum - minimum
        x = ((nd - minimum[:, None, :]) / ranges[:, None, :])

    return x.reshape(-1, 42)
