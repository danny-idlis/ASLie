import timeit

from flask import Flask
from flask_socketio import SocketIO, emit

from hand_detector import HandDetector
from utils import decode_base64, substract_background, filter_small_boxes

app = Flask(__name__)
socketio = SocketIO(app)

@socketio.on("frame")
def handle_frame(data):
    print("Got Frame")
    start = timeit.default_timer()
    image = decode_base64(data['frame'])
    image = substract_background(img=image)
    boxes, scores = hand_detector.get_boxes(image, data["threshold"])
    if len(boxes) > 0:
        boxes, scores = filter_small_boxes(boxes, scores, 0.2)
    print(f"Found {len(boxes)} hands, with max score of {max(scores or [0])}")
    emit("box", {'boxes': boxes, 'scores': scores})  # Send the client the box to show

    print(f"Finished processing frame in {timeit.default_timer() - start}sec")


@app.route('/', methods=['GET'])
def hello():
    return "Welcome to ASLie"


if __name__ == '__main__':
    print("Starting ASLie...")
    print("Loading hand detector...")
    hand_detector = HandDetector()
    print("Hand detector loaded.")
    print("ASLie ready :)")
    socketio.run(app, host="0.0.0.0", port="1607")
