import cv2
from PIL import Image
import numpy as np

from hand_detector import HandDetector
from utils import crop

detector = HandDetector()

img = Image.open("b.jpg", "r")
img.show()
boxes, scores = detector.get_boxes(img)
print(boxes[0])
cropped = crop(img, box=boxes[0], score=scores[0])

cropped.show()
from matplotlib import pyplot as plt
# img.show()

hsv = cropped.convert("HSV")

# h = hsv.getchannel("H")
s = hsv.getchannel("S")
# v = hsv.getchannel("V")
# hsv.show()
# h.show()
s.show()
# v.show()

sarr = np.asarray(s)
mask = (sarr > 40)
Image.fromarray(mask).show()
Image.fromarray(mask * sarr).show()
Image.fromarray(np.asarray(cropped) * mask[:,:, None]).show()
