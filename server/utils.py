import base64
from io import BytesIO

import numpy as np
from PIL import Image


def decode_base64(data):
    """Decode base64, padding being optional.

    :param data: Base64 image data as an ASCII byte string
    :returns: PIL Image.

    """
    img64 = base64.b64decode(data[len("data:image/jpeg;base64"):])
    return Image.open(BytesIO(img64))


def box_to_region(img, box, score=1):
    """
        Converts a box given as (upper, left, bottom, right)-tuple with percentages,
        to PIL region as (left, upper, right, lower)-tuple
    :param img: PIL image.
    :param box: A box given as (upper, left, bottom, right)-tuple with percentages.
    :param score:
        The score given to the box from the hand detection model.
        If less than 0.9, padding will be applied base on this score.
        For no padding, keep default, or send 0.9+.
    :return: A region in the picture in absolute pixels for the PIL crop function0, padded from center if needed.
    """

    region = (box[1] * img.width), \
             (box[0] * img.height), \
             (box[3] * img.width), \
             (box[2] * img.height)

    if score < 0.9:
        scale = min(1 / score, 1.2)

        region = (region[0] - (img.width * (scale - 1) / 2)), \
                 (region[1] - (img.height * (scale - 1) / 2)), \
                 (region[2] + (img.height * (scale - 1) / 2)), \
                 (region[3] + (img.height * (scale - 1) / 2))

        # Clipping the padded region to the image
        region = (max(region[0], 0),
                  max(region[1], 0),
                  min(region[2], img.width),
                  min(region[3], img.height))

    if region[0] < 0 or region[1] < 0 or region[2] > img.width or region[3] > img.height:
        print("<WTF>")
        print(box)
        print(region)
        print(score, min(1 / score, 1.2))
        print(img.width, img.height)
        print("</WTF?")
    return region


def substract_background(img):
    def get_mask(arr):
        mean = np.mean(arr)
        std = np.std(arr)
        return arr > mean + (mean / std)

    hsv = img.convert("HSV")
    s = hsv.getchannel("S")
    sarr = np.asarray(s)
    return Image.fromarray(img * get_mask(sarr)[:, :, None])


def filter_small_boxes(boxes, scores, threshold):
    t = tuple(zip(
        *filter(lambda b: b[0][2] - b[0][0] > threshold and b[0][3] - b[0][1] > threshold, list(zip(boxes, scores)))))
    if len(t) == 0:
        return [], []
    return t


def crop(img, box, score=1):
    # Convert box to (left, upper, right, lower)-tuple.
    region = box_to_region(img, box, score)
    return img.crop(region)
