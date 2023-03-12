import pprint
from copy import deepcopy
from functools import lru_cache
from PIL import Image
import base64, io, os
import numpy as np
import torch
import torchvision.transforms.functional as F
import tensorflow as tf
from ultralytics.yolo.utils.plotting import Annotator, colors


def plot(orig_img, result, show_conf=True, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc', cls_2_name={0:"clean", 1:"dirty"}):
        """
        Plots the detection results on an input RGB image. Accepts a numpy array (cv2) or a PIL Image.
        Args:
            show_conf (bool): Whether to show the detection confidence score.
            line_width (float, optional): The line width of the bounding boxes. If None, it is scaled to the image size.
            font_size (float, optional): The font size of the text. If None, it is scaled to the image size.
            font (str): The font to use for the text.
            pil (bool): Whether to return the image as a PIL Image.
            example (str): An example string to display. Useful for indicating the expected format of the output.
        Returns:
            (None) or (PIL.Image): If `pil` is True, a PIL Image is returned. Otherwise, nothing is returned.
        """
        img = deepcopy(orig_img)
        annotator = Annotator(img, line_width, font_size, font, pil, example)
        boxes = result.boxes
        logits = result.probs
        names = cls_2_name
        masks = None
        # names = [result[0].boxes.boxes[i][-1] for i in range(len(result[0].boxes.boxes))]
        if boxes is not None:
            for d in reversed(boxes):
                cls, conf = d.cls.squeeze(), d.conf.squeeze()
                c = int(cls)
                label = (f'{names[c]}' if names else f'{c}') + (f'{conf:.2f}' if show_conf else '')

                annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))

        if masks is not None:
            im = torch.as_tensor(img, dtype=torch.float16, device=masks.data.device).permute(2, 0, 1).flip(0)
            im = F.resize(im.contiguous(), masks.data.shape[1:]) / 255
            annotator.masks(masks.data, colors=[colors(x, True) for x in boxes.cls], im_gpu=im)

        if logits is not None:
            n5 = min(len(names), 5)
            top5i = logits.argsort(0, descending=True)[:n5].tolist()  # top 5 indices
            text = f"{', '.join(f'{names[j] if names else j} {logits[j]:.2f}' for j in top5i)}, "
            annotator.text((32, 32), text, txt_color=(255, 255, 255))  # TODO: allow setting colors

        return img


def create_model(image_shape, lr=1e-4, weights_path="/kaggle/input/vgg16-weights-tf-dim-ordering-tf-kernels-notop/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"):
    base_model = tf.keras.applications.VGG16(
        weights=weights_path,
        include_top=False,
        input_shape=image_shape
    )
    model = tf.keras.models.Sequential()
    model.add(base_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.layers[0].trainable = False

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=[
            'accuracy',
            tf.keras.metrics.BinaryCrossentropy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.TruePositives(),
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.FalseNegatives()
        ]
    )

    return model

def pil_to_b64(im, enc_format="png", **kwargs):
    """
    Converts a PIL Image into base64 string for HTML displaying
    :param im: PIL Image object
    :param enc_format: The image format for displaying. If saved the image will have that extension.
    :return: base64 encoding
    """

    buff = io.BytesIO()
    im.save(buff, format=enc_format, **kwargs)
    encoded = base64.b64encode(buff.getvalue()).decode("utf-8")

    return encoded


