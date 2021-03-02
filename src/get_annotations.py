from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import spatial, stats

from src.api_models import (
    DimensionSchema,
    ObjectsSchema,
    PositionSchema,
    UIDesignPattern,
    WireframeSchema,
)
from src.sagan_models import create_generator

GEN = create_generator(image_size=128, z_dim=128, filters=16, kernel_size=4)

GEN.load_weights("./checkpoints/g/cp-007000.ckpt")

color_map_file = Path("./resources/ui_labels_color_map.csv")
color_map = pd.read_csv(color_map_file, index_col=0, header=None)
color_np_list = color_map.to_numpy()
labels = color_map.index.values

kdt = spatial.KDTree(color_np_list)


def resize_screen(s, interpolation):
    image = ((s[:, :, ::-1] + 1) * 127).astype(np.uint8)
    return cv2.resize(
        image,
        (360, 576),
        interpolation=interpolation,
    )


def sub_threshold(img, erode_flag=False, unsharp_flag=False):
    if unsharp_flag:
        img = unsharp(img)
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if erode_flag:
        thresh = erode(thresh)
    return thresh


def threshold(img):
    m1 = sub_threshold(img[:, :, 0], True, True)
    m2 = sub_threshold(img[:, :, 1], True, True)
    m3 = sub_threshold(img[:, :, 2], True, True)

    res = cv2.add(m1, cv2.add(m2, m3))
    return res


def erode(thresh):
    kernel = np.ones((3, 4), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=3)
    return thresh


def unsharp(imgray):
    imgray = imgray.copy()
    gaussian = cv2.GaussianBlur(imgray, (7, 7), 10.0)
    unsharp_image = cv2.addWeighted(imgray, 2.5, gaussian, -1.5, 0, imgray)
    return unsharp_image


def get_nearest_dominant_color(img):

    pixels = img.reshape(-1, 3)
    if len(pixels) < 50:
        return None, None
    _, ind = kdt.query(pixels)
    m = stats.mode(ind)
    closest_color = color_np_list[m[0][0]]
    label = labels[m[0][0]]
    return (int(closest_color[0]), int(closest_color[1]), int(closest_color[2])), label


def get_wireframe(i, image, category):
    original = image.copy()
    thresh = threshold(image)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    objects = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 25 or h < 25:
            continue
        ROI = original[y : y + h, x : x + w]
        dominant_color, label = get_nearest_dominant_color(ROI)

        if label == "name" and category == UIDesignPattern.product_listing:
            label = "filter"
        if label == "filter" and category == UIDesignPattern.splash:
            label = "sign_up"
        if label == "rating" and category == UIDesignPattern.splash:
            label = "image"
        if label == "sort" and category == UIDesignPattern.splash:
            label = "button"

        if dominant_color is None:
            continue

        position = PositionSchema(x=x, y=y)
        dimension = DimensionSchema(width=w, height=h)
        element = ObjectsSchema(name=label, position=position, dimension=dimension)
        # WireframeSchema
        objects.append(element)

    height, width, _ = original.shape

    wireframe: WireframeSchema = WireframeSchema(
        id=str(i), width=width, height=height, objects=objects
    )

    return wireframe


def get_category_value(category: UIDesignPattern):
    if category == UIDesignPattern.login:
        return 0
    elif category == UIDesignPattern.account_creation:
        return 1
    elif category == UIDesignPattern.product_listing:
        return 2
    elif category == UIDesignPattern.product_description:
        return 3
    else:
        return 4


def generate_wireframe_samples(category: UIDesignPattern, sample_num=16, z_dim=128):
    global GEN

    z = tf.random.truncated_normal(shape=(sample_num, z_dim), dtype=tf.float32)
    c = tf.ones(sample_num, dtype=tf.int32) * get_category_value(category)
    c = tf.reshape(c, [sample_num, 1])
    samples = GEN([z, c])[0].numpy()
    images = np.array([resize_screen(x, cv2.INTER_NEAREST) for x in samples])
    wireframes = [get_wireframe(i, image, category) for i, image in enumerate(images)]
    return wireframes
