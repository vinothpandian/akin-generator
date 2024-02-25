from __future__ import division

import argparse
import json
import os

import cv2
import numpy as np

from scipy import spatial
from scipy import stats

from src.colorMapper import ColorMapper


def sub_threshold(img, st, erode_flag=False, unsharp_flag=False):
    if unsharp_flag:
        img = unsharp(img, st)
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if erode_flag:
        thresh = erode(thresh, st)
    # cv2.imwrite('res_' + str(st) + '.jpg', thresh)
    return thresh


def threshold(img):
    m1 = sub_threshold(img[:, :, 0], 1, True, True)  # --- threshold on blue channel
    m2 = sub_threshold(img[:, :, 1], 2, True, True)  # --- threshold on green channel
    m3 = sub_threshold(img[:, :, 2], 3, True, True)  # --- threshold on red channel

    # cv2.imwrite("image_thresh.png", res)
    return cv2.add(m1, cv2.add(m2, m3))


def erode(thresh, st):
    kernel = np.ones((3, 4), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=3)
    # cv2.imwrite('image_erode_'+str(st)+'.png', thresh)
    return thresh


def unsharp(imgray, st):
    # Unsharp mask here
    imgray = imgray.copy()
    gaussian = cv2.GaussianBlur(imgray, (7, 7), 10.0)
    # cv2.imwrite("unsharp_"+str(st)+".jpg", unsharp_image)

    return cv2.addWeighted(imgray, 2.5, gaussian, -1.5, 0, imgray)


def get_bounding_boxes(dir, image_name, dst_path, dir_name):
    elements = []
    image = cv2.imread(os.path.join(dir, image_name))
    original = image.copy()
    thresh = threshold(image)
    new_semantic = np.ones_like(original) * 255
    # Find contours, obtain bounding box, extract and save ROI
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
        ROI = original[y:y + h, x:x + w]
        dominant_color, label = get_nearest_dominant_color(ROI)
        if dominant_color is None:
            continue
        cv2.rectangle(new_semantic, (x, y), (x + w, y + h), dominant_color, 3)
        elements.append({"points": [[x, y], [x + w, y + h]], "label": label})
    cv2.imwrite(os.path.join(dst_path, f"{image_name[:-4]}0.png"), image)
    cv2.imwrite(os.path.join(dst_path, f"{image_name[:-4]}1.png"), new_semantic)
    create_json_file(
        os.path.join(dst_path, f"{image_name[:-4]}.json"), elements, dir_name
    )


def get_nearest_dominant_color(img):
    pixels = img.reshape(-1, 3)
    if len(pixels) < 50:
        return None, None
    _, ind = kdt.query(pixels)
    m = stats.mode(ind)
    closest_color = color_np_list[m[0][0]]
    label = labels[m[0][0]]
    return (int(closest_color[0]), int(closest_color[1]), int(closest_color[2])), label


def create_json_file(path, elements, flag):
    data = {"shapes": elements,
            "imageHeight": 567,
            "imageWidth": 360,
            "flags": {flag: True}
            }
    if data is not None and data:
        with open(path, "w+") as ff:
            json.dump(data, ff, indent=True)


if __name__ == '__main__':
    """
        all folder stuctures are ../path_to_src_or_dst/category
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic_images_folder", "-s", help="path to generated semantic images",
                        default="../../data/output_semantic_images")
    parser.add_argument("--destination_path", "-d", help="path to destination folder",
                        default="../../data/postprocessed")
    parser.add_argument("--color_map_file", "-c", help="path to file for mapping of color to UI element type",
                        default="../resources/ui_labels_color_map.csv")

    args = parser.parse_args()

    folder_path = args.semantic_images_folder
    dest_path = args.destination_path
    color_map_file = args.color_map_file
    color_map = ColorMapper.read_label_color_map(color_map_file, bgr=False)
    color_np_list = np.array(list(color_map.values()))
    labels = list(color_map.keys())
    kdt = spatial.KDTree(color_np_list)
    for dir in os.listdir(folder_path):
        print(dir)
        folder = os.path.join(folder_path, dir)
        if os.path.isdir(folder):
            for file in os.listdir(folder):
                print(file)
                dst_path = os.path.join(dest_path, dir)
                if not os.path.exists(dst_path):
                    os.mkdir(dst_path)
                get_bounding_boxes(folder, file, dst_path, dir)
