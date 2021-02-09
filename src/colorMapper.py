from __future__ import division

import os
import sys

from .uiLabelFileManager import UILabelFileManager

foo_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(foo_dir, "..", "..")))

import cv2
import numpy as np
from scipy.spatial.distance import cdist


class ColorMapper:
    @staticmethod
    def map_color_and_save(color_string, sorted_labels, hierarchy_file, level, save_csv_file, save_image_file):
        s = color_string.split(",")
        colors = []
        max = 0
        max_i = 0
        ctr = 0
        for color in s:
            if "#" in color:
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
                if r * g * b > max:
                    max = r * g * b
                    max_i = ctr
                colors.append(np.array([r, g, b]))
                ctr += 1
        sorted_colors = ColorMapper.get_sorted_colors(colors, max_i)
        if sorted_labels is None:
            sorted_labels = UILabelFileManager.get_sorted_labels_for_hierarchy(sorted_label_file, hierarchy_file, level)
        label_color_map = ColorMapper.map_colors_to_labels(sorted_colors, sorted_labels)
        ColorMapper.save_label_map(label_color_map, save_csv_file)
        ColorMapper.save_color_map(label_color_map, save_image_file)

    @staticmethod
    def save_color_map(label_color_map, save_image_file):
        side = 30
        h = len(label_color_map.keys()) * side
        w = side * 4
        img = np.zeros((h, w, 3), np.uint8)
        for y in range(h):
            for x in range(w):
                img[y, x] = [255, 255, 255]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.4
        fontColor = (0, 0, 0)
        lineType = 1
        i = 0
        for k, v in label_color_map.items():
            img[(i * side) : (i * side) + side, 0:side, :] = [v[0], v[1], v[2]]
            bottomLeftCornerOfText = (int(side * 1.5), (i * side) + int(side / 2))
            img = cv2.putText(img, str(k), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
            i += 1
        cv2.imwrite(save_image_file, img)

    @staticmethod
    def get_sorted_colors(colors, max_i):
        sorted_color = []
        sorted_color.append(colors.pop(max_i))
        metric = "euclidean"
        while len(colors) > 0:
            n = np.array(sorted_color)
            m = np.array(colors)
            dist = cdist(n, m, metric=metric)
            sum = np.sum(dist, axis=0)
            result = np.where(sum == np.amax(sum))
            sorted_color.append(colors.pop(int(result[0][0])))
        return sorted_color

    @staticmethod
    def map_colors_to_labels(sorted_colors, sorted_labels):
        print(len(sorted_colors))
        print(len(sorted_labels))
        assert len(sorted_colors) == len(sorted_labels)
        label_map = {}
        for i, label in enumerate(sorted_labels):
            label_map[label] = sorted_colors[i]
        return label_map

    @staticmethod
    def save_label_map(label_map, file):
        with open(file, "w+") as f:
            for k, v in label_map.items():
                s = k + "," + str(v[0]) + "," + str(v[1]) + "," + str(v[2]) + "\n"
                f.write(s)

    @staticmethod
    def read_label_color_map(file, bgr=False):
        label_color_map = {}
        if os.path.exists(file):
            with open(file, "r") as f:
                data = f.readlines()
                for row in data:
                    s = row.replace("\n", "").split(",")
                    label = s[0]
                    if bgr:
                        color = [int(s[3]), int(s[2]), int(s[1])]
                    else:
                        color = [int(s[1]), int(s[2]), int(s[3])]
                    label_color_map[label] = color
        else:
            print(str(file) + " file does not exists")
        return label_color_map

    @staticmethod
    def read_label_color_map_string(file):
        label_color_map = ColorMapper.read_label_color_map(file)
        for k, v in label_color_map.items():
            print('elif "' + str(k) + '" == label:\n\treturn (' + str(v[0]) + "," + str(v[1]) + "," + str(v[2]) + ")")

    @staticmethod
    def create_color_pallate(file, image_file):
        color_map = ColorMapper.read_label_color_map(file)
        ColorMapper.save_color_map(color_map, image_file)

    @staticmethod
    def print_color_map_as_hex(file):
        label_color_map = ColorMapper.read_label_color_map(file, bgr=True)
        for key, v in label_color_map.items():
            color = "%02x%02x%02x" % (v[0], v[1], v[2])
            k = key.replace("_", " ")
            print(k + " & $\\#\\mathrm{" + color + "}$ \\\\")


if __name__ == "__main__":
    color_8 = "black,#000000,red,#ff0000,gold,#ffd700,mediumvioletred,#c71585,lime,#00ff00,blue,#0000ff,dodgerblue,#1e90ff,aquamarine,#7fffd4"
    color_29 = "black,#000000,dimgray,#696969,lightgray,#d3d3d3,darkolivegreen,#556b2f,saddlebrown,#8b4513,forestgreen,#228b22,darkslateblue,#483d8b,darkgoldenrod,#b8860b,darkcyan,#008b8b,navy,#000080,darkseagreen,#8fbc8f,darkmagenta,#8b008b,maroon3,#b03060,red,#ff0000,gold,#ffd700,lime,#00ff00,mediumspringgreen,#00fa9a,blueviolet,#8a2be2,crimson,#dc143c,aqua,#00ffff,deepskyblue,#00bfff,blue,#0000ff,greenyellow,#adff2f,fuchsia,#ff00ff,dodgerblue,#1e90ff,khaki,#f0e68c,salmon,#fa8072,plum,#dda0dd,deeppink,#ff1493"
    sorted_label_file = "../../../data/ui_labels_sorted.csv"
    hierarchy_file = "../../../data/ui_labels_hierarchy.csv"
    level = 1
    save_csv_file = "../../../data/ui_labels_color_map.csv"
    save_image_file = "../../../data/ui_labels_color.jpg"
    sorted_labels = [
        "text",
        "image",
        "log_in",
        "sign_up",
        "username",
        "password",
        "icon",
        "forgot",
        "sm_button",
        "button",
        "box",
        "privacy",
        "check",
        "name",
        "navigation_dots",
        "number",
        "selector",
        "search",
        "edit_number",
        "edit_string",
        "filter",
        "top_bar",
        "heart_icon",
        "sort",
        "rating",
        "bottom_bar",
        "card_add",
        "other",
        "buy",
    ]

    ColorMapper.map_color_and_save(color_29, sorted_labels, hierarchy_file, level, save_csv_file, save_image_file)
