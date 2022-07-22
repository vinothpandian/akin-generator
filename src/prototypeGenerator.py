import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .semanticJsonParser import SemanticJsonParser
from .uiLabelFileManager import UILabelFileManager


def load_all_ui_images():
    android_label_map = {}
    with open(android_element_mapping_file, "r") as f:
        data = f.readlines()
        for line in data:
            s = line.split(";")
            label = s[0]
            img_name = s[1]
            if len(img_name) > 0:
                img_path = os.path.join(android_elements_base_path, f"{img_name}.jpg")
                if not os.path.exists(img_path):
                    img_path = os.path.join(android_elements_base_path, f"{img_name}.png")
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            else:
                img = None
            text = s[2]
            text = None if text is None or len(text) == 0 else text.strip().split(",")
            resize = int(s[3])
            android_label_map[label] = {"img": img, "text": text, "resize": resize, "label": label}
    return android_label_map


def get_elements(path, real):
    elements = []
    try:
        data = None
        with open(path, "r") as f:
            data = json.load(f)
            if real:
                return SemanticJsonParser.read_json(data, label_hierarchy_map)
            shapes = data["shapes"]
            flags = data["flags"]
            for shape in shapes:
                label = shape["label"]
                points = shape["points"]
                elements.append(
                    [label, [int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1])]]
                )
    except Exception as e:
        print(e)
    return elements


def element_resize(img, w, h, flag, base_shade):
    """

    :param img:
    :param flag: 0 -> no
                1 -> normal
                2 -> maintain aspect ratio with center align
                3 -> maintain aspect ratio with left align
                4 -> maintain aspect ratio with right align
                5 -> normal but double text
    :return:
    """
    if flag == 0:
        return img, 0
    elif flag in [1, 5]:
        return cv2.resize(img, (w, h)), 0
    elif flag in [2, 3, 4]:
        label_image = np.ones((h, w, 3)) * base_shade
        ih = img.shape[0]
        iw = img.shape[1]
        rw = w / iw
        rh = h / ih
        fw = 0
        if rw == 1 and rh == 1:
            label_image = img
        elif rw < rh:
            fw = reshape_to_w(h, ih, img, iw, label_image, w, flag)
        else:
            fw = reshape_to_h(h, ih, img, iw, label_image, w, flag)
        return label_image, fw


def element_resize_old(img, w, h, flag, base_shade):
    """

    :param img:
    :param flag: 0 -> no
                1 -> normal
                2 -> maintain aspect ratio with center align
                3 -> maintain aspect ratio with left align
                4 -> maintain aspect ratio with right align
                5 -> normal but double text
    :return:
    """
    if flag == 0:
        return img, 0
    elif flag in [1, 5]:
        return cv2.resize(img, (w, h)), 0
    elif flag in [2, 3, 4]:
        label_image = np.ones((h, w, 3)) * base_shade
        ih = img.shape[0]
        iw = img.shape[1]
        dw = w - iw
        dh = h - ih
        fw = 0
        if dw == 0 and dh == 0:
            label_image = img
        elif dw > 0 and dh > 0:
            if dw < dh:
                fw = reshape_to_w(h, ih, img, iw, label_image, w, flag)
            else:
                fw = reshape_to_h(h, ih, img, iw, label_image, w, flag)
        elif dw <= 0 and dh >= 0:
            fw = reshape_to_w(h, ih, img, iw, label_image, w, flag)
        elif dw >= 0 and dh <= 0:
            fw = reshape_to_h(h, ih, img, iw, label_image, w, flag)
        elif dw < 0 and dh < 0:
            rw = w / iw
            rh = h / ih
            if rw < rh:
                fw = reshape_to_w(h, ih, img, iw, label_image, w, flag)
            else:
                fw = reshape_to_h(h, ih, img, iw, label_image, w, flag)
        return label_image, fw


def reshape_to_h(h, ih, img, iw, label_image, w, align):
    r = h / ih
    tw = int(iw * r)
    img = cv2.resize(img, (tw, h))
    if align == 2:  # center
        t = int((w - tw) / 2)
        label_image[0:h, t : t + tw] = img
    elif align == 3:  # left
        label_image[0:h, 0:tw] = img
    elif align == 4:  # right
        label_image[0:h, w - tw - 1 : w - 1] = img
    return tw


def reshape_to_w(h, ih, img, iw, label_image, w, align):
    r = w / iw
    th = int(ih * r)
    img = cv2.resize(img, (w, th))
    if align == 2:  # center
        t = int((h - th) / 2)
        label_image[t : t + th, 0:w] = img
    elif align == 3:  # left
        label_image[0:th, 0:w] = img
    elif align == 4:  # right
        label_image[h - th - 1 : h - 1, 0:w] = img
    return w


def create_img(elements, dst_file_path, cat, real=True):
    base_image = np.ones((img_h, img_w, 3)) * 255
    elements.sort(key=lambda x: (x[1][1], x[1][0]))
    element_counted = {}
    for label, bb in elements:
        x1 = int(bb[0])
        y1 = int(bb[1])
        x2 = int(bb[2]) - 1
        y2 = int(bb[3]) - 1
        w = x2 - x1
        h = y2 - y1
        if x1 >= 0 and y1 >= 0 and x2 < img_w and y2 < img_h and w > 0 and h > 0:
            x = x1
            y = y1
            if not real and (h < 20 or w < 20):
                continue
            elif y >= img_h or x >= img_w:
                continue
            if label == "name" and cat == "product_listing":
                label = "filter"
            if label == "filter" and cat == "splash":
                label = "sign_up"
            if label == "rating" and cat == "splash":
                label = "image"
            if label == "sort" and cat == "splash":
                label = "button"
            label_image = android_label_map[label]["img"]
            if label_image is None:
                continue
            base_label_image = label_image.copy()
            label_text = android_label_map[label]["text"]
            label_resize = android_label_map[label]["resize"]
            if label in ["navigation_dots", "sort, heart_icon", "sort", "rating", "filter"]:
                base_shade = 189
            else:
                base_shade = 224
            # print(label)
            label_image, fw = element_resize(label_image, w, h, label_resize, base_shade)
            if label in ["image", "icon"]:
                cv2.line(label_image, (0, 0), (w - 1, h - 1), (79, 79, 79), thickness=1)
                cv2.line(label_image, (0, h - 1), (w - 1, 0), (79, 79, 79), thickness=1)
            if label_resize == 4:
                fw = 0
            if label_text is not None:
                text = label_text[0]
                if label in element_counted:
                    c = element_counted[label]
                    if len(label_text) > c:
                        text = label_text[c]
                label_image, text_ignored = add_text_pil(label_image, text, 0, label_resize, fw)
                if text_ignored and label_resize == 3:
                    label_image, fw = element_resize(base_label_image, w, h, flag=2, base_shade=189)
            try:
                base_image[y : y + h, x : x + w, :] = label_image
                if label in element_counted:
                    element_counted[label] += 1
                else:
                    element_counted[label] = 1
            except Exception as e:
                print(e)
    # base_image = cv2.rectangle(base_image, (0,0), (img_w-1, img_h-1), (0,0,0), thickness=2)
    cv2.imwrite(dst_file_path, base_image)


def find_font_scale_pil(fontScale, h, label_text, w, reduce_text):
    given_fontScale = fontScale
    font = ImageFont.truetype(fontpath, fontScale)
    textsize = font.getsize(label_text)
    tw = textsize[0]
    th = textsize[1]
    font_scale_reduction = 0
    while fontScale > 1 and (th > h * 0.90 or tw > w * 0.85):
        fontScale -= 1
        font_scale_reduction += 1
        if reduce_text and font_scale_reduction > 5:
            label_text, reduced = reduce_text_size(label_text)
            font_scale_reduction = 0
            if reduced:
                fontScale = given_fontScale
        font = ImageFont.truetype(fontpath, fontScale)
        textsize = font.getsize(label_text)
        th = textsize[1]
        tw = textsize[0]
    return font, textsize, label_text, fontScale


def reduce_text_size(text):
    if len(text) < 9:
        return text, False
    new_length = len(text) - 4
    r = text[:new_length]
    return r, True


def add_text_pil(label_image, label_text, align, label_resize, fw):
    if label_text is not None and len(label_text) > 0:
        reduce_text = False
        if label_text == "lorem ipsum dolor":
            reduce_text = True
        if "/\\" in label_text:
            fontScale = default_fontScale - 4
            reduce_text = True
        else:
            fontScale = default_fontScale
        label_image = label_image.astype(np.uint8)
        h = label_image.shape[0]
        w = label_image.shape[1]
        accesible_w = w - fw if label_resize == 3 and fw > 0 and w > fw else w
        font, textsize, label_text, fontScale = find_font_scale_pil(fontScale, h, label_text, accesible_w, reduce_text)
        if label_resize == 3 and fontScale < 12:
            return label_image, True
        if align == 0:  # center
            textX = int(((label_image.shape[1] - fw) - textsize[0]) / 2) + fw
            textY = int((label_image.shape[0] - textsize[1]) / 2)
            img_pil = Image.fromarray(label_image)
            draw = ImageDraw.Draw(img_pil)
            draw.text((textX, textY - 1), label_text, font=font, fill=(51, 51, 51, 0))
            label_image = np.array(img_pil)
    return label_image, False


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_w", "-w", help="image width", default=360)
    parser.add_argument("--img_h", "-h", help="image height", default=576)
    parser.add_argument(
        "--json_file_location", "-l", help="path to category/json_files", default="../../data/transformerLayout"
    )
    parser.add_argument("--human_generated", "-r", help="use if the files are generated by humans", action="store_true")
    parser.add_argument(
        "--android_element_mapping_file",
        "-m",
        help="path to android element mapping csv",
        default="../resources/ui_labels_android_map.csv",
    )
    parser.add_argument(
        "--android_elements_base_path",
        "-a",
        help="path to android element images",
        default="../../data/android_elements",
    )
    parser.add_argument("--destination_folder", "-d", help="path to destination", default="../../data/final")
    parser.add_argument("--font_path", "-f", help="complete path to font ttf file")
    parser.add_argument(
        "--label_hierarchy_file",
        "-h",
        help="path to label hierarchy csv file",
        default="../resources/ui_labels_hierarchy.csv",
    )
    args = parser.parse_args()

    img_w = args.img_w
    img_h = args.img_h
    json_location = args.json_file_location
    android_element_mapping_file = args.android_element_mapping_file
    android_elements_base_path = args.android_elements_base_path
    dst_folder = args.destination_folder
    # fontpath = "C:/Users/Nishit/AppData/Local/Microsoft/Windows/Fonts/Roboto-Regular.ttf"
    fontpath = args.font_path
    label_hierarchy_file = args.label_hierarchy_file
    real = args.human_generated

    level = 3
    default_fontScale = 20
    label_hierarchy_map = UILabelFileManager.get_hierarchy_label_map(label_hierarchy_file, level)
    android_label_map = load_all_ui_images()

    for dir in os.listdir(json_location):
        dir_path = os.path.join(json_location, dir)
        if os.path.isdir(dir_path):
            count = 0
            print(dir)
            for file in os.listdir(dir_path):
                if ".json" in file:
                    print(file)
                    json_path = os.path.join(dir_path, file)
                    dst_folder_path = os.path.join(dst_folder, dir)
                    if not os.path.exists(dst_folder_path):
                        os.mkdir(dst_folder_path)
                    dst_file_path = os.path.join(dst_folder_path, f"{str(count)}_{str(real)}.jpg")
                    # dst_file_path = os.path.join(dst_folder, file[:-5]+".jpg")
                    elements = get_elements(json_path, real)
                    try:
                        create_img(elements, dst_file_path, dir, real)
                        print(file, count)
                    except Exception as e:
                        print(e, file)
                    count += 1
