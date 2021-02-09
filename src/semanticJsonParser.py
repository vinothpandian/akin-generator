import argparse
import json
import os
import sys

import cv2
import numpy as np

from .colorMapper import ColorMapper
from .uiLabelFileManager import UILabelFileManager

foo_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(foo_dir, "..", "..")))


class SemanticJsonParser:
    @staticmethod
    def get_list(src):
        img_json_dict = {}
        if os.path.exists(src):
            for name in os.listdir(src):
                if ".json" in name:
                    with open(os.path.join(src, name), "r") as f:
                        data = f.read()
                        data = json.loads(data)
                        k = name[:-5]
                        SemanticJsonParser.add_to_dict(k, img_json_dict, "json", data)
                elif ".jpg" in name:
                    img = cv2.imread(os.path.join(src, name), cv2.IMREAD_COLOR)
                    k = name[:-4]
                    SemanticJsonParser.add_to_dict(k, img_json_dict, "img", img)
        return img_json_dict

    @staticmethod
    def add_to_dict(k, dict, name, value):
        if k in dict.keys():
            dict[k].update({name: value})
        else:
            dict[k] = {name: value}

    @staticmethod
    def read_json(data, label_hierarchy_map, max_elements=-1):
        try:
            element_bounds = []
            if data is not None:
                shapes = data["shapes"]
                flags = data["flags"]
                # done = flags["done"]
                for shape in shapes:
                    label = shape["label"]
                    if label in ["separator_line", "separator_line_v"]:
                        continue
                    if label_hierarchy_map:
                        if label in label_hierarchy_map.keys():
                            label = label_hierarchy_map[label]
                        else:
                            label = "other"
                    points = shape["points"]
                    bounds = [points[0][0], points[0][1], points[1][0], points[1][1]]
                    area = (points[1][0] - points[0][0]) * (points[1][1] - points[0][1])
                    element_bounds.append([label, bounds, area])

            element_bounds.sort(key=lambda x: x[2], reverse=True)
            # element_bounds = SemanticJsonParser.remove_sub_elements(element_bounds)
            if max_elements > 0:
                element_bounds = element_bounds[:max_elements]
            # element_bounds.sort(key=lambda x: (x[1][0], x[1][1]))
            element_bounds = [[x[0], x[1]] for x in element_bounds]
            # print(element_bounds)
            return element_bounds
        except Exception as e:
            print(e)

    @staticmethod
    def remove_sub_elements(element_bounds):
        within = []
        nothing_within = [
            "text",
            "log_in",
            "sign_up",
            "username",
            "password",
            "icon",
            "forgot",
            "sm_button",
            "button",
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
            "cart_add",
            "buy",
        ]
        remove = ["other"]
        for i, (label, bounds, area) in enumerate(element_bounds):
            if label in remove:
                within.append(i)
            elif label in nothing_within:
                j = i + 1
                while j < len(element_bounds):
                    # sub_label = element_bounds[j][0]
                    if j not in within:
                        sub_bounds = element_bounds[j][1]
                        sub_area = element_bounds[j][2]
                        if SemanticJsonParser.is_within(bounds, sub_bounds, sub_area):
                            within.append(j)
                    j += 1
        within = sorted(within, reverse=True)
        for index in within:
            element_bounds.pop(index)
        return element_bounds

    @staticmethod
    def is_within(large, small, small_area):
        try:
            dx = min(large[2], small[2]) - max(large[0], small[0])
            dy = min(large[3], small[3]) - max(large[1], small[1])
            if dx < 0 and dy < 0:
                return False
            else:
                overlap_area = dx * dy
                if overlap_area / small_area > 0.96:
                    return True
                else:
                    return False
        except Exception as e:
            print(e)
            return True

    @staticmethod
    def mark_image(img, meta, label_hierarchy_map, label_color_map, border=True, gan_output=False, skip_box=False):
        edge = 2
        img = SemanticJsonParser.set_image_background(img)
        for box in meta:
            thickness = -1
            color, element = SemanticJsonParser.get_element_color(
                box[0], label_color_map, label_hierarchy_map, gan_output
            )
            # print(color, element)
            if skip_box and "box" in element:
                continue
            if "box" in element:
                thickness = 2
            bounds = box[1]
            start = (int(bounds[0] + edge), int(bounds[1] + edge))
            end = (int(bounds[2] - edge), int(bounds[3] - edge))
            if color and SemanticJsonParser.valid_area(start, end):
                col = (int(color[0]), int(color[1]), int(color[2]))
                # img = cv2.UMat(img)
                if border:
                    img = cv2.rectangle(img, start, end, col, thickness=2)
                else:
                    img = cv2.rectangle(img, start, end, col, thickness=thickness)
        return img

    @staticmethod
    def set_image_background(img):
        start = (0, 0)
        # end = (360, 598)
        end = (360, 640)
        color = (255, 255, 255)
        return cv2.rectangle(img, start, end, color, -1)

    @staticmethod
    def parse_data(img_json_dict, label_hierarchy_map, label_color_map, dst):
        for key, value in img_json_dict.items():
            try:
                json = value["json"]
                meta = SemanticJsonParser.read_json(json, label_hierarchy_map)
                img_z = np.zeros((640, 360, 3))
                # img = SemanticJsonParser.mark_image(value["img"], meta, None, label_color_map, border=True, skip_box=True)
                img = SemanticJsonParser.mark_image(img_z, meta, None, label_color_map, border=True, skip_box=True)
                img = SemanticJsonParser.remove_top_bottom(img)
                SemanticJsonParser.save_img(img, path=os.path.join(dst, key + ".jpg"))
            except Exception as e:
                print(key, e)

    @staticmethod
    def remove_top_bottom(img):
        return img[21:597, :]

    @staticmethod
    def valid_area(point_a, point_b):
        x = point_b[0] - point_a[0]
        y = point_b[1] - point_a[1]
        area = x * y
        rel = area / 230400
        if rel < 0.4:
            return True
        else:
            return False

    @staticmethod
    def save_img(img, path):
        cv2.imwrite(path, img)

    @staticmethod
    def get_element_color(element, label_color_map, label_hierarchy_map=None, gan_output=False):
        if gan_output:
            element = element[0]
        # print(element)
        if label_hierarchy_map:
            if element in label_hierarchy_map.keys():
                label = label_hierarchy_map[element]
            else:
                label = "other"
        else:
            label = element
        if label in label_color_map.keys():
            color = label_color_map[label]
            return color, label
        else:
            return None, label

    @staticmethod
    def convert_bb_to_image(
        predictions_bb,
        predictions_label,
        img_h,
        img_w,
        label_hierarchy_map,
        label_color_map,
        dst,
        ctr=0,
        combined=False,
    ):
        print(label_hierarchy_map)
        print(label_color_map)
        all_images = np.zeros((predictions_bb.shape[0], img_h, img_w, 3))
        for i in range(len(predictions_bb)):
            prediction_bb = predictions_bb[i]
            prediction_label = predictions_label[i]
            meta = []
            for j in range(len(prediction_bb)):
                meta.append([prediction_label[j], prediction_bb[j]])
            img = np.zeros((img_h, img_w, 3))
            img = SemanticJsonParser.mark_image(img, meta, label_hierarchy_map, label_color_map, gan_output=True)
            if combined:
                all_images[i] = img
            else:
                img = SemanticJsonParser.remove_top_bottom(img)
                SemanticJsonParser.save_img(
                    img, path=os.path.join(dst, "predicted_" + str(ctr) + "_" + str(i) + ".jpg")
                )
        if combined:
            all_images = all_images.reshape((-1, img_w, 3))
            SemanticJsonParser.save_img(all_images, path=os.path.join(dst, "predicted_epoch" + str(ctr) + ".jpg"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_folder", "-s", help="path to annotation json files", default="../../../data/folder_labels"
    )
    parser.add_argument(
        "--destination_path", "-d", help="path to destination folder", default="../../../data/semantic_all"
    )
    parser.add_argument(
        "--color_map_file",
        "-c",
        help="path to file for mapping of color to UI element type",
        default="../../data/ui_labels_color_map.csv",
    )
    parser.add_argument(
        "--label_hierarchy_file",
        "-h",
        help="path to label hierarchy csv file",
        default="../../../data/ui_labels_hierarchy.csv",
    )

    args = parser.parse_args()
    all_elements = {""}
    src = args.source_folder
    dst = args.destination_path
    label_color_map_file = args.color_map_file
    label_hierarchy_file = args.label_hierarchy_file

    level = 3

    label_color_map = ColorMapper.read_label_color_map(label_color_map_file)
    label_hierarchy_map = UILabelFileManager.get_hierarchy_label_map(label_hierarchy_file, level)

    for dp in os.listdir(src):
        dp_path = os.path.join(src, dp)
        img_json_dict = SemanticJsonParser.get_list(dp_path)
        dst1 = os.path.join(dst, dp)
        if not os.path.exists(dst1):
            os.makedirs(dst1)
        SemanticJsonParser.parse_data(img_json_dict, label_hierarchy_map, label_color_map, dst1)
