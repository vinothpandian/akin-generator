import json
import os
import shutil


class Utils:

    @staticmethod
    def copy_files(names, src, dst, overwrite=True):
        if not os.path.exists(dst):
            os.makedirs(dst)
        for name in names:
            try:
                src_path = os.path.join(src, name)
                dst_path = os.path.join(dst, name)
                if overwrite or not os.path.exists(dst_path):
                    shutil.copyfile(src_path, dst_path)
            except Exception as e:
                print(e)

    @staticmethod
    def copy_json_files(src, dst):
        if os.path.exists(dst):
            img_names = os.listdir(dst)
            json_names = []
            for name in img_names:
                if ".directory" not in name:
                    print(name)
                    json_name = f"{name[:-3]}json"
                    json_names.append(json_name)
            Utils.copy_files(json_names, src, dst)

    @staticmethod
    def read_file(file):
        data = []
        if os.path.exists(file):
            with open(file, "r") as f:
                data = f.readlines()
        return data

    @staticmethod
    def read_json_file(file):
        data = []
        if os.path.exists(file):
            with open(file, "r") as f:
                data = json.load(f)
        return data