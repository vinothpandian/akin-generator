import os, sys

foo_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(foo_dir, '..', '..')))

from src.utils import Utils


class UILabelFileManager:

    @staticmethod
    def get_ui_hierarchy_map(file):
        data = Utils.read_file(file)
        heirarchy_map = {}
        for row in data:
            s = row.replace("\n", "").split(";")
            labels = s[1:]
            heirarchy_map[s[0]] = labels
        return heirarchy_map

    @staticmethod
    def get_label_counts(file, sorted_flag=False):
        data = Utils.read_file(file)
        if sorted_flag:
            sorted_labels = []
            for row in data:
                s = row.split(";")
                sorted_labels.append(s[0])
            return sorted_labels
        else:
            label_count = {}
            for row in data:
                s = row.split(";")
                label_count[s[0]] = int(s[1])
            return label_count

    @staticmethod
    def get_sorted_labels_for_hierarchy(label_count_file, heirarchy_file, level=0, with_count=False):
        if level == 0:
            return UILabelFileManager.get_label_counts(label_count_file, sorted_flag=True)
        label_count = UILabelFileManager.get_label_counts(label_count_file)
        hierarchy_map = UILabelFileManager.get_ui_hierarchy_map(heirarchy_file)
        hierarchy_count = {}
        for k, v in hierarchy_map.items():
            label = v[-1] if len(v) < level - 1 else v[level - 1]
            if k not in label_count.keys():
                count = 0
                print(f"{k} not in label count")
            else:
                count = label_count[k]
            if label in hierarchy_count:
                hierarchy_count[label] += count
            else:
                hierarchy_count[label] = count
        heirarchy_count_list = list(hierarchy_count.items())
        heirarchy_count_list.sort(key=lambda x: x[1], reverse=True)
        print(heirarchy_count_list)
        return (
            heirarchy_count_list
            if with_count
            else [v[0] for v in heirarchy_count_list]
        )

    @staticmethod
    def get_sorted_labels_based_on_pairings(level):
        if level == 3:
            return ['text', 'image', 'log_in', 'sign_up', 'username', 'password', 'icon', 'forgot', 'sm_button',
                    'button', 'box', 'privacy', 'check', 'name', 'navigation_dots', 'number', 'selector', 'search',
                    'edit_number', 'edit_string', 'filter', 'top_bar', 'heart_icon', 'sort', 'rating',
                    'bottom_bar', 'card_add', 'buy', 'other']
        elif level == 1:
            return ['text', 'image', 'button', 'edit', 'check', 'box', 'bar', 'other']

    @staticmethod
    def get_hierarchy_label_map(file, level=1):
        if level < 1:
            return {}
        hierarchy_map = UILabelFileManager.get_ui_hierarchy_map(file)
        hierarchy_label_map = {}
        for k, v in hierarchy_map.items():
            label = v[-1] if len(v) < level - 1 else v[level - 1]
            hierarchy_label_map[k] = label
        return hierarchy_label_map
