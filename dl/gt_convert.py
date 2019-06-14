import os
import random
import shutil

import sys

GT_from_PATH = sys.argv[1]   # where json is
GT_to_PATH = sys.argv[2]   # where you want to save


def copy_file(from_dir, to_dir, Name_list):
    if not os.path.isdir(to_dir):
        os.mkdir(to_dir)
    for name in Name_list:
        try:
            if not os.path.isfile(os.path.join(from_dir, name)):
                print("{} is not existed".format(os.path.join(from_dir, name)))
            shutil.copy(os.path.join(from_dir, name), os.path.join(to_dir, name))
        except:
            pass

if __name__ == '__main__':
    filepath_list = os.listdir(GT_from_PATH)
    for i, file_path in enumerate(filepath_list):
        gt_path = "{}/{}_gt.png".format(os.path.join(GT_from_PATH, filepath_list[i]), file_path[:-5])
        gt_name = ["{}_gt.png".format(file_path[:-5])]
        gt_file_path = os.path.join(GT_from_PATH, file_path)
        copy_file(gt_file_path, GT_to_PATH, gt_name)
    print ('Done!')