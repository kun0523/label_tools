import os
import shutil
import random

if __name__ == "__main__":
    img_dir = r"E:\DataSets\pol\pick_bright\UNKNOWN"
    img_files = os.listdir(img_dir)
    keep_file_num = 200
    del_files = random.choices(img_files, k=len(img_files)-keep_file_num)
    for f in del_files:
        del_file = os.path.join(img_dir, f)
        if os.path.exists(del_file):
            os.remove(os.path.join(img_dir, f))


    print("done")