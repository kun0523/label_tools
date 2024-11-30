import os
import re
import shutil
import random
import cv2

if __name__ == "__main__":

    org_dir = r"E:\DataSets\vacuum_package\test0918\org02"
    save_dir = r"E:\DataSets\vacuum_package\test0918\cutPatches02"
    # img_name = r"2024-9-18-0-10-50_OK.jpg"
    imgs = [f for f in os.listdir(org_dir) if f.endswith(".jpg")]
    roi = [350, 10, 1550, 810]

    for img_name in imgs:
        img = cv2.imread(os.path.join(org_dir, img_name))
        part = img[roi[1]:roi[3], roi[0]:roi[2]]

        cv2.imwrite(os.path.join(save_dir, img_name), part)






    print("done")