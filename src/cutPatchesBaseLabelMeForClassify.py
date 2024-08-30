import os
import cv2
import json
import numpy as np
import random
import shutil
import time

def createDir(target_dir):
    if(os.path.isdir(target_dir)):
        shutil.rmtree(target_dir)

    os.makedirs(target_dir)

def getRandomPatch(img_w, img_h, tcx_, tcy_, tw_, th_, patch_size_, margin_ratio_):
    margin_size = max(tw_, th_) * margin_ratio_
    # 在 (cx, cy) 为圆心  (patch_size - margin_size)/2 为半径的范围内随机产生 Patch 的中心
    x_shift = random.uniform(-(patch_size_-margin_size)/2, (patch_size_-margin_ratio_)/2)
    y_shift = random.uniform(-(patch_size_ - margin_size) / 2, (patch_size_ - margin_ratio_) / 2)
    patch_center_x = tcx_ + x_shift
    patch_center_y = tcy_ + y_shift
    patch_left = max(0, int(patch_center_x-patch_size_))
    patch_top = max(0, int(patch_center_y-patch_size_))
    patch_right = min(img_w, int(patch_center_x+patch_size_))
    patch_bottom = min(img_h, int(patch_center_y+patch_size_))
    if((patch_left<0) or (patch_top<0) or (patch_right<0) or (patch_bottom<0)):
        print()
    return patch_left, patch_top, patch_right, patch_bottom

'''
- 用于 传统视觉在大图像上筛选缺陷位置，用分类模型进一步筛选缺陷
1. 是LabelMe标注缺陷位置
2. 使用脚本随机切出缺陷位置
'''
if __name__ == "__main__":
    # src_dir = r"E:\DataSets\edge_crack\original_iqc"
    src_dir = r"E:\DataSets\edge_crack\original_iqc\org_img"
    dst_dir = r"E:\DataSets\edge_crack\cut_patches_0822"
    save_class = "crack"
    save_dir = os.path.join(dst_dir, save_class)
    createDir(save_dir)
    patch_size = 60
    margin_ratio = 1  # 几倍目标物体的大小 >=1
    rand_repeat_num = 3

    json_lst = [f for f in os.listdir(src_dir) if f.endswith(".json")]
    for json_file in json_lst:
        img_name = json_file.replace(".json", ".jpg")
        img_pth = os.path.join(src_dir, img_name)
        img = cv2.imread(img_pth)
        imgH, imgW, _ = img.shape

        with open(os.path.join(src_dir, json_file), 'r') as fp:
            content = json.load(fp)
        use_bbox = [box for box in content["shapes"] if box["label"]==save_class]

        for box in use_bbox:
            b = np.array(box["points"])
            xmin = b[:, 0].min()
            xmax = b[:, 0].max()
            ymin = b[:, 1].min()
            ymax = b[:, 1].max()
            cx = (xmin+xmax)/2
            cy = (ymin+ymax)/2
            w = xmax - xmin
            h = ymax - ymin

            for i in range(rand_repeat_num):
                (plt_x, plt_y, pbr_x, pbr_y) = getRandomPatch(imgW, imgH, cx, cy, w, h, patch_size, margin_ratio)
                patch = img[plt_y:pbr_y, plt_x:pbr_x]
                # cv2.imshow("test", patch)
                # cv2.waitKey(600)
                save_name = f"{img_name.replace('.jpg', '')}_rp_{i}_t{int(time.time()*1000)%100000}.jpg"
                if(random.uniform(0, 1)>0.5):
                    patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(os.path.join(save_dir, save_name), patch)

    print()




    print("done")