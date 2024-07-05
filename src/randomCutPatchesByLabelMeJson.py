import os
import cv2
import numpy as np
import json
import random
import shutil
from tqdm import tqdm

"""
根据已有的LabelMeJson 标注信息 裁剪图片 用于生成YOLO训练图片
"""

def parse_json():
    pass

def create_dir(target_dir):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)


def calculate_iou(box1, box2):
    # box1 = (x1, y1, x2, y2), box2 = (x1, y1, x2, y2)
    tlx1, tly1 = min(box1[0], box1[2]), min(box1[1], box1[3])
    brx1, bry1 = max(box1[0], box1[2]), max(box1[1], box1[3])

    tlx2, tly2 = min(box2[0], box2[2]), min(box2[1], box2[3])
    brx2, bry2 = max(box2[0], box2[2]), max(box2[1], box2[3])

    # 计算交集的坐标
    xA = max(tlx1, tlx2)
    yA = max(tly1, tly2)
    xB = min(brx1, brx2)
    yB = min(bry1, bry2)

    # 计算交集面积
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # 计算两个框的面积
    box1Area = (brx1 - tlx1 + 1) * (bry1 - tly1 + 1)
    box2Area = (brx2 - tlx2 + 1) * (bry2 - tly2 + 1)

    # 计算并集面积
    unionArea = box1Area + box2Area - interArea
    targetArea = min(box1Area, box2Area)

    # 计算 IoU
    # iou = interArea / unionArea
    iou = interArea / targetArea

    return iou

class LabelMe:
    def __init__(self, image_dir_, image_name_, label_class_):
        self.label_class = label_class_
        self.shapes = []
        self.image_dir = image_dir_
        self.image_pth = image_name_
        img_mat = cv2.imread(os.path.join(image_dir_, image_name_))
        self.image_height, self.image_width, _ = img_mat.shape

    def shapeAppend(self, label: int, points):
        left, top, right, bottom = points
        # points = np.array([(left, top), (right, top), (right, bottom), (left, bottom)])
        points = np.array([(left, top), (right, bottom)])
        points = points.reshape(-1, 2).astype(int).tolist()

        for point in points:
            point[0] = min(max(0, point[0]), self.image_width)
            point[1] = min(max(0, point[1]), self.image_height)

        self.shapes.append({
            "label": self.label_class[int(label)],
            "points": points,
            # "shape_type": "polygon",
            "shape_type": "rectangle"
        })

    def toJson(self):
        res_dict = {
            "shapes": self.shapes,
            "imagePath": self.image_pth,
            "imageData": None,
            "imageHeight": self.image_height,
            "imageWidth": self.image_width
        }

        with open(os.path.join(self.image_dir, self.image_pth.replace(".jpg", ".json")), "w") as fp:
            json.dump(res_dict, fp)


def gen_patch_with_dent(img_dir_, save_dir_, patch_size_=1000, random_patch_num_=3, mini_patch_area_=600*600):
    img_dir = img_dir_
    json_lst = [f for f in os.listdir(img_dir) if f.endswith(".json")]

    save_dir = save_dir_  # 保存切出的patch 和 labelme格式的标注文件
    create_dir(save_dir)

    for json_file in tqdm(json_lst):
        image_name = json_file.replace(".json", ".jpg")
        org_img = cv2.imread(os.path.join(img_dir, image_name))
        imgH, imgW, _ = org_img.shape

        with open(os.path.join(img_dir, json_file), "r") as fp:
            content = json.load(fp)

        if len(content["shapes"]) > 1:
            continue

        for bbox_id, obj in enumerate(content["shapes"]):
            # 默认是 矩形 两点框
            (tlx, tly), (brx, bry) = obj["points"]
            boxH, boxW = bry - tly, brx - tlx
            assert max(boxH, boxW) < patch_size, "Error Dent bigger than patch!!!"
            xRange = patch_size - boxW
            yRange = patch_size - boxH
            patch_counter = 0
            patch_left, patch_right, patch_top, patch_bottom = 0, 0, 0, 0
            while (patch_counter < random_patch_num):
                x_shift = random.randint(0, int(xRange))
                x_shift += min(0, tlx - x_shift)
                y_shift = random.randint(0, int(yRange))
                y_shift += min(0, tly - y_shift)
                patch_left = int(tlx - x_shift)
                patch_top = int(tly - y_shift)
                patch_right = min(patch_left + patch_size, imgW)
                patch_bottom = min(patch_top + patch_size, imgH)

                patch_area = (patch_right - patch_left) * (patch_bottom - patch_top)
                if patch_area < mini_patch_area:
                    continue
                patch = org_img[patch_top:patch_bottom, patch_left:patch_right]
                save_name = f"{image_name.replace('.jpg', '')}_box{bbox_id}_counter{patch_counter}.jpg"
                assert cv2.imwrite(os.path.join(save_dir, save_name), patch), "Error, Image Save Failed"

                new_tlx = int(x_shift)
                new_tly = int(y_shift)
                new_brx = int(brx - (tlx - x_shift))
                new_bry = int(bry - (tly - y_shift))
                # patch_ = cv2.rectangle(patch.copy(), (new_tlx, new_tly), (new_brx, new_bry), (0,0,255), 3)
                # cv2.imshow("patch", cv2.resize(patch_, None, fx=0.5, fy=0.5))
                # cv2.waitKey(0)
                patch_counter += 1

                label_obj = LabelMe(save_dir, save_name, CLASSES)
                # TODO: 有可能一个patch里有多个dent，暂时先手动剔除
                label_obj.shapeAppend(0, [new_tlx, new_tly, new_brx, new_bry])
                label_obj.toJson()


def gen_patch_no_dent(img_dir_, save_dir_, patch_size_=1000, random_patch_num_=3):
    img_dir = img_dir_
    img_lst = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    save_dir = save_dir_  # 保存切出的patch 和 labelme格式的标注文件
    create_dir(save_dir)

    for image_name in tqdm(img_lst):
        # if image_name != "0_1_0.jpg": continue
        json_file = image_name.replace(".jpg", ".json")
        org_img = cv2.imread(os.path.join(img_dir, image_name))
        imgH, imgW, _ = org_img.shape

        xRange = imgW - patch_size_
        yRange = imgH - patch_size_

        patch_counter = 0
        while patch_counter<random_patch_num_:
            tlx = random.randint(0, int(xRange))
            tly = random.randint(0, int(yRange))
            brx = tlx + patch_size
            bry = tly + patch_size

            if os.path.isfile(os.path.join(img_dir, json_file)):
                # 有标注压痕
                with open(os.path.join(img_dir, json_file), "r") as fp:
                    content = json.load(fp)

                is_contained = False
                for obj in content["shapes"]:
                    box1 = np.array(obj["points"]).reshape(-1).astype(int)
                    box2 = (tlx, tly, brx, bry)
                    # org_img = cv2.rectangle(org_img, box1[:2], box1[2:], (0,0,255), 5)
                    # org_img = cv2.rectangle(org_img, box2[:2], box2[2:], (0,255,0), 5)
                    # cv2.imshow("draw", cv2.resize(org_img, None, fx=0.2, fy=0.2))
                    # cv2.waitKey(0)

                    iou = calculate_iou(box1, box2)

                    if iou > 0.1:
                        is_contained = True

                if is_contained:
                    continue
            else:
                # 没有标注压痕
                pass

            patch_counter += 1

            patch = org_img[tly:bry, tlx:brx]
            # cv2.imshow("patch", patch)
            # cv2.waitKey(0)
            save_name = f"{image_name.replace('.jpg', '')}_counter{patch_counter}.jpg"
            assert cv2.imwrite(os.path.join(save_dir, save_name), patch), "Error, Image Save Failed"

            label_obj = LabelMe(save_dir, save_name, CLASSES)
            label_obj.toJson()


def only_gen_lableme_json(img_dir_):
    for img_file in os.listdir(img_dir_):
        if not img_file.endswith(".jpg"): continue
        label_obj = LabelMe(img_dir_, img_file, CLASSES)
        label_obj.toJson()


# 随机裁剪图片，生成labelme训练集
# parse labelme json
# 围绕 压痕位置 随机产生 patch  每个压痕随机产生5个子图
# 生成 labelme 标注文件
# 生成 yolo 标注文件
# 还要产生一些没有压痕的图片作为负例
CLASSES = ['dent', ]
if __name__ == "__main__":
    patch_size = 1200
    random_patch_num = 10
    mini_patch_area = 600*600

    save_img_dir = r"E:\DataSets\dents_det\cut_patches\01"  # 保存切出的patch 和 labelme格式的标注文件
    create_dir(save_img_dir)

    # src_img_dir = r"E:\DataSets\dents_det\cut_cells\with_dent"
    # gen_patch_with_dent(src_img_dir, save_img_dir, patch_size, random_patch_num, mini_patch_area)

    # src_img_dir = r"E:\DataSets\dents_det\org_2D\with_label"
    # gen_patch_no_dent(src_img_dir, save_img_dir, patch_size, random_patch_num)

    only_gen_lableme_json(r"E:\DataSets\dents_det\cut_patches\no_dent")



    print("done")