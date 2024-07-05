import os
import shutil
import json
import random
import cv2
import numpy as np

CLASSES = ["dent", ]
# CLASSES = ["background", "cell", ]
# CLASSES = ["FK01", "FK02", "FK03", "924"]
# CLASSES = ["rotate_0", "rotate_180"]

def createDir(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    else:
        shutil.rmtree(target_dir)
        os.makedirs(target_dir)


def parseJson2YoloTxt(json_pth, save_pth):
    img = cv2.imread(json_pth.replace(".json", ".jpg"))
    imgh, imgw, _ = img.shape
    with open(json_pth, "r") as j:
        json_info = json.load(j)

    with open(save_pth, "w") as t:
        label_str = ""
        for obj in json_info["shapes"]:
            bbox_type = obj["shape_type"]
            label_str += f"{CLASSES.index(obj['label'])} "
            # det
            if(obj["shape_type"] == "polygon"):
                r = cv2.minAreaRect(np.array(obj["points"]).reshape(-1,1,2).astype(int))
                r = cv2.boxPoints(r)
                tlx, tly, brx, bry = min(r[:,0]), min(r[:,1]), max(r[:,0]), max(r[:,1])
                coords = np.array([[tlx, tly], [brx, bry]], dtype=np.float32)
            else:
                box = np.array(obj["points"])
                tlx = box[:, 0].min()
                tly = box[:, 1].min()
                brx = box[:, 0].max()
                bry = box[:, 1].max()
                coords = np.array([(tlx, tly), (brx, bry)])
            cx = coords[:, 0].mean()
            cy = coords[:, 1].mean()
            w = coords[1, 0] - coords[0, 0]
            h = coords[1, 1] - coords[0, 1]
            label_str += f"{cx/imgw:.6f} {cy/imgh:.6f} {w/imgw:.6f} {h/imgh:.6f}"

            # # poly
            # coords = [(x/imgw, y/imgh) for x, y in obj['points']]
            # for normal_x, normal_y in coords:
            #     label_str += f" {normal_x:.4f} {normal_y:.4f}"

            label_str += "\n"
        t.write(label_str)

    return

def baseYoloLabel2show(img_jpg_pth, label_txt_pth, save_pth=""):
    class BBox:
        def __init__(self, yolo_label_line, imgH, imgW):
            self.cls = yolo_label_line[0]
            cx, cy, w, h = np.array(yolo_label_line[1:]).astype(np.float32)
            width, height = w*imgW, h*imgH
            cx, cy = cx*imgW, cy*imgH
            self.points = np.array([cx-width/2, cy-height/2, cx+width/2, cy+height/2]).reshape(-1, 2).astype(int)

    img = cv2.imread(img_jpg_pth)
    imgh, imgw, _ = img.shape
    bbox_lst = []
    with open(label_txt_pth, "r") as fp:
        lines = fp.readlines()
        lines = [line.split() for line in lines]
        bbox_lst = [BBox(line, imgh, imgw) for line in lines]

    for bbox in bbox_lst:
        # cv2.polylines(img, [bbox.points], True, (0,0,255), 2)
        cv2.circle(img, bbox.points[0], 5, (0,255,0), 5)
        cv2.rectangle(img, bbox.points[0], bbox.points[1], (0,0,255), 3)

    if (save_pth!=""):
        cv2.imwrite(save_pth, img)
    else:
        cv2.imshow("test", img)
        cv2.waitKey(0)
    return

# TODO: 旋转框坐标有问题！！！有的样本没有标注框！！！  cx cy w h   tlx tly brx bry
'''
# labelme json to yolo obb
# 默认 图片 和 标注json 在同一目录下
# 保存文件结构：
- yolo_obb
    - images
        - train
        - val
        - test
    - labels
        - train
        - val
        - test
'''
if __name__ == "__main__":

    # 查看标注结果
    img_dir = r"E:\DataSets\dents_det\cut_patches\yolo\images"
    save_dir = r"E:\DataSets\dents_det\cut_patches\yolo_show"
    for root_dir, sub_dir, file_lst in os.walk(img_dir):
        for file in file_lst:
            if not file.endswith(".jpg"): continue
            baseYoloLabel2show(
                os.path.join(root_dir, file),
                os.path.join(root_dir.replace("images", "labels"), file.replace(".jpg", ".txt")),
                os.path.join(save_dir, file))

    # # 转换标注信息
    # # TODO: 优化，兼容更多场景！！！
    # org_dir = r"E:\DataSets\dents_det\cut_patches\with_dent"
    # json_files = [f for f in os.listdir(org_dir) if f.endswith(".json")]
    # save_dir_name = "yolo_det"
    #
    # train_num = int(0.8*len(json_files))
    # random.shuffle(json_files)
    # train_json_lst = json_files[:train_num]
    # val_json_lst = json_files[train_num:]
    # print(f"Train Data Num:{len(train_json_lst)} Val Data Num:{len(val_json_lst)}")
    #
    # last_dir = org_dir.split(os.sep)[-1]
    # train_image_dir = org_dir.replace(last_dir, f"{save_dir_name}/images/train")
    # createDir(train_image_dir)
    # train_label_dir = org_dir.replace(last_dir, f"{save_dir_name}/labels/train")
    # createDir(train_label_dir)
    # for train_json in train_json_lst:
    #     image_name = train_json.replace(".json", ".jpg")
    #     shutil.copyfile(os.path.join(org_dir, image_name),
    #                     os.path.join(train_image_dir, image_name))
    #     parseJson2YoloTxt(os.path.join(org_dir, train_json),
    #                       os.path.join(train_label_dir, train_json.replace(".json", ".txt")))
    #
    # val_image_dir = org_dir.replace(last_dir, f"{save_dir_name}/images/val")
    # createDir(val_image_dir)
    # val_label_dir = org_dir.replace(last_dir, f"{save_dir_name}/labels/val")
    # createDir(val_label_dir)
    # for val_json in val_json_lst:
    #     image_name = val_json.replace(".json", ".jpg")
    #     shutil.copyfile(os.path.join(org_dir, image_name),
    #                     os.path.join(val_image_dir, image_name))
    #     parseJson2YoloTxt(os.path.join(org_dir, val_json),
    #                       os.path.join(val_label_dir, val_json.replace(".json", ".txt")))





    print("done")