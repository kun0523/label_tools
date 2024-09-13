import os
import cv2
import numpy as np
import json
import shutil

from ultralytics import YOLO

def createDir(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    else:
        shutil.rmtree(target_dir)
        os.makedirs(target_dir)

class LabelMe:
    def __init__(self, image_dir_, image_name_, label_class_):
        self.label_class = label_class_
        self.shapes = []
        self.image_dir = image_dir_
        self.image_pth = image_name_
        img_mat = cv2.imread(os.path.join(image_dir_, image_name_))
        self.image_height, self.image_width, _ = img_mat.shape

    def shapeAppend(self, label:int, points:np.ndarray):
        left, top, right, bottom = points[0]
        # points = np.array([(left, top), (right, top), (right, bottom), (left, bottom)])
        # points = points.reshape(-1, 2).astype(int).tolist()
        #
        # for point in points:
        #     point[0] = min(max(0, point[0]), self.image_width)
        #     point[1] = min(max(0, point[1]), self.image_height)
        #
        self.shapes.append({
            "label": self.label_class[int(label)],
            "points": [[float(left), float(top)],
                       [float(right), float(bottom)]],
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

# TODO： 兼容目标检测 obb  seg
CLASSES = ["dent", ]
# CLASSES = ["FK01", "FK02", "FK03", "924"]
# CLASSES = ["rotate_0", "rotate_180"]
if __name__ == "__main__":
    model_pth = r"D:\share_dir\impression_detect\workdir\yolov8\det_dent_d2\yolov8n_freeze9_sgd2\weights\yolov8_det01.pt"
    model = YOLO(model_pth)

    save_dir = r"E:\DataSets\dents_det\org_2D\baoma\cutPatches0905\tmp"
    createDir(save_dir)

    img_dir = r"E:\DataSets\dents_det\org_2D\baoma\cutPatches0905\ok"
    img_set = set(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
    has_labeled_img_set = set(f.replace(".json", ".jpg") for f in os.listdir(img_dir) if f.endswith(".json"))
    need_label_img_list = list(img_set.difference(has_labeled_img_set))
    for img_name in need_label_img_list[:]:
        shutil.copyfile(os.path.join(img_dir, img_name), os.path.join(save_dir, img_name))

        det_res = model.predict(os.path.join(save_dir, img_name), device="cpu")

        label_obj = LabelMe(save_dir, img_name, CLASSES)
        # 可能没有检测结果
        for obj in det_res[0].boxes:
            label_obj.shapeAppend(obj.cls.item(), obj.xyxy.numpy())

        label_obj.toJson()
    print("done")