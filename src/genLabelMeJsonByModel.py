import os
import cv2
import numpy as np
import json
import shutil
import torch
from tqdm import tqdm

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
# CLASSES = ["a", 'b', 'c', ]
# CLASSES = ["dent", ]
CLASSES = ["FK01", "FK02", "FK03", "924"]
# CLASSES = ["rotate_0", "rotate_180"]
if __name__ == "__main__":
    # model_pth = r"D:\share_dir\impression_detect\workdir\yolov10\dent_det\yolov10s_freeze8_use_sgd8\weights\yolov10_det_05.pt"
    # model_pth = r"E:\AI_edu\trainDemo\PokemonDet\yolo11n_complex_bg2\weights\best.pt"
    model_pth = r"D:\share_dir\pd_mix\workdir\det_words_0308\yolov12s2\weights\best.pt"
    model = YOLO(model_pth)

    save_dir = r"E:\DataSets\pd_mix\pcb_part\0311_new"
    createDir(save_dir)

    img_dir = r"E:\DataSets\pd_mix\pcb_part\0311"
    img_set = set(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
    has_labeled_img_set = set(f.replace(".json", ".jpg") for f in os.listdir(img_dir) if f.endswith(".json"))
    need_label_img_list = list(img_set.difference(has_labeled_img_set))
    for img_name in tqdm(need_label_img_list[:]):
    # for img_name in list(img_set):
        # if img_name != "20240120_00001_P51-L_3_17_C1_FW_A2FW1S3CQV4DA066_A2FW1S4156HAE026_6212.jpg":
        #     continue
        try:
            shutil.copyfile(os.path.join(img_dir, img_name), os.path.join(save_dir, img_name))

            tmp_img_pth = os.path.join(save_dir, img_name)
            if not os.path.isfile(tmp_img_pth):
                print(f"Image not exist: {tmp_img_pth}")
                continue

            det_res = model.predict(tmp_img_pth, device="cuda")
            label_obj = LabelMe(save_dir, img_name, CLASSES)

            # 可能没有检测结果
            for obj in det_res[0]:
                if(obj.boxes.xyxy.device==torch.device("cuda:0")):
                    box = obj.boxes.xyxy.cpu().numpy()
                    cls = obj.boxes.cls.cpu().item()
                else:
                    box = obj.boxes.xyxy.numpy()
                    cls = obj.boxes.cls.item()

                label_obj.shapeAppend(cls, box)

            label_obj.toJson()
        except Exception as e:
            print(">>>>>> Catch Error:")
            print(e)
            continue


    print("done")