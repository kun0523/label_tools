import os
import cv2
import numpy as np
import json
import shutil
from PIL import Image

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
CLASSES = ["crack", ]
# CLASSES = ["FK01", "FK02", "FK03", "924"]
# CLASSES = ["rotate_0", "rotate_180"]
if __name__ == "__main__":
    # model_pth = r"D:\share_dir\pd_edge_crack\workdir\det_crack\yolol_freeze9_sgd_aug2\weights\yolov8_det04.pt"
    # model = YOLO(model_pth)

    save_dir = r"E:\DataSets\edge_crack\cut_patches_0825\tmp"
    createDir(save_dir)

    img_dir = r"E:\DataSets\edge_crack\cut_patches_0825\good01"
    img_set = set(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
    has_labeled_imgs = [f.replace(".json", ".jpg") for f in os.listdir(img_dir) if f.endswith(".json")]
    # need_label_img_list = list(img_set.difference(has_labeled_img_set))
    for img_name in has_labeled_imgs:
        print("Image Name: ", img_name)
        img_mat = cv2.imread(os.path.join(img_dir, img_name))
        img_h, img_w, _ = img_mat.shape

        # det_res = model.predict(os.path.join(save_dir, img_name.replace(".jpg", "_r.jpg")), device="cpu")
        file = img_name.replace(".jpg", ".json")
        save_img_name = img_name.replace(".jpg", "_r.jpg")
        with open(os.path.join(img_dir, file), 'r') as fp:
            content = json.load(fp)

        center = (img_h//2, img_w//2)
        angle = 90
        scale = 1.0
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        rotated_image = cv2.warpAffine(img_mat, rotation_matrix, (img_w, img_h))
        cv2.imwrite(os.path.join(save_dir, save_img_name), rotated_image)
        label_obj = LabelMe(save_dir, save_img_name, CLASSES)

        for shape in content["shapes"]:
            points = np.array(shape["points"], dtype=np.int32)
            # print(">>>: ", points)
            # cv2.rectangle(img_mat, points[0], points[1], (0, 255, 0), 2)

            points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
            rotated_points = rotation_matrix @ points_homogeneous.T
            rotated_points = rotated_points[:2, :].T
            rotated_points = rotated_points.astype(np.int32)
            # print(rotated_points)
            label_obj.shapeAppend(0, rotated_points.reshape(1, -1))
            # cv2.rectangle(rotated_image, rotated_points[0], rotated_points[1], (0,0,255), 2)

        # concate_img = np.hstack([img_mat, rotated_image])
        # cv2.imshow("show", concate_img)
        # cv2.waitKey(0)

        label_obj.toJson()
    print("done")