import os
import cv2
import numpy as np
import json
import shutil

from ultralytics import YOLO

"""
使用YOLO模型裁剪ROI区域
"""


def createDir(target_dir):
    if os.path.exists(target_dir):
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

    def shapeAppend(self, label: int, points: np.ndarray):
        left, top, right, bottom = points[0]
        points = np.array([(left, top), (right, top), (right, bottom), (left, bottom)])
        points = points.reshape(-1, 2).astype(int).tolist()

        for point in points:
            point[0] = min(max(0, point[0]), self.image_width)
            point[1] = min(max(0, point[1]), self.image_height)

        self.shapes.append({
            "label": self.label_class[int(label)],
            "points": points,
            "shape_type": "polygon",
            # "shape_type": "rectangle"
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
CLASSES = ["background", "cell"]
if __name__ == "__main__":
    model_pth = r"D:\share_dir\cell_det\workdir\runs\detect\det_s_freeze10_sgd\weights\best.pt"
    model = YOLO(model_pth)

    save_dir = r"E:\DataSets\dents_det\cut_cells\01"
    createDir(save_dir)

    padding = 50
    area_lst = []
    img_dir = r"E:\DataSets\dents_det\org\no_dent"
    for img_name in os.listdir(img_dir):
        if (not img_name.endswith(".jpg")):
            continue

        img_pth = os.path.join(img_dir, img_name)
        img_mat = cv2.imread(img_pth)
        imgH, imgW, _ = img_mat.shape
        det_res = model.predict(img_pth, device="cpu")
        bboxes = det_res[0].boxes.xyxy.cpu().numpy().astype(int)
        for i, (tlx, tly, brx, bry) in enumerate(bboxes):
            tlx = max(0, tlx-padding)
            tly = max(0, tly-padding)
            brx = min(imgW, brx+padding)
            bry = min(imgH, bry+padding)
            width, height = brx-tlx, bry-tly
            area = width*height
            print("Area: ", area)
            # 过滤面积过小的ROI
            if area < 300*1e4:
                continue
            area_lst.append(area)

            save_name = f"{i}_{img_name}"
            cv2.imwrite(os.path.join(save_dir, save_name), img_mat[tly:bry, tlx:brx])

            draw_img = cv2.rectangle(img_mat.copy(), (tlx, tly), (brx, bry), (0,0,255), 4)
            draw_img = cv2.resize(draw_img, None, fx=0.2, fy=0.2)
            cv2.imshow("test", draw_img)
            cv2.waitKey(50)

    t = sorted(area_lst)
    print(t)


    print("done")
