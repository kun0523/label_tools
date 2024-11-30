import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

# TODO: pytorch_grad_cam
from pytorch_grad_cam import AblationCAM


# img = cv2.imread(r"E:\DataSets\dents_det\org_D1\gold_scf\cutPatches640\NG\20240801_00001_B12_QRCODE=A743111237_1_17_C4_FW_A2FW1SD6GSFDH073_A2FW1S47HEHAH030_5256.jpg")
# img = cv2.imread(r"E:\DataSets\dents_det\org_D1\gold_scf\cutPatches640\NG\4_5398.jpg")
# img = cv2.imread(r"E:\yolo_cam\images\puppies.jpg")
img = cv2.imread(r"D:\share_dir\class_proj\mmpretrain\demo\demo.JPEG")
img = cv2.resize(img, (640, 640))
rgb_img = img.copy()
img = np.float32(img)/255


# model = YOLO(r"D:\share_dir\impression_detect\workdir\yolov11\det_dent_gold_scf\yolo11s_sgd3\weights\best.pt")
model = YOLO(r"yolov8n.pt")
model.cpu()

target_layers = [model.model.model[-2], model.model.model[-3], model.model.model[-4],]
# target_layers = [model.model.model[-2],]
cam = EigenCAM(model, target_layers, task='od')

grayscale_cam = cam(rgb_img)[0, :, :]
cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
result = np.hstack([rgb_img, cam_image])
plt.imshow(result)
plt.show()
# cv2.imshow("test", result)
# cv2.waitKey(0)
