import os
import re
import cv2
import numpy as np
import time

# 小型号  1200*2400
# 大型号  3200*3200
# 全图 1200*1200 缺陷 40*50 ~ 0.1%
FRAME_SCALE_RATIO = [1.0, 0.8, 0.5, 0.3, 0.1]
RATIO_IND = 0
WINDOW_SIZE = (640, 640)
ORG_FRAME = np.zeros((1024, 1024, 3), dtype=np.uint8)
SHOW_FRAME = np.zeros((1024, 1024, 3), dtype=np.uint8)
IMAGE_NAME = "xxxx.jpg"
WINDOW_NAME = "SHOW_IMG"
SRC_DIR = r"E:\DataSets\dents_det\org_D1\gold_scf\NG"
SAVE_DIR = r"E:\DataSets\dents_det\org_D1\gold_scf\cutPatches240\NG"
IMAGE_LST = []
IMG_INDEX = 0

def get_rect_points(center_x, center_y, rect_w, rect_h, max_w, max_h):
    tl_x = int(max(0, center_x - rect_w/2))
    tl_y = int(max(0, center_y - rect_h/2))
    br_x = int(min(max_w, center_x + rect_w/2))
    br_y = int(min(max_h, center_y + rect_h/2))
    return (tl_x, tl_y), (br_x, br_y)

# 鼠标悬浮事件 
def hover_event(event, x, y, flags, params):
    global ORG_FRAME
    global SHOW_FRAME
    global IMG_INDEX
    global RATIO_IND

    h, w, _ = ORG_FRAME.shape
    scale_ratio = FRAME_SCALE_RATIO[RATIO_IND]
    cv2.setWindowTitle(WINDOW_NAME, f'=== IMGID:{IMG_INDEX} + {IMAGE_NAME} + Scale:{scale_ratio} ===')

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'Clicked at: ({x}, {y})')
        timestamp = int(time.time())%10000
        (tlx, tly), (brx, bry) = get_rect_points(x/scale_ratio, y/scale_ratio, WINDOW_SIZE[0], WINDOW_SIZE[1], w, h)
        save_mat = ORG_FRAME[tly:bry, tlx:brx]
        save_path = os.path.join(SAVE_DIR, IMAGE_NAME.replace(".jpg", f"_{timestamp}.jpg"))
        save_res = cv2.imwrite(save_path, save_mat)
        assert save_res, f"Error, image save failed!! {save_path}"

    if event == cv2.EVENT_RBUTTONDOWN:
        RATIO_IND = (RATIO_IND+1)%len(FRAME_SCALE_RATIO)

    if event == cv2.EVENT_MOUSEMOVE:
        # 在窗口状态栏显示当前鼠标坐标
        SHOW_FRAME = np.copy(ORG_FRAME)
        SHOW_FRAME = cv2.resize(SHOW_FRAME, None, fx=scale_ratio, fy=scale_ratio)
        (tlx, tly), (brx, bry) = get_rect_points(x, y, WINDOW_SIZE[0]*scale_ratio, WINDOW_SIZE[1]*scale_ratio, w*scale_ratio, h*scale_ratio)
        cv2.rectangle(SHOW_FRAME, (tlx, tly), (brx, bry), (0, 0, 255), int(5*scale_ratio))
        cv2.imshow(WINDOW_NAME, SHOW_FRAME)

    if event == cv2.EVENT_MBUTTONDOWN:
        ORG_FRAME = cv2.rotate(ORG_FRAME, cv2.ROTATE_90_CLOCKWISE)

'''
使用说明：
1. 配置：源图像路径
2. 配置：保存图像路径
3. 配置：裁剪图片宽高
4. 鼠标左键：保存图像块
5. 鼠标中键：图像顺时针旋转90度
6. 鼠标右键：调整显示图像缩放比例
7. 键盘a键：显示上一张图像
8. 键盘f键：显示下一张图像
'''

if __name__ == "__main__":
    IMAGE_LST = [f for f in os.listdir(SRC_DIR) if f.endswith(".jpg")]

    while True:
        IMAGE_NAME = IMAGE_LST[IMG_INDEX]
        ORG_FRAME = cv2.imread(os.path.join(SRC_DIR, IMAGE_NAME))
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE|cv2.WINDOW_KEEPRATIO|cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow(WINDOW_NAME, ORG_FRAME)
        cv2.setWindowTitle(WINDOW_NAME, f'=== IMGID:{IMG_INDEX} + {IMAGE_NAME} ===')

        cv2.setMouseCallback(WINDOW_NAME, hover_event)
        flag = cv2.waitKey(0)
        if flag == 97:  # a  prev image
            # IMG_INDEX = max(0, (IMG_INDEX-1))
            IMG_INDEX -= 1
            IMG_INDEX = IMG_INDEX if IMG_INDEX >= 0 else len(IMAGE_LST)+IMG_INDEX
            continue
        elif flag == 102:  # f  next image
            IMG_INDEX = (IMG_INDEX+1) % len(IMAGE_LST)
            continue
        elif flag == 27:
            print("done")
            exit()
        else:
            print(f"Wrong input....{flag}")
