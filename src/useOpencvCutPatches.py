import os
import re
import cv2
import numpy as np
import time

WINDOW_SIZE = (100, 100)
ORG_FRAME = np.zeros((1024, 1024, 3), dtype=np.uint8)
SHOW_FRAME = np.zeros((1024, 1024, 3), dtype=np.uint8)
IMAGE_NAME = "xxxx.jpg"
WINDOW_NAME = "SHOW_IMG"
SRC_DIR = r"E:\DataSets\edge_crack\original_iqc\org_img"
SAVE_DIR = r"E:\DataSets\edge_crack\cut_patches_0830\crack"
IMAGE_LST = []
IMG_INDEX = 0

# 鼠标悬浮事件
def hover_event(event, x, y, flags, params):
    global ORG_FRAME
    global SHOW_FRAME
    global IMG_INDEX

    h, w, _ = ORG_FRAME.shape
    tl_x = int(max(0, x - WINDOW_SIZE[0] / 2))
    tl_y = int(max(0, y - WINDOW_SIZE[1] / 2))
    br_x = int(min(w, x + WINDOW_SIZE[0] / 2))
    br_y = int(min(h, y + WINDOW_SIZE[1] / 2))
    # cv2.setWindowTitle(WINDOW_NAME, f'{IMAGE_NAME} - Cursor at: ({x}, {y})')

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'Clicked at: ({x}, {y})')
        timestamp = int(time.time())%10000
        save_mat = ORG_FRAME[tl_y:br_y, tl_x:br_x]
        save_path = os.path.join(SAVE_DIR, IMAGE_NAME.replace(".jpg", f"_{timestamp}.jpg"))
        save_res = cv2.imwrite(save_path, save_mat)
        assert save_res, f"Error, image save failed!! {save_path}"

    if event == cv2.EVENT_MOUSEMOVE:
        # 在窗口状态栏显示当前鼠标坐标
        # print(f"tl_x:{tl_x} tl_y:{tl_y} br_x:{br_x} br_y:{br_y}")
        SHOW_FRAME = np.copy(ORG_FRAME)
        cv2.rectangle(SHOW_FRAME, (tl_x, tl_y), (br_x, br_y), (0, 0, 255), 5)
        cv2.imshow(WINDOW_NAME, SHOW_FRAME)

    if event == cv2.EVENT_MBUTTONDOWN:
        ORG_FRAME = cv2.rotate(ORG_FRAME, cv2.ROTATE_90_CLOCKWISE)


if __name__ == "__main__":
    IMAGE_LST = [f for f in os.listdir(SRC_DIR) if f.endswith(".jpg")]

    while True:
        IMAGE_NAME = IMAGE_LST[IMG_INDEX]
        ORG_FRAME = cv2.imread(os.path.join(SRC_DIR, IMAGE_NAME))
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
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
