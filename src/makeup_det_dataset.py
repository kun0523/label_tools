import os
import cv2
import numpy as np
import random
import time

def get_mask(fg):
    # 抠图
    gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
    thresh, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    return binary

def paste_img_patch(fg_pth, bg, coord=None):
    fg = cv2.imread(fg_pth, cv2.IMREAD_UNCHANGED)
    assert fg.shape[-1] == 4, "Error, Foreground Need PNG image!!"

    bg_h, bg_w, _ = bg.shape
    fg_h, fg_w, _ = fg.shape

    mask = fg[:,:,3]
    mask = mask.astype(np.float32) / 255
    mask = mask[..., np.newaxis].astype(np.uint8)
    mask = np.concatenate([mask, mask, mask], axis=2)

    fg = fg[:,:,:3]

    if coord is None:
        coord = (bg_w//2, bg_h//2)
    elif not isinstance(coord, tuple):
        print("input coord in tuple")
    else:
        pass

    x_begin, y_begin = max(int(coord[0]-fg_w//2), 0), max(0, int(coord[1]-fg_h//2))
    x_end, y_end = min(bg_w, x_begin+fg_w), min(bg_h, y_begin+fg_h)
    actual_w = x_end - x_begin
    actual_h = y_end - y_begin

    container = bg.copy()
    container[y_begin:y_end, x_begin:x_end] = cv2.multiply(container[y_begin:y_end, x_begin:x_end], 1-mask[0:actual_h, 0:actual_w])
    fg[0:actual_h, 0:actual_w] = cv2.multiply(fg[0:actual_h, 0:actual_w], mask[0:actual_h, 0:actual_w])
    cv2.add(container[y_begin:y_end, x_begin:x_end], fg[0:actual_h, 0:actual_w], container[y_begin:y_end, x_begin:x_end])
    return container

def get_background(size=(2048, 2048), img_pth=None):
    container = np.ones((size[0], size[1], 3), dtype=np.uint8)*255
    for i in range(3):
        container[:,:,i] *= random.choice([50, 100, 200, 255])

    if img_pth is not None and os.path.isfile(img_pth):
        img = cv2.imread(img_pth)
        container = cv2.resize(img, size)
    return container

def gen_coord(bg_h, bg_w, num):
    for i in range(num):
        x = np.random.randint(0, bg_w, size=1)
        y = np.random.randint(0, bg_h, size=1)
        yield (x, y)

"""
生成Pokeman目标检测数据集

"""
if __name__ == "__main__":
    orig_dir = r"E:\DataSets\TmpDLTrainData\detection\pokeman_det\origin_img"
    save_dir = r"E:\DataSets\TmpDLTrainData\detection\pokeman_det\new_img"

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    random_size = True
    loop_num = 5
    for i in range(loop_num):
        for file in os.listdir(orig_dir):
            if "bg" in file: continue
            tmp_pth = os.path.join(orig_dir, file)
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            try:
                img = cv2.imread(tmp_pth)
            except Exception as ex:
                print(f"Image File:{tmp_pth} Not support by Opencv-python")
                continue
            print(f"Image: {tmp_pth} shape: {img.shape}")

            if random_size:
                bg_h = bg_w = random.choice([1024, 2048, 4096])
            else:
                bg_h = bg_w = 2048
            gen = gen_coord(bg_h, bg_w, loop_num)
            for coord in gen:
                # container = get_background((bg_h,bg_w),)
                bg_img_pth = random.choice([r"E:\DataSets\TmpDLTrainData\detection\pokeman_det\origin_img\bg1.png", r"E:\DataSets\TmpDLTrainData\detection\pokeman_det\origin_img\bg2.png"])
                container = get_background((bg_h,bg_w), img_pth=bg_img_pth)

                ret = paste_img_patch(tmp_pth, container, coord)
                ret = cv2.resize(ret, (640, 640))
                # cv2.imshow("org", img)
                # cv2.imshow("compress", ret)
                # cv2.waitKey(300)

                timestamp = int(time.time()*1000)
                cv2.imwrite(os.path.join(save_dir, f"{img.shape[0]}_{bg_h}_{timestamp}.jpg"), ret)




    print("done")