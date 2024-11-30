import os
import cv2
import PIL
from PIL import Image, ImageEnhance
import numpy as np

IMG_TYPE = ["jpg", "jpeg", "png", "bmp"]  # 除这些格式以外的格式会被过滤掉

def posterize_img(image:Image, levels):
    return image.point(lambda p: (p//(256//levels))*(256//levels))

def get_rect_points(center_x, center_y, rect_w, rect_h, max_w, max_h):
    tl_x = int(max(0, center_x - rect_w/2))
    tl_y = int(max(0, center_y - rect_h/2))
    br_x = int(min(max_w, center_x + rect_w/2))
    br_y = int(min(max_h, center_y + rect_h/2))
    return (tl_x, tl_y), (br_x, br_y)
def test_enhance():
    img_dir = r"E:\DataSets\pol\A45BB_DL"
    save_dir = r"E:\DataSets\pol\A45BB_DL_bright"

    # img_dir = r"E:\DataSets\pol\test\test_images"
    # save_dir = r"E:\DataSets\pol\test\test_images_bright"

    posterize_level = 3
    alpha = 1
    beta = 4
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))

    for parent_dir, sub_dir, file_lst in os.walk(img_dir):
        if("NG" in parent_dir): continue
        for file in file_lst:
            sufix = file.split(".")[-1]
            if sufix not in IMG_TYPE: continue

            cls = parent_dir.split(os.sep)[-1]
            # pil_img = Image.open(os.path.join(parent_dir, file))
            org_img = cv2.imread(os.path.join(parent_dir, file), cv2.IMREAD_GRAYSCALE)
            height, width = org_img.shape

            bright_img = cv2.convertScaleAbs(org_img, alpha, beta)
            bright_img = clahe.apply(bright_img)

            # enhanced_img = posterize_img(pil_img, posterize_level)
            # mat = np.array(enhanced_img)
            # filtered_mat = cv2.medianBlur(mat, 3)
            # enhanced_img = Image.fromarray(filtered_mat)

        # 优化效果可视化
            result = np.hstack([org_img, bright_img])
            # result.show()
            cv2.imshow(f"Class:{cls}", result)
            cv2.waitKey(600)
        cv2.destroyAllWindows()

            # # 保存优化后的图片
            # curr_save_dir = os.path.join(save_dir, cls)
            # if not os.path.exists(curr_save_dir):
            #     os.makedirs(curr_save_dir)
            # cv2.imwrite(os.path.join(curr_save_dir, file), bright_img)

if __name__ == "__main__":

    test_enhance()

    # 根据灰度值 取宝马型号的产品区域  根据灰度值提取异常区域，再做分类
    # 是不是可以取一个标准产品，对新的产品进行归一化，然后做差分，跟标准产品不一致的区域，就是疑似异常，然后再做深度学习检查、
    # 是不是可以根据FPC对产品进行旋转，使得位置进行统一？？

    # window_size = 20
    # img_dir = r"E:\DataSets\dents_det\org_2D\baoma"
    # for file in os.listdir(img_dir):
    #     if not file.endswith(".jpg"): continue
    #
    #     img = cv2.imread(os.path.join(img_dir, file), cv2.IMREAD_GRAYSCALE)
    #     img_h, img_w = img.shape
    #
    #     # 取中心区域
    #     (tlx, tly), (brx, bry) = get_rect_points(img_w/2, img_h/2, window_size, window_size, img_w, img_h)
    #
    #     part = img[tly:bry, tlx:brx]
    #     m = part.mean()
    #     s = part.std()
    #
    #     thresh, res = cv2.threshold(img, m-50*s, 255, cv2.THRESH_BINARY)
    #
    #     # 提取面积最大的区域
    #
    #     con = np.hstack([img, res])
    #     con = cv2.resize(con, None, fx=0.3, fy=0.3)
    #     cv2.imshow("test", con)
    #     cv2.waitKey(0)
    #     print()





    print("done")
