import os
import cv2
import PIL
from PIL import Image, ImageEnhance
import numpy as np

def posterize_img(image:Image, levels):
    return image.point(lambda p: (p//(256//levels))*(256//levels))

if __name__ == "__main__":

    img_dir = r"E:\DataSets\edge_crack\cut_patches_0828\crack"
    save_dir = r"E:\DataSets\edge_crack\cut_patches_0828\tmp"
    posterize_level = 3
    for file in os.listdir(img_dir):
        if not file.endswith(".jpg"): continue
        # if file != "20240813160703252_6069.jpg": continue
        # img_pth = r"E:\DataSets\edge_crack\cut_patches_0825\crack\20240813160703252_6069.jpg"

        pil_img = Image.open(os.path.join(img_dir, file))
        height = pil_img.height
        width = pil_img.width
        # enhancer = ImageEnhance.Contrast(pil_img)
        # enhancer = ImageEnhance.Brightness(pil_img)
        # enhanced_img = enhancer.enhance(0.8)
        enhanced_img = posterize_img(pil_img, posterize_level)
        # mat = np.array(enhanced_img)
        # filtered_mat = cv2.medianBlur(mat, 3)
        # enhanced_img = Image.fromarray(filtered_mat)

        result = Image.new('RGB', (2*pil_img.width+20, pil_img.height))
        result.paste(pil_img, (0, 0))
        result.paste(enhanced_img, (width+20, 0))
        # result.show()

        result.save(os.path.join(save_dir, file))

    print("done")
