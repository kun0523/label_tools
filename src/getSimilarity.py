import os
import cv2
from collections import defaultdict

import pandas as pd
from ultralytics import YOLO
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F


input_tensor = 0
intermediate_input = 0
intermediate_output = 0


# 定义一个钩子函数，用于捕获中间层的输出
def get_intermediate_feature(module, input, output):
    global intermediate_input
    global intermediate_output
    intermediate_input = input
    intermediate_output = output

def get_input_tensor(module, input, output):
    global input_tensor
    input_tensor = input

features = []
def hook(module, input, output):
    features.append(output)

def load_image(image_pth):
    image = Image.open(image_pth).convert("RGB")
    image = preprocess(image)
    image = image.unsqueeze(0)
    return image

def get_features(image_pth):
    image = load_image(image_pth)
    features.clear()
    with torch.no_grad():
        model(image)
    return features[0].squeeze()

"""
使用模型最近输出的特征向量，计算图片之间的相似度
做聚类，剔除掉非常相似的图像，控制过拟合
"""
if __name__ == "__main__":

    model = models.resnet50(pretrained=True)
    model.eval()

    hook_handle = model.avgpool.register_forward_hook(hook)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 对比 OK中每一张图片与UNK中每一张图片之间的相似度
    rec = defaultdict(list)
    ok_dir = r"E:\DataSets\pol\A45BB_bright\OK"
    unk_dir = r"E:\DataSets\pol\A45BB_bright\UNKNOWN"
    ok_file_names = [ f for f in os.listdir(ok_dir) if f.endswith(".bmp")]
    unk_file_names = [ f for f in os.listdir(unk_dir) if f.endswith(".bmp")]
    for i, ok_file in enumerate(ok_file_names):
        ok_feat = get_features(os.path.join(ok_dir, ok_file))

        for unk_file in unk_file_names[:20]:
            unk_feat = get_features(os.path.join(unk_dir, unk_file))

            similarity = F.cosine_similarity(ok_feat, unk_feat, dim=0)
            print(f"{i}/{len(ok_file_names)} Cosine Similarity: {ok_file} - {unk_file} ~ {similarity.item()}")
            rec[ok_file].append(similarity.item())

    rec_df = pd.DataFrame(rec, index=unk_file_names[:20])
    rec_df.to_csv("similarity2.csv")



    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = YOLO(r"D:\share_dir\pd_edge_crack\workdir\edge_crack_up\yolol_freeze8_sgd_aug2\weights\yolov8_cls02.pt")
    # yolo = YOLO(r"D:\share_dir\pd_edge_crack\workdir\edge_crack_up\yolol_freeze8_sgd_aug2\weights\yolov8_cls02.pt")
    # yolo.model.model[0].conv.register_forward_hook(get_input_tensor)
    #
    # # img_dir = r"E:\DataSets\edge_crack\cut_patches_0830\crack"
    # # img_lst = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
    # img_pth1 = r"E:\DataSets\edge_crack\cut_patches_0830\crack\2024082714300_U1_unk_2696.jpg"
    # img_pth2 = r"E:\DataSets\edge_crack\cut_patches_0830\crack\2024082714292_U1_unk_2690.jpg"
    # img_pth3 = r"E:\DataSets\edge_crack\cut_patches_0830\good\2024082215425_U3_unk_1933.jpg"
    # img_pth4 = r"E:\DataSets\edge_crack\cut_patches_0830\good\2024082215443_U3_unk_1953.jpg"
    #
    # r = yolo(img_pth1)
    #
    # model = model.model.model
    # # 在感兴趣的层上注册钩子
    # hook_handle = model[9].linear.register_forward_hook(get_intermediate_feature)
    #
    # # 创建一个示例输入
    # input_tensor = torch.randn(1, 3, 128, 128)  # 假设输入是 1x28x28 的灰度图像
    #
    # # 进行前向传播
    # output = model(input_tensor)
    #
    # # 打印中间层的输出（特征向量）
    # print(intermediate_output.shape)
    # print(intermediate_output)
    # print(intermediate_input[0].shape)
    # print(intermediate_input[0])

    print("done")