import os
import shutil
import random
from pathlib import Path
import re


class Sample:
    CLASSES = ["crack", "good"]

    def __init__(self, img_pth_, is_train_data_=True):
        self.path = Path(img_pth_)
        self.name = self.path.name
        self.is_train_data = is_train_data_
        self.cls = self.getClassId()

    def getClassId(self):
        parent_dir = self.path.parts[-2]
        assert parent_dir in Sample.CLASSES, f"Error cls:[{parent_dir}] not in CLASSES:[{' '.join(Sample.CLASSES)}]"
        cls_id = Sample.CLASSES.index(parent_dir)
        return cls_id


def createDir(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    else:
        print(f"Target_Dir: {target_dir} already exists!!")
    return target_dir

def createDatasets(sample_obj_lst, save_dir, dir_type="train"):
    # 1. 复制图片  2. 写 txt
    img_save_dir = createDir(os.path.join(save_dir, dir_type))
    label_txt = ""
    for obj in sample_obj_lst:
        # 图片名中不能有空格
        save_name = re.sub("\s", "", obj.name)
        save_pth = os.path.join(img_save_dir, save_name)
        if os.path.exists(save_pth):
            print(f"File: {obj.name} already exists in {img_save_dir}")
            continue
        shutil.copyfile(obj.path, save_pth)
        label_txt += f"{dir_type}/{save_name} {obj.cls}\n"

    with open(os.path.join(save_dir, f"{dir_type}_list.txt"), 'w') as fp:
        fp.write(label_txt)

    return

def splitTrainTest(src_dir, dst_dir):
    cls = src_dir.split(os.sep)[-1]
    train_save_dir = os.path.join(dst_dir, "train", cls)
    val_save_dir = os.path.join(dst_dir, "val", cls)
    createDir(train_save_dir)
    createDir(val_save_dir)

    file_lst = os.listdir(src_dir)
    total_num = len(file_lst)
    train_percent = 0.8
    train_num = int(total_num*train_percent)
    random.shuffle(file_lst)
    for f in file_lst[:train_num]:
        shutil.copyfile(os.path.join(src_dir, f), os.path.join(train_save_dir, f))

    for f in file_lst[train_num:]:
        shutil.copyfile(os.path.join(src_dir, f), os.path.join(val_save_dir, f))



    return

"""
从分类文件夹转为 标注txt
1. 指定root路径，路径下是各个类别图像组成的文件夹
2. 最终将图像保存在 'xxx/train/'  'xxx/val/' 路径下
3. 依次遍历各个类别文件夹，随机划分 'train' 'val'，复制图片的同时，写入txt中 标注信息
"""

IMG_TYPE = ["jpg", "jpeg", "png", "bmp"]  # 除这些格式以外的格式会被过滤掉
if __name__ == "__main__":

    # src_dir = r"E:\DataSets\edge_crack\cut_patches_0905"
    # dst_dir = r"E:\DataSets\edge_crack\classify_ppcls_1106"
    # for parent_dir, sub_dirs, file_lst in os.walk(src_dir):
    #     for sub_dir in sub_dirs:
    #         splitTrainTest(os.path.join(parent_dir, sub_dir), dst_dir)

    src_dir = r"E:\DataSets\edge_crack\cut_patches_0905"
    assert os.path.isdir(src_dir), f"Error src_dir:{src_dir} incorrect!!"

    dst_dir = createDir(r"E:\DataSets\edge_crack\classify_ppcls_1106")
    train_percent = 80
    train_sample_lst, val_sample_lst = [], []
    total_sample_num = 0
    for parent_dir, sub_dir, file_lst in os.walk(src_dir):
        for file in file_lst:
            if not file.split(".")[-1] in IMG_TYPE:
                continue

            total_sample_num += 1
            rand = random.randint(0, 100)
            if rand < train_percent:
                train_sample_lst.append(Sample(os.path.join(parent_dir, file), is_train_data_=True))
            else:
                val_sample_lst.append(Sample(os.path.join(parent_dir, file), is_train_data_=False))

    createDatasets(train_sample_lst, dst_dir, "train")
    createDatasets(val_sample_lst, dst_dir, "val")

    print(f"Total Sample Num: {total_sample_num}")
    print(f"Train Sample Num: {len(train_sample_lst)}  Ratio: {len(train_sample_lst)/total_sample_num*100:.2f}%")
    print(f"Val Sample Num: {len(val_sample_lst)}  Ratio: {len(val_sample_lst)/total_sample_num*100:.2f}%")
