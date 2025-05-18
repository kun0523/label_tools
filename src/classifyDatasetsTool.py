import os
import shutil
import random

"""
构建符合特定训练框架的分类数据集
- 输入的文件结构
    - cls1/
    - cls2/

- 输出的文件结构：
- Ultralytics 文件结构
    - train/
        - cls1/
        - cls2/
    - val/
        - cls1/
        - cls2/
        
- PaddleClas 文件结构
    - images/
    - train_list.txt   `images/xxxx.jpg 0`
    - val_list.txt
    - label.txt   `0 OK`
"""

IMG_TYPE = ["jpg", "jpeg", "png", "bmp", "JPG", "JPEG"]  # 除这些格式以外的格式会被过滤掉


class UltralyticsDataSet:
    def __init__(self, src_dir, train_percent=0.8):
        assert os.path.isdir(src_dir), f"Dir Not Found: {src_dir}"
        self.src_dir = src_dir
        self.CLASSES = tuple(os.listdir(self.src_dir))

        self.dst_dir = self.src_dir + "_ultr"
        if os.path.isdir(self.dst_dir):
            shutil.rmtree(self.dst_dir)
        os.makedirs(self.dst_dir)

        self.train_percent = train_percent
        return

    def generate(self):
        train_val_files_map = {"train": [], "val": []}
        random.seed(1014)

        for cls in self.CLASSES:
            src_cls_dir = os.path.join(self.src_dir, cls)
            file_lst = [f for f in os.listdir(src_cls_dir) if f.split(".")[-1] in IMG_TYPE]
            random.shuffle(file_lst)
            total_num = len(file_lst)
            assert total_num > 10, "Error, Image Sample Num <= 10"
            train_file_num = int(total_num * self.train_percent)
            train_val_files_map['train'] = file_lst[:train_file_num]
            train_val_files_map['val'] = file_lst[train_file_num:]

            for train_val_type, files in train_val_files_map.items():
                curr_save_dir = os.path.join(self.dst_dir, train_val_type, cls)
                if not os.path.isdir(curr_save_dir):
                    os.makedirs(curr_save_dir)

                [shutil.copy2(os.path.join(src_cls_dir, f), os.path.join(curr_save_dir, f)) for f in files]
        return


class PaddleClsDataSet:
    def __init__(self, src_dir, train_percent=0.8):
        assert os.path.isdir(src_dir), f"Dir Not Found: {src_dir}"
        self.src_dir = src_dir
        self.CLASSES = tuple(os.listdir(self.src_dir))

        self.dst_dir = self.src_dir + "_pdcls"
        if os.path.isdir(self.dst_dir):
            shutil.rmtree(self.dst_dir)
        os.makedirs(self.dst_dir)

        self.train_percent = train_percent
        with open(os.path.join(self.dst_dir, "label.txt"), "w") as fp:
            for i, cls in enumerate(self.CLASSES):
                fp.write(f"{i} {cls}\n")
        return

    def generate(self):
        random.seed(1014)

        img_save_dir = os.path.join(self.dst_dir, "images")
        if not os.path.isdir(img_save_dir):
            os.makedirs(img_save_dir)

        train_label_file_hd = open(os.path.join(self.dst_dir, "train.txt"), "w")
        val_label_file_hd = open(os.path.join(self.dst_dir, "val.txt"), "w")
        train_val_files_map = {train_label_file_hd: [], val_label_file_hd: []}
        for cls in self.CLASSES:
            src_cls_dir = os.path.join(self.src_dir, cls)
            file_lst = [f for f in os.listdir(src_cls_dir) if f.split(".")[-1] in IMG_TYPE]
            random.shuffle(file_lst)
            total_num = len(file_lst)
            assert total_num > 10, "Error, Image Sample Num <= 10"
            train_file_num = int(total_num * self.train_percent)
            train_val_files_map[train_label_file_hd] = file_lst[:train_file_num]
            train_val_files_map[val_label_file_hd] = file_lst[train_file_num:]

            for label_file_hd, files in train_val_files_map.items():
                for file in files:
                    tmp_src_file = os.path.join(src_cls_dir, file)
                    tmp_dst_file = os.path.join(img_save_dir, file)

                    label_file_hd.write(f"{os.path.join('images', file)} {self.CLASSES.index(cls)}\n")
                    shutil.copy2(tmp_src_file, tmp_dst_file)

        train_label_file_hd.close()
        val_label_file_hd.close()
        return


if __name__ == "__main__":
    # dataset = PaddleClsDataSet(r"E:\DataSets\tuffy_crack\train_1204", train_percent=0.8)
    # dataset.generate()

    dataset = UltralyticsDataSet(r"E:\DataSets\iqc_cracks\train_0308", train_percent=0.8)
    dataset.generate()

    # dataset = UltralyticsDataSet(r"locally110cc4")

    exit(0)
