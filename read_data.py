import os
import csv
import torch
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.functional import one_hot


class ISICDataSet(Dataset):

    def __init__(self, data_dir, image_list_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        with open(image_list_file, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            for line in reader:
                image_id = line[0]
                melanoma_val = float(line[1])
                seborrheic_val = float(line[2])

                # 确定标签（0: 痣, 1: 脂溢性角化病, 2: 黑色素瘤）
                if melanoma_val == 1.0:
                    label = 2
                elif seborrheic_val == 1.0:
                    label = 1
                else:
                    label = 0

                image_path = os.path.join(data_dir, f"{image_id}.jpg")
                if os.path.exists(image_path):
                    self.image_paths.append(image_path)
                    self.labels.append(label)
                else:
                    print(f"不存在该图片:{image_path}")

    def __getitem__(self, index):
        # 加载图像并应用变换
        image = Image.open(self.image_paths[index]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        # 标签转为one-hot编码
        label = self.labels[index]
        one_hot_label = one_hot(torch.tensor(label, dtype=torch.long), 3)

        return image, one_hot_label.float()

    def __len__(self):
        return len(self.image_paths)

    def get_labels(self):

        return one_hot(torch.tensor(self.labels, dtype=torch.long), 3).float()
