import torch
import os
import cv2
from torch.utils.data import Dataset


class FabVisDataset(Dataset):
    def __init__(self, img_path, lbl_path):
        self.img_path = img_path
        self.lbl_path = lbl_path
        self.images = sorted(os.listdir(img_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        # Concatenate the image name to the path and read it using opencv
        img = cv2.imread(os.path.join(self.img_path, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape # height, width and channels
        img  = torch.tensor(img/255.0, dtype=torch.float32).permute(2, 0, 1)

        boxes = []
        labels = []

        lbl_path = os.path.join(self.lbl_path, img_name.replace(".jpg", ".txt"))
        with open(lbl_path, "r") as f:
            for line in f:
                cls, x_center, y_center, box_w, box_h = map(float, line.split())
                x_center *= w
                y_center *= h
                box_w *= w
                box_h *= h

                boxes.append((x_center - box_w/2, y_center - box_h/2, x_center + box_w/2, y_center + box_h/2))
                labels.append(int(cls))

        target = {"boxes": torch.tensor(boxes, dtype=torch.float32),
                  "labels": torch.tensor(labels, dtype=torch.int64)}

        return img, target