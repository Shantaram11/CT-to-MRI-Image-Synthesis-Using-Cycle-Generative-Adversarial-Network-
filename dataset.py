import os
import cv2
import torch
from torch.utils.data import Dataset


class CTMRIDataset(Dataset):
    def __init__(self, ct_dir, mri_dir):
        self.ct_dir = ct_dir
        self.mri_dir = mri_dir

        # 只取对应后缀的文件
        self.ct_files = sorted([f for f in os.listdir(ct_dir) if f.lower().endswith(".png")])
        self.mri_files = sorted([f for f in os.listdir(mri_dir) if f.lower().endswith(".jpg")])

        print("CT 目录:", self.ct_dir, "文件数:", len(self.ct_files))
        print("MRI目录:", self.mri_dir, "文件数:", len(self.mri_files))

        if len(self.ct_files) == 0:
            raise RuntimeError(f"CT 目录里没有找到 .png 文件: {self.ct_dir}")
        if len(self.mri_files) == 0:
            raise RuntimeError(f"MRI 目录里没有找到 .jpg 文件: {self.mri_dir}")

        assert len(self.ct_files) == len(self.mri_files), \
            f"CT数量 {len(self.ct_files)} != MRI数量 {len(self.mri_files)}"

    def __len__(self):
        return len(self.ct_files)

    def __getitem__(self, idx):
        ct_name = self.ct_files[idx]      # 比如 ct279.png
        mri_name = self.mri_files[idx]    # 比如 mri279.jpg

        ct_path = os.path.join(self.ct_dir, ct_name)
        mri_path = os.path.join(self.mri_dir, mri_name)

        ct = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)
        mri = cv2.imread(mri_path, cv2.IMREAD_GRAYSCALE)

        # ====== 关键调试：如果读不到图，直接报出路径 ======
        if ct is None:
            raise FileNotFoundError(f"无法读取 CT 图像: {ct_path}")
        if mri is None:
            raise FileNotFoundError(f"无法读取 MRI 图像: {mri_path}")
        # ==============================================

        ct = cv2.resize(ct, (256, 256)) / 255.0
        mri = cv2.resize(mri, (256, 256)) / 255.0

        ct = torch.tensor(ct).float().unsqueeze(0)
        mri = torch.tensor(mri).float().unsqueeze(0)

        return ct, mri
